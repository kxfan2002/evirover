"""Segmentation scorer: SAM3 turns the model's box+points into a mask, scored by
mask IoU against the GT mask (with IoU@0.5 aggregated later).

The model answers in 0-1000 normalized coords; we scale to pixels using the actual
image size, run SAM3, then compare to the GT boolean mask (resized if needed).
"""
import math

import numpy as np
from PIL import Image

from .. import parsing, config
from . import register, ScoreResult


def _mask_iou(pred: np.ndarray, gt: np.ndarray):
    """Return (iou, inter_px, union_px). inter/union are needed for cIoU
    (cumulative IoU = Σinter / Σunion over all samples)."""
    if pred.shape != gt.shape:
        # Resize pred to GT shape via nearest-neighbor.
        pred_img = Image.fromarray(pred.astype(np.uint8) * 255)
        pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
        pred = np.array(pred_img) > 127
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    iou = float(inter) / float(union) if union > 0 else 0.0
    return iou, inter, union


def _scale(coords, w, h):
    """Scale a list of [x,y] (or a 4-box) from 0-1000 to pixels."""
    return [c / 1000.0 * (w if i % 2 == 0 else h) for i, c in enumerate(coords)]


def _smart_resize(h, w, factor, min_pixels, max_pixels):
    """Qwen2-VL / Qwen2.5-VL smart_resize: the (h, w) the vision encoder actually
    sees.

    A model answering in absolute pixels is using this grid, not the original size.
    """
    hb = round(h / factor) * factor
    wb = round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = math.floor(h / beta / factor) * factor
        wb = math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


@register("segmentation")
def score(answer_text, sample, ctx=None) -> ScoreResult:
    coord_order = (ctx or {}).get("coord_order", "xy")
    parsed = parsing.parse_seg(answer_text, coord_order=coord_order)
    if parsed is None:
        return ScoreResult(score=0.0, parse_ok=False, components={"iou": 0.0}, pred=None)
    box, pos, neg = parsed

    # ctx must provide a `segment` callable and mask loader; if SAM is disabled,
    # we cannot score seg, so report parse-only.
    if ctx is None or ctx.get("segment") is None:
        return ScoreResult(
            score=0.0, parse_ok=True,
            components={"iou": 0.0, "sam_skipped": True}, pred={"boxes": box},
        )

    gt_mask = ctx["load_gt_mask"](sample.gt.get("mask_path"))
    if gt_mask is None:
        return ScoreResult(
            score=0.0, parse_ok=True,
            components={"iou": 0.0, "gt_missing": True}, pred={"boxes": box},
        )

    # Determine pixel size from the image (authoritative) for prompt scaling.
    with Image.open(sample.image_path) as im:
        w, h = im.size

    # coord_space=pixel: the answer is in absolute pixels on the smart_resize grid.
    # Convert back to 0-1000; everything after that is identical to norm1000.
    if (ctx or {}).get("coord_space") == "pixel":
        cfg = ctx.get("resize_cfg")
        if not cfg:
            raise ValueError("coord_space=pixel but ctx has no resize_cfg")
        rh, rw = _smart_resize(h, w, cfg["factor"], cfg["min_pixels"], cfg["max_pixels"])

        def _to_norm(c):
            return [v / (rw if i % 2 == 0 else rh) * 1000.0 for i, v in enumerate(c)]

        box = _to_norm(box)
        pos = [_to_norm(p) for p in pos]
        neg = [_to_norm(p) for p in neg]

    box_px = _scale(box, w, h)
    pos_px = [_scale(p, w, h) for p in pos]
    neg_px = [_scale(p, w, h) for p in neg]

    try:
        pred_mask = ctx["segment"](sample.image_path, box_px, pos_px, neg_px)
    except Exception as e:  # noqa: BLE001
        # SAM failed: empty prediction vs non-empty GT. inter=0, union=|GT|
        # so this sample correctly drags cIoU down (not silently dropped).
        return ScoreResult(
            score=0.0, parse_ok=True,
            components={"iou": 0.0, "sam_error": str(e)[:200],
                        "inter": 0, "union": int(gt_mask.sum())},
            pred={"boxes": box},
        )

    iou, inter, union = _mask_iou(pred_mask, gt_mask)
    return ScoreResult(
        score=iou,
        parse_ok=True,
        components={"iou": iou, "hit@0.5": float(iou >= config.IOU_THRESHOLD),
                    "inter": inter, "union": union},
        pred={"boxes": box, "n_pos": len(pos), "n_neg": len(neg)},
    )
