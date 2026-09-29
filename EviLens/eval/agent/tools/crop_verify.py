"""Local image tools: crop, verify, verify_part, verify_mask.

All operate on images already in the ImageStore (ORIGINAL or a prior IMG_NNN),
produce a new image, register it, and reference it as IMG_NNN in the response —
matching the response_format in tools.json. Coordinates are normalized 0-1000
xyxy (the model's native format for this checkpoint).
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Dict, List, Optional, Callable

from PIL import Image

from ..image_store import compare_lr_image, crop_image, verify_image, norm_xyxy_to_pixels
from . import messages as MSG
from .base import ERR_INFRA, ERR_MODEL, ToolContext, ToolImage, ToolResult


class _ToolInputError(Exception):
    """Internal signal that the model's arguments are bad, translated at the call()
    boundary into ToolResult(ok=False, model_error).

    An exception keeps the helpers' return types unchanged, but it must never
    escape call(): outside, it would be flattened into a generic string and every
    failure would look alike again.
    """


def _bbox_from_args(args: Dict[str, Any]) -> List[float]:
    bbox = args.get("boxes")
    if bbox is None:
        bbox = args.get("box") or args.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return [float(x) for x in bbox]
        except (TypeError, ValueError):
            pass
    raise _ToolInputError(MSG.A1_BAD_BOXES)


def _src_path(ctx: ToolContext, image_id: str) -> str:
    item = ctx.image_store.find(image_id or "ORIGINAL")
    if not item:
        # List every available id, untruncated: lookup is an exact string match, so
        # the model needs to see the exact options.
        avail = MSG.fmt_available([i.image_id for i in ctx.image_store.images])
        raise _ToolInputError(MSG.A2_UNKNOWN_IMAGE_ID.format(got=image_id, available=avail))
    return item.path


def _guard(fn):
    """Translate the three failures that can escape call() into structured results.

      _ToolInputError  bad arguments or image_id  -> A1 / A2
      ValueError       zero-area box from crop    -> A3
      OSError          source image missing       -> A6
    """
    @wraps(fn)
    def inner(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            return fn(self, args, ctx)
        except _ToolInputError as exc:
            return _fail(self.name, str(exc))
        except ValueError as exc:
            if "invalid crop bbox" not in str(exc):
                raise
            bbox = args.get("boxes") or args.get("box") or args.get("bbox")
            which = _degenerate_axis(bbox, ctx, args)
            return _fail(self.name, MSG.A3_EMPTY_BOX.format(box=bbox, which=which))
        except OSError:
            return _fail(self.name, MSG.A6_IMAGE_UNREADABLE.format(
                image_id=str(args.get("image_id") or "ORIGINAL")))
    return inner


def _degenerate_axis(bbox: Any, ctx: ToolContext, args: Dict[str, Any]) -> str:
    """Say whether the width or the height collapsed; "invalid bbox" is not
    actionable."""
    try:
        item = ctx.image_store.find(str(args.get("image_id") or "ORIGINAL"))
        with Image.open(item.path) as im:
            w, h = im.size
        x1, y1, x2, y2 = norm_xyxy_to_pixels(bbox, w, h)
        if x2 <= x1 and y2 <= y1:
            return "width and height"
        return "width" if x2 <= x1 else "height"
    except Exception:  # noqa: BLE001
        return "width or height"


def _fail(tool: str, msg: str) -> ToolResult:
    """Uniform failure return: keeps the envelope, carries a structured error_class."""
    return ToolResult(
        text=f"--- {tool} result ---\n{msg}\n--- end {tool} result ---",
        ok=False,
        error_class=ERR_MODEL,
    )


class CropTool:
    name = "crop"

    @_guard
    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        image_id = str(args.get("image_id") or "ORIGINAL")
        bbox = _bbox_from_args(args)
        img = crop_image(_src_path(ctx, image_id), bbox, padding=0.1, scale=2)
        stored = ctx.image_store.save_pil(img, "crop")
        text = (
            f"--- crop result ---\n"
            f"{stored.image_id}: <image>\n"
            f"source_image_id: {image_id}\n"
            f"--- end crop result ---"
        )
        return ToolResult(text=text, images=[ToolImage(stored.image_id, stored.rel_path, stored.path)],
                          raw={"boxes": bbox, "source_image_id": image_id})


class VerifyTool:
    name = "verify"

    @_guard
    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        image_id = str(args.get("image_id") or "ORIGINAL")
        bbox = _bbox_from_args(args)
        img = verify_image(_src_path(ctx, image_id), bbox)
        stored = ctx.image_store.save_pil(img, "verify")
        text = (
            f"--- verify result ---\n"
            f"{stored.image_id}: <image>\n"
            f"source_image_id: {image_id}\n"
            f"--- end verify result ---"
        )
        return ToolResult(text=text, images=[ToolImage(stored.image_id, stored.rel_path, stored.path)],
                          raw={"boxes": bbox, "source_image_id": image_id})


class VerifyPartTool:
    name = "verify_part"

    @_guard
    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        image_id = str(args.get("image_id") or "ORIGINAL")
        bbox = _bbox_from_args(args)
        img = crop_image(_src_path(ctx, image_id), bbox, padding=0.0, scale=2)
        stored = ctx.image_store.save_pil(img, "verify_part")
        text = (
            f"--- verify_part result ---\n"
            f"{stored.image_id}: <image>\n"
            f"source_image_id: {image_id}\n"
            f"--- end verify_part result ---"
        )
        return ToolResult(text=text, images=[ToolImage(stored.image_id, stored.rel_path, stored.path)],
                          raw={"boxes": bbox, "source_image_id": image_id})


class CompareLRTool:
    """spot-the-difference only: put a left-panel region beside its right-panel
    counterpart.

    A difference exists only between two versions of the same location, and in
    whole-image 0-1000 coordinates the counterpart of (x,y) is (x+500,y). A single
    crop covering both would have to be more than half the image wide, i.e. no
    zoom at all -- so comparison is impossible with the other tools.
    """

    name = "compare_lr"

    @_guard
    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        image_id = str(args.get("image_id") or "ORIGINAL")
        bbox = _bbox_from_args(args)
        if max(bbox[0], bbox[2]) > 500:
            raise _ToolInputError(MSG.A7_NOT_LEFT_PANEL.format(x_max=max(bbox[0], bbox[2])))
        img = compare_lr_image(_src_path(ctx, image_id), bbox, padding=0.1)
        stored = ctx.image_store.save_pil(img, "compare_lr")
        right = [bbox[0] + 500, bbox[1], bbox[2] + 500, bbox[3]]
        text = (
            f"--- compare_lr result ---\n"
            f"{stored.image_id}: <image>\n"
            f"source_image_id: {image_id}\n"
            f"left_box: {[round(v) for v in bbox]}\n"
            f"right_box: {[round(v) for v in right]}\n"
            f"layout: left region | magenta divider | right region (same size)\n"
            f"--- end compare_lr result ---"
        )
        return ToolResult(text=text, images=[ToolImage(stored.image_id, stored.rel_path, stored.path)],
                          raw={"boxes": bbox, "right_boxes": right, "source_image_id": image_id})


def _points_from_args(args: Dict[str, Any], key: str) -> List[List[float]]:
    pts = args.get(key) or []
    out: List[List[float]] = []
    if isinstance(pts, (list, tuple)):
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                out.append([float(p[0]), float(p[1])])
    return out


class VerifyMaskTool:
    """Run local SAM3 from box + optional points and overlay the mask in cyan.

    segment_fn(image_path, box_px, pos_px, neg_px) -> bool mask (H,W). When SAM
    is unavailable (segment_fn is None), degrades to a plain verify box overlay
    with a note, so the loop keeps going.
    """
    name = "verify_mask"

    def __init__(self, segment_fn: Optional[Callable] = None):
        self.segment_fn = segment_fn

    def _degraded(self, ctx: ToolContext, src: str, bbox, image_id: str, note: str) -> ToolResult:
        """Fall back to a drawn box when SAM is unavailable, saying explicitly that
        it is not a mask."""
        img = verify_image(src, bbox)
        stored = ctx.image_store.save_pil(img, "verify_mask")
        text = (
            f"--- verify_mask result ---\n"
            f"{stored.image_id}: <image>\n"
            f"source_image_id: {image_id}\n"
            f"{note}\n"
            f"--- end verify_mask result ---"
        )
        return ToolResult(
            text=text,
            images=[ToolImage(stored.image_id, stored.rel_path, stored.path)],
            raw={"boxes": bbox, "sam": False},
            ok=False,
            error_class=ERR_INFRA,
        )


    @_guard
    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        image_id = str(args.get("image_id") or "ORIGINAL")
        bbox = _bbox_from_args(args)
        pos = _points_from_args(args, "positive_points")
        neg = _points_from_args(args, "negative_points")
        src = _src_path(ctx, image_id)

        if self.segment_fn is None:
            # No segment_fn wired in; say so, or the model assumes it sees a mask.
            return self._degraded(ctx, src, bbox, image_id, MSG.F1_SAM_DISABLED)

        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            base = im.copy()
        box_px = norm_xyxy_to_pixels(bbox, w, h)
        pos_px = [[p[0] / 1000.0 * w, p[1] / 1000.0 * h] for p in pos]
        neg_px = [[p[0] / 1000.0 * w, p[1] / 1000.0 * h] for p in neg]
        try:
            mask = self.segment_fn(src, box_px, pos_px, neg_px)  # bool (H,W)
        except Exception:  # noqa: BLE001
            # SAM call failed; again, do not imply a mask was produced.
            return self._degraded(ctx, src, bbox, image_id, MSG.F2_SAM_FAILED)

        overlay = _cyan_overlay(base, mask)
        stored = ctx.image_store.save_pil(overlay, "verify_mask")
        text = (
            f"--- verify_mask result ---\n"
            f"{stored.image_id}: <image>\n"
            f"source_image_id: {image_id}\n"
            f"--- end verify_mask result ---"
        )
        return ToolResult(text=text, images=[ToolImage(stored.image_id, stored.rel_path, stored.path)],
                          raw={"boxes": bbox, "n_pos": len(pos), "n_neg": len(neg), "sam": True})


def _cyan_overlay(base: Image.Image, mask) -> Image.Image:
    """Blend a cyan mask over the base image."""
    import numpy as np

    arr = np.array(base).astype(np.float32)
    m = np.asarray(mask)
    if m.shape[:2] != arr.shape[:2]:
        from PIL import Image as _I
        m_img = _I.fromarray((m.astype("uint8") * 255)).resize((arr.shape[1], arr.shape[0]), _I.NEAREST)
        m = np.array(m_img) > 127
    cyan = np.array([0, 255, 255], dtype=np.float32)
    alpha = 0.5
    sel = m.astype(bool)
    arr[sel] = (1 - alpha) * arr[sel] + alpha * cyan
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))
