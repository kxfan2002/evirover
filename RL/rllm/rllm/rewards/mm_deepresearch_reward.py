"""Task-specific scalar rewards for multimodal DeepResearch RL.

This module mirrors the scoring conventions of the EviLens evaluation harness,
but is self-contained here:

- grounding_bbox: single bbox IoU
- spot_diff: greedy multi-box F1 with IoU>=0.5 and left-panel filtering
- segmentation: SAM3 mask IoU from predicted box/points vs GT mask
- counting: exact integer match

The workflow should train on RewardOutput.reward. RewardOutput.is_correct is kept
only for logging / optional rejection sampling.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PIL import Image

from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
INT_RE = re.compile(r"-?\d+")
IOU_THRESHOLD = float(os.getenv("MM_REWARD_IOU_THRESHOLD", "0.5"))
# Dense spatial reward shaping for grounding_bbox (and spot_diff matching).
#   "iou"  -> default; reward = raw IoU (dead-zero for any non-overlapping box).
#   "giou" -> reward = (GIoU+1)/2; continuous & directional even for disjoint
#             boxes, so all-miss groups get non-degenerate std -> gradient flows.
#   "diou" -> reward = (DIoU+1)/2; adds center-distance term on top of IoU.
# Only affects TRAINING reward; the eval harness always scores raw IoU. Default
# keeps v8 behavior byte-identical so this is a clean env-gated A/B lever.
GROUNDING_SHAPING = os.getenv("MM_REWARD_GROUNDING_SHAPING", "iou").strip().lower()
# Soft matching for spot_diff F1.
#   "hard"     -> default; a pred box counts as a TP only at IoU>=IOU_THRESHOLD
#                 (0.5). Near-region boxes (IoU 0.1-0.5) contribute 0, so a hard
#                 question where NO rollout clears 0.5 collapses the whole prompt
#                 group to F1=0 -> std=0 -> zero GRPO gradient.
#   "soft_iou" -> greedy 1-1 matching keeps any pair with IoU>=SPOTDIFF_FLOOR, and
#                 each matched pair contributes its IoU (not 1). A 0.3 box adds 0.3,
#                 so tightening it raises reward -> gradient. Restores intra-group
#                 variance on the hard (all-<0.5) questions the hard rule zeroed out.
# Only affects TRAINING reward; the "f1"/"precision"/"recall"/"tp" components stay
SPOTDIFF_SHAPING = os.getenv("MM_REWARD_SPOTDIFF_SHAPING", "hard").strip().lower()
SPOTDIFF_FLOOR = float(os.getenv("MM_REWARD_SPOTDIFF_FLOOR", "0.1"))
#
#
#   floor 0.3 => 23.3%, floor 0.4 => 41.9%, hard(0.5) => 51.2%;
#
SPOTDIFF_SOFT_POW = float(os.getenv("MM_REWARD_SPOTDIFF_SOFT_POW", "1.0"))
SEG_REQUIRE_POINTS = os.getenv("MM_REWARD_SEG_REQUIRE_POINTS", "1").lower() in {"1", "true", "yes"}


@dataclass
class ScoreResult:
    score: float
    parse_ok: bool
    components: dict[str, Any] = field(default_factory=dict)
    pred: Any = None


def extract_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    text = text.strip()
    return text if text else None


def _loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        try:
            return ast.literal_eval(s)
        except Exception:  # noqa: BLE001
            return None


def _all_numbers(x: Any, n: int | None = None) -> bool:
    if not isinstance(x, (list, tuple)):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        for v in x:
            float(v)
        return True
    except Exception:  # noqa: BLE001
        return False


def _as_float_box(box: Any) -> Optional[list[float]]:
    if _all_numbers(box, 4):
        return [float(v) for v in box]
    return None


def parse_bbox(answer: str | None) -> Optional[list[float]]:
    if answer is None:
        return None
    obj = _loads(answer)
    box = _as_float_box(obj)
    if box is not None:
        return box
    nums = FLOAT_RE.findall(answer)
    if len(nums) >= 4:
        return [float(v) for v in nums[:4]]
    return None


def parse_bbox_list(answer: str | None) -> Optional[list[list[float]]]:
    if answer is None:
        return None
    obj = _loads(answer)
    boxes: list[list[float]] = []
    if isinstance(obj, (list, tuple)):
        for b in obj:
            box = _as_float_box(b)
            if box is not None:
                boxes.append(box)
        if not boxes:
            box = _as_float_box(obj)
            if box is not None:
                boxes = [box]
    return boxes or None


def parse_int(answer: str | None) -> Optional[int]:
    if answer is None:
        return None
    obj = _loads(answer)
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        return int(round(obj))
    m = INT_RE.search(answer)
    if not m:
        return None
    try:
        return int(m.group())
    except Exception:  # noqa: BLE001
        return None


def parse_seg(answer: str | None) -> Optional[tuple[list[float], list[list[float]], list[list[float]]]]:
    """Parse segmentation answer: {boxes:[...], positive_points:[[x,y]...], ...}."""
    if answer is None:
        return None
    obj = _loads(answer)
    if isinstance(obj, dict):
        raw_box = obj.get("boxes", obj.get("box", obj.get("bbox")))
        box = _as_float_box(raw_box)
        if box is None:
            return _parse_seg_lenient(answer)

        def pts(key: str) -> list[list[float]]:
            out: list[list[float]] = []
            raw = obj.get(key) or []
            if isinstance(raw, (list, tuple)):
                for p in raw:
                    if _all_numbers(p, 2):
                        out.append([float(p[0]), float(p[1])])
            return out

        return box, pts("positive_points"), pts("negative_points")
    return _parse_seg_lenient(answer)


def _parse_seg_lenient(answer: str) -> Optional[tuple[list[float], list[list[float]], list[list[float]]]]:
    box_m = re.search(r'"?(?:boxes|box|bbox)"?\s*:\s*\[([^\]]*)\]', answer)
    if not box_m:
        # final fallback: first four numbers are the box
        box = parse_bbox(answer)
        return (box, [], []) if box is not None else None
    box_nums = FLOAT_RE.findall(box_m.group(1))
    if len(box_nums) < 4:
        return None
    box = [float(v) for v in box_nums[:4]]

    def pts(key: str) -> list[list[float]]:
        m = re.search(
            r'"?' + re.escape(key) + r'"?\s*:\s*(.*?)(?="?(?:positive_points|negative_points|boxes|box|bbox)"?\s*:|\}|$)',
            answer,
            re.DOTALL,
        )
        if not m:
            return []
        nums = [float(v) for v in FLOAT_RE.findall(m.group(1))]
        return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]

    return box, pts("positive_points"), pts("negative_points")


def iou_2d(box1: Sequence[float], box2: Sequence[float]) -> float:
    if not _all_numbers(box1, 4) or not _all_numbers(box2, 4):
        return 0.0
    x1, y1, x2, y2 = map(float, box1)
    X1, Y1, X2, Y2 = map(float, box2)
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = a1 + a2 - inter
    return inter / union if union > 1e-12 else 0.0


def _iou_union(box1: Sequence[float], box2: Sequence[float]) -> tuple[float, float]:
    """Return (iou, union_area) for two xyxy boxes; (0,0) if either is invalid."""
    if not _all_numbers(box1, 4) or not _all_numbers(box2, 4):
        return 0.0, 0.0
    x1, y1, x2, y2 = map(float, box1)
    X1, Y1, X2, Y2 = map(float, box2)
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = a1 + a2 - inter
    iou = inter / union if union > 1e-12 else 0.0
    return iou, union


def giou_2d(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Generalized IoU in [-1, 1]. Falls to -1 as boxes grow far apart, so it
    stays differentiable (and monotone in overlap+proximity) even when the
    boxes are disjoint and plain IoU is a flat zero."""
    iou, union = _iou_union(box1, box2)
    if union <= 0.0:
        return -1.0
    x1, y1, x2, y2 = map(float, box1)
    X1, Y1, X2, Y2 = map(float, box2)
    cx1, cy1 = min(x1, X1), min(y1, Y1)
    cx2, cy2 = max(x2, X2), max(y2, Y2)
    c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    if c_area <= 1e-12:
        return iou
    return iou - (c_area - union) / c_area


def diou_2d(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Distance-IoU in (-1, 1]: IoU minus (center-distance / enclosing-diagonal)^2.
    Directional toward center alignment even when boxes are disjoint."""
    iou, union = _iou_union(box1, box2)
    if union <= 0.0:
        return -1.0
    x1, y1, x2, y2 = map(float, box1)
    X1, Y1, X2, Y2 = map(float, box2)
    cx_a, cy_a = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx_b, cy_b = (X1 + X2) / 2.0, (Y1 + Y2) / 2.0
    rho2 = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2
    ex1, ey1 = min(x1, X1), min(y1, Y1)
    ex2, ey2 = max(x2, X2), max(y2, Y2)
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    if c2 <= 1e-12:
        return iou
    return iou - rho2 / c2


def shaped_bbox_score(pred: Sequence[float], gt: Sequence[float], iou: float) -> float:
    """Map the configured spatial metric to a [0,1] training reward.
    GIoU/DIoU live in [-1,1]; (m+1)/2 keeps the full range positive so a far-off
    box still earns a non-zero, ranked reward (no clamp-induced dead-zone)."""
    if GROUNDING_SHAPING in ("hit", "01", "binary"):
        # Binary threshold reward: 1.0 iff IoU>=IOU_THRESHOLD else 0.0 (no partial
        # credit). The spot_diff analog is SPOTDIFF_SHAPING="hard" (F1 over boxes
        # matched at IoU>=IOU_THRESHOLD).
        return float(iou >= IOU_THRESHOLD)
    if GROUNDING_SHAPING == "giou":
        return (giou_2d(pred, gt) + 1.0) / 2.0
    if GROUNDING_SHAPING == "diou":
        return (diou_2d(pred, gt) + 1.0) / 2.0
    return iou


def _resolve_path(task_info: dict[str, Any], path_value: Any) -> Optional[str]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    data_root = task_info.get("data_root") or os.getenv("MM_DATA_ROOT") or os.getenv("DATA_ROOT")
    if data_root:
        return str(Path(str(data_root)) / path)
    return str(path)


def load_gt_mask(mask_path: str | None) -> Optional[np.ndarray]:
    if not mask_path or not os.path.isfile(mask_path):
        return None
    im = Image.open(mask_path)
    arr = np.array(im)
    if arr.ndim == 2:
        return arr > 127
    return arr[..., 0] > 127


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, int, int]:
    if pred.shape != gt.shape:
        pred_img = Image.fromarray(pred.astype(np.uint8) * 255)
        pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
        pred = np.array(pred_img) > 127
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    iou = float(inter) / float(union) if union > 0 else 0.0
    return iou, inter, union


def scale_norm_to_px(coords: Sequence[float], w: int, h: int) -> list[float]:
    return [float(c) / 1000.0 * (w if i % 2 == 0 else h) for i, c in enumerate(coords)]


_SAM_LOCK = threading.Lock()
_SAM_BACKEND: Any = None


def _sam_checkpoint() -> str:
    candidates = [
        os.getenv("SAM3_CHECKPOINT"),
        os.path.expanduser("~/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"),
        os.path.expanduser("~/.cache/huggingface/sam3/sam3.pt"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    # Deterministic path to fail on, so the error names SAM3_CHECKPOINT.
    return candidates[0] or "sam3.pt (set SAM3_CHECKPOINT)"


def _ensure_sam_repo() -> None:
    candidates = [
        os.getenv("SAM3_REPO"),
    ]
    for repo in candidates:
        if repo and os.path.isdir(repo) and repo not in sys.path:
            sys.path.insert(0, repo)
            return


class _Sam3Backend:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        _ensure_sam_repo()
        import torch  # local import so non-seg runs do not require torch/SAM
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        #   RuntimeError: Input type (CUDABFloat16Type) and weight type (torch.FloatTensor)
        #
        dev_str = str(device)
        self.device_index: Optional[int] = None
        if dev_str.startswith("cuda") and torch.cuda.is_available():
            self.device_index = int(dev_str.split(":", 1)[1]) if ":" in dev_str else torch.cuda.current_device()
            torch.cuda.set_device(self.device_index)
            build_device = "cuda"
        else:
            build_device = "cpu"
        self.autocast_device = "cuda" if build_device == "cuda" else "cpu"
        torch.autocast(self.autocast_device, dtype=torch.bfloat16).__enter__()
        if build_device == "cuda":
            if torch.cuda.get_device_properties(self.device_index or 0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.model = build_sam3_image_model(
            device=build_device,
            checkpoint_path=checkpoint,
            load_from_HF=False,
            compile=False,
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(self.model, device=build_device)

    def segment(
        self,
        image_path: str,
        box_px: Sequence[float],
        pos_pts_px: Optional[list[list[float]]] = None,
        neg_pts_px: Optional[list[list[float]]] = None,
    ) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        empty = np.zeros((h, w), dtype=bool)

        x1, y1, x2, y2 = box_px
        x1 = max(0.0, min(float(x1), w - 1))
        y1 = max(0.0, min(float(y1), h - 1))
        x2 = max(0.0, min(float(x2), w - 1))
        y2 = max(0.0, min(float(y2), h - 1))
        if x2 <= x1 or y2 <= y1:
            return empty

        kwargs: dict[str, Any] = {
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "multimask_output": False,
        }
        pos_pts_px = pos_pts_px or []
        neg_pts_px = neg_pts_px or []
        if pos_pts_px or neg_pts_px:
            pts = [[float(p[0]), float(p[1])] for p in pos_pts_px + neg_pts_px]
            labels = [1] * len(pos_pts_px) + [0] * len(neg_pts_px)
            kwargs["point_coords"] = np.array(pts, dtype=np.float32)
            kwargs["point_labels"] = np.array(labels, dtype=np.int32)

        #   RuntimeError: Input type (CUDABFloat16Type) and weight type (torch.FloatTensor)
        torch = self.torch
        from contextlib import nullcontext
        dev_ctx = (torch.cuda.device(self.device_index)
                   if self.device_index is not None else nullcontext())
        with dev_ctx:
            with torch.autocast(self.autocast_device, dtype=torch.bfloat16):
                state = self.processor.set_image(image)
                masks, _scores, _ = self.model.predict_inst(state, **kwargs)
        mask = np.asarray(masks[0])
        return mask if mask.dtype == bool else mask > 0.5


def segment_with_sam3(image_path: str, box_px: Sequence[float], pos_px=None, neg_px=None) -> np.ndarray:
    global _SAM_BACKEND
    if os.getenv("MM_REWARD_DISABLE_SAM", "0").lower() in {"1", "true", "yes"}:
        raise RuntimeError("SAM3 disabled by MM_REWARD_DISABLE_SAM")
    device = os.getenv("MM_REWARD_SAM_DEVICE") or f"cuda:{os.getenv('VDR_SAM_GPU', '0')}"
    with _SAM_LOCK:
        if _SAM_BACKEND is None:
            print(f"[sam3] loading backend -> {device} (pid={os.getpid()})", flush=True)
            _SAM_BACKEND = _Sam3Backend(checkpoint=_sam_checkpoint(), device=device)
        return _SAM_BACKEND.segment(image_path, box_px, pos_px, neg_px)


def score_grounding(answer_text: str | None, task_info: dict[str, Any]) -> ScoreResult:
    pred = parse_bbox(answer_text)
    gt = task_info.get("bbox_1000_xyxy") or task_info.get("answer")
    if pred is None:
        return ScoreResult(0.0, False, {"iou": 0.0}, None)
    iou = iou_2d(pred, gt)
    reward = shaped_bbox_score(pred, gt, iou)
    comp = {"iou": iou, "hit@0.5": float(iou >= IOU_THRESHOLD)}
    if GROUNDING_SHAPING != "iou":
        comp["shaping"] = GROUNDING_SHAPING
        comp["shaped"] = reward
    return ScoreResult(reward, True, comp, pred)


def score_counting(answer_text: str | None, task_info: dict[str, Any]) -> ScoreResult:
    pred = parse_int(answer_text)
    gt = task_info.get("answer")
    if pred is None or gt is None:
        return ScoreResult(0.0, False, {}, pred)
    try:
        gt_int = int(gt)
    except Exception:  # noqa: BLE001
        return ScoreResult(0.0, False, {"gt_parse_error": True}, pred)
    correct = float(pred == gt_int)
    return ScoreResult(correct, True, {"correct": correct, "abs_err": abs(pred - gt_int)}, pred)


def _valid_left_panel(box: Sequence[float]) -> bool:
    return bool(_all_numbers(box, 4) and float(box[0]) <= 500 and float(box[2]) <= 500)


def score_spot_diff(answer_text: str | None, task_info: dict[str, Any]) -> ScoreResult:
    preds = parse_bbox_list(answer_text)
    gts = task_info.get("bbox_1000_xyxy") or []
    n_gt = len(gts) if isinstance(gts, list) else 0
    if preds is None:
        return ScoreResult(
            0.0,
            False,
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_gt": n_gt, "n_pred": 0},
            None,
        )
    valid = [p for p in preds if _valid_left_panel(p)]
    # Greedy IoU-descending 1-1 matching. `floor` is the *candidacy* cut: hard mode
    # only considers pairs >=0.5 (byte-identical to the old behavior), soft mode lets
    # near-region pairs (>=SPOTDIFF_FLOOR) in. From the SAME matched pairs we tally
    # hard_tp (>=0.5, binary -> the anchored metric) and soft_tp (sum of IoU -> the
    # graded training signal).
    floor = SPOTDIFF_FLOOR if SPOTDIFF_SHAPING == "soft_iou" else IOU_THRESHOLD
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(valid):
        for gi, g in enumerate(gts):
            iou = iou_2d(p, g)
            if iou >= floor:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    hard_tp = 0
    soft_tp = 0.0
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        soft_tp += iou ** SPOTDIFF_SOFT_POW
        if iou >= IOU_THRESHOLD:
            hard_tp += 1
    n_pred = len(valid)

    def _f1(tp: float) -> float:
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gt if n_gt else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    hard_f1 = _f1(hard_tp)
    soft_f1 = _f1(soft_tp)
    reward = soft_f1 if SPOTDIFF_SHAPING == "soft_iou" else hard_f1

    # Components stay anchored to the hard@0.5 metric (precision/recall/f1/tp) so the
    # eval curve is unchanged; soft_f1 is reported alongside for diagnosis only.
    precision = hard_tp / n_pred if n_pred else 0.0
    recall = hard_tp / n_gt if n_gt else 0.0
    return ScoreResult(
        reward,
        True,
        {
            "precision": precision,
            "recall": recall,
            "f1": hard_f1,
            "soft_f1": soft_f1,
            "tp": hard_tp,
            "n_gt": n_gt,
            "n_pred": n_pred,
        },
        valid,
    )


def score_segmentation(answer_text: str | None, task_info: dict[str, Any]) -> ScoreResult:
    parsed = parse_seg(answer_text)
    if parsed is None:
        return ScoreResult(0.0, False, {"iou": 0.0}, None)
    box, pos, neg = parsed

    #
    if SEG_REQUIRE_POINTS and (len(pos) < 1 or len(neg) < 1):
        return ScoreResult(
            0.0,
            False,
            {"iou": 0.0, "format_gate": "need >=1 positive and >=1 negative point"},
            {"boxes": box, "n_pos": len(pos), "n_neg": len(neg)},
        )

    image_path = _resolve_path(task_info, task_info.get("image_path"))
    mask_path = _resolve_path(task_info, task_info.get("mask_path") or task_info.get("mask_path_benchmark"))
    if not image_path or not os.path.isfile(image_path):
        return ScoreResult(0.0, True, {"iou": 0.0, "image_missing": True}, {"boxes": box})
    gt_mask = load_gt_mask(mask_path)
    if gt_mask is None:
        return ScoreResult(0.0, True, {"iou": 0.0, "gt_missing": True}, {"boxes": box})

    with Image.open(image_path) as im:
        w, h = im.size
    box_px = scale_norm_to_px(box, w, h)
    pos_px = [scale_norm_to_px(p, w, h) for p in pos]
    neg_px = [scale_norm_to_px(p, w, h) for p in neg]

    sam_fallback = False
    sam_error = ""
    try:
        pred_mask = segment_with_sam3(image_path, box_px, pos_px, neg_px)
    except Exception as exc:  # noqa: BLE001
        #
        #
        sam_fallback = True
        sam_error = str(exc)[:200]
        pred_mask = np.zeros_like(gt_mask, dtype=bool)
        x0, y0, x1, y1 = (int(round(v)) for v in box_px)
        y0 = max(0, min(y0, gt_mask.shape[0]))
        y1 = max(0, min(y1, gt_mask.shape[0]))
        x0 = max(0, min(x0, gt_mask.shape[1]))
        x1 = max(0, min(x1, gt_mask.shape[1]))
        if y1 > y0 and x1 > x0:
            pred_mask[y0:y1, x0:x1] = True

    iou, inter, union = mask_iou(pred_mask, gt_mask)
    comp = {"iou": iou, "hit@0.5": float(iou >= IOU_THRESHOLD), "inter": inter, "union": union}
    if sam_fallback:
        comp["sam_fallback"] = 1.0
        comp["sam_error"] = sam_error
    return ScoreResult(
        iou,
        True,
        comp,
        {"boxes": box, "n_pos": len(pos), "n_neg": len(neg)},
    )


SCORERS: dict[str, Callable[[str | None, dict[str, Any]], ScoreResult]] = {
    "grounding_bbox": score_grounding,
    "spot_diff": score_spot_diff,
    "segmentation": score_segmentation,
    "counting": score_counting,
}


class RewardMMDeepResearchFn:
    """Task-aware scalar reward for multimodal DeepResearch tasks."""

    def __init__(self, config: RewardConfig):
        self.config = config

    async def async_call(self, input: RewardInput) -> RewardOutput:
        return await asyncio.to_thread(self.__call__, input)

    def __call__(self, input: RewardInput) -> RewardOutput:
        task_info: dict[str, Any] = input.task_info or {}
        action = input.action.action if hasattr(input.action, "action") else input.action
        answer_text = extract_answer(str(action or ""))
        task = task_info.get("task")
        scorer = SCORERS.get(str(task))
        if scorer is None:
            return RewardOutput(
                reward=self.config.unk_error_reward,
                is_correct=False,
                metadata={"error": f"unsupported task: {task}", "task": task, "answer_text": answer_text},
            )

        try:
            result = scorer(answer_text, task_info)
        except Exception as exc:  # noqa: BLE001
            return RewardOutput(
                reward=self.config.unk_error_reward,
                is_correct=False,
                metadata={"error": f"scorer_error: {type(exc).__name__}: {exc}", "task": task, "answer_text": answer_text},
            )

        score = float(max(0.0, min(1.0, result.score)))
        # Kept only for logging/rejection sampling; workflow trains on `reward`.
        if task == "counting":
            is_correct = bool(result.components.get("correct", 0.0) >= 1.0)
        elif "hit@0.5" in result.components:
            # Anchor correctness to the TRUE IoU@0.5 hit, not the (possibly
            # GIoU-shaped) reward, so hit-rate metrics stay comparable across
            # shaping modes. Identical to `score>=IOU_THRESHOLD` when shaping off.
            is_correct = bool(result.components["hit@0.5"] >= 1.0)
        elif "f1" in result.components:
            # spot_diff: anchor to the HARD F1@0.5 component (never the soft_iou
            # reward), so solve_none / is_correct stay comparable to hard-mode runs.
            # Identical to `score>=IOU_THRESHOLD` when soft matching is off.
            is_correct = bool(result.components["f1"] >= IOU_THRESHOLD)
        else:
            is_correct = bool(score >= IOU_THRESHOLD)
        metadata = {
            "task": task,
            "answer_text": answer_text,
            "parse_ok": result.parse_ok,
            "score": score,
            "components": result.components,
            "pred": result.pred,
        }
        return RewardOutput(reward=score, is_correct=is_correct, metadata=metadata)
