"""Crop-quality diagnosis for grounding rollouts (eval side).

Answers *why* a grounding rollout missed, so the crop-shaping RL run (v14) can be
judged on its CAUSAL lever (does it convert bad-view misses into hits?) rather than
on process metrics that move for free (crop count, wander).

A grounding MISS (final IoU < IOU_THRESHOLD) is split into:

  no_crop         : ZERO original-frame localization crops -> the agent answered
                    without ever zooming to the target (mode-A direct answer).
  wrong_crop      : the agent DID crop, but no crop was a *good view* of the GT
                    (contained it AND was a genuine zoom) -> it looked in the WRONG
                    place / never framed the target.
  imprecise_final : >=1 crop gave a good view of the GT, yet the final box was still
                    off -> localization-precision ceiling GIVEN a good crop. This is
                    the residual the crop-shaping reward CANNOT fix.

Key numbers (per ckpt):
  bad_view_share  = (no_crop + wrong_crop) / n_miss   -- crop-shaping should DROP this
  imprecise_share = imprecise_final / n_miss          -- the residual precision ceiling

The TREND across checkpoints is the signal, not the absolute level (it depends on the
CONTAIN_THRESH / MAX_CROP_AREA_FRAC thresholds below). Same original-frame /1000-xyxy
containment convention as the training reward (crop_containment_reward.gt_containment),
so the eval diagnosis and the training φ agree on what "contains the GT" means.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

# crop boxes and GT are /1000 xyxy relative to the ORIGINAL image => A_orig is fixed.
A_ORIG = 1000.0 * 1000.0
CONTAIN_THRESH = 0.9          # GT is ~fully inside the crop
MAX_CROP_AREA_FRAC = 0.5      # crop must be <= 50% of the image (>=2x zoom) to count as
                              # a real "view" -- kills the whole-image trivial-containment case.

# Localization crops that actually frame/zoom a region (NOT `verify`, which only overlays a
# box on the full image without zooming). Mirrors the training reward's CROP_TOOL_NAMES intent.
CROP_TOOL_NAMES = {"crop", "verify_part", "crop_and_search", "crop_image"}


def _norm(b: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def gt_containment(crop: Sequence[float], gt: Sequence[float]) -> float:
    """Fraction of the GT box's area inside `crop` (1.0 == GT fully contained)."""
    ga = _area(_norm(gt))
    if ga <= 0.0:
        return 0.0
    return _inter(_norm(crop), _norm(gt)) / ga


def crop_box_from_args(args: object) -> Optional[List[float]]:
    """Original-frame crop box [x1,y1,x2,y2] from a tool-call args dict, else None.

    Nested crops (image_id set to a sub-image) live in a different coordinate frame,
    so their coords aren't comparable to the /1000 ORIGINAL frame -> skip (return None).
    """
    if not isinstance(args, dict):
        return None
    image_id = args.get("image_id")
    if image_id not in (None, "", "ORIGINAL"):
        return None
    bx = args.get("boxes")
    if bx is None:
        bx = args.get("box") or args.get("bbox")
    if not (isinstance(bx, (list, tuple)) and len(bx) == 4):
        return None
    try:
        return _norm([float(v) for v in bx])
    except (TypeError, ValueError):
        return None


def is_good_view(crop: Sequence[float], gt: Sequence[float]) -> bool:
    """True if `crop` both contains the GT and is a genuine zoom (not near-whole-image)."""
    return (gt_containment(crop, gt) >= CONTAIN_THRESH
            and _area(_norm(crop)) <= MAX_CROP_AREA_FRAC * A_ORIG)


def best_containment(gt: Sequence[float], crop_boxes: Sequence[Sequence[float]]) -> float:
    return max((gt_containment(c, gt) for c in crop_boxes if c is not None), default=0.0)


def classify_grounding_miss(
    iou: float,
    gt: Sequence[float],
    crop_boxes: Sequence[Optional[Sequence[float]]],
    iou_thresh: float = 0.5,
) -> str:
    """-> 'hit' | 'no_crop' | 'wrong_crop' | 'imprecise_final' for a grounding rollout."""
    if iou >= iou_thresh:
        return "hit"
    orig = [c for c in crop_boxes if c is not None]
    if not orig:
        return "no_crop"
    if any(is_good_view(c, gt) for c in orig):
        return "imprecise_final"
    return "wrong_crop"
