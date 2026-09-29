"""Shared geometry helpers (bbox IoU), matching the training reward convention."""
from typing import List, Sequence


def is_number_list(x, n=None) -> bool:
    if not isinstance(x, (list, tuple)):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        for v in x:
            float(v)
        return True
    except Exception:
        return False


def iou_2d(box1: Sequence[float], box2: Sequence[float]) -> float:
    """IoU of two [x1,y1,x2,y2] boxes. Returns 0 on malformed input."""
    if not is_number_list(box1, 4) or not is_number_list(box2, 4):
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
