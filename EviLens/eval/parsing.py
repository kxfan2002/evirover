"""Tolerant extraction of the model's <answer> and per-family parsing."""
import ast
import json
import re
from typing import List, Optional

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: str) -> Optional[str]:
    """Return the content of the last <answer>...</answer>, or None."""
    if not isinstance(text, str):
        return None
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _swap_box(box):
    """[y1,x1,y2,x2] -> [x1,y1,x2,y2]."""
    return [box[1], box[0], box[3], box[2]]


def _swap_point(pt):
    """[y,x] -> [x,y]."""
    return [pt[1], pt[0]]


def _loads(s: str):
    """Parse JSON, falling back to python-literal eval (handles single quotes)."""
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def _all_numbers(x, n=None) -> bool:
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


def parse_bbox(answer: str, coord_order: str = "xy") -> Optional[List[float]]:
    """Parse a single [x1,y1,x2,y2] bbox from the answer text.

    coord_order="yx" means the model emits [y1,x1,y2,x2]; we swap it to canonical
    xyxy before returning so all downstream scoring stays in one convention.
    """
    if answer is None:
        return None
    obj = _loads(answer)
    box = None
    if _all_numbers(obj, 4):
        box = [float(v) for v in obj]
    else:
        # Fallback: grab the first 4 numbers in the string.
        nums = _FLOAT_RE.findall(answer)
        if len(nums) >= 4:
            box = [float(v) for v in nums[:4]]
    if box is None:
        return None
    return _swap_box(box) if coord_order == "yx" else box


def parse_int(answer: str) -> Optional[int]:
    """Parse an integer count from the answer text."""
    if answer is None:
        return None
    obj = _loads(answer)
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        return int(round(obj))
    m = _INT_RE.search(answer)
    if m:
        try:
            return int(m.group())
        except Exception:
            return None
    return None


def parse_seg(answer: str, coord_order: str = "xy"):
    """Parse {"boxes":[..4..], "positive_points":[[x,y]*3], "negative_points":[[x,y]*3]}.

    Returns (box, pos_points, neg_points) or None if the required box is missing.
    Points are best-effort (empty lists allowed). coord_order="yx" means the model
    emits box [y1,x1,y2,x2] and points [y,x]; both are swapped to xyxy / [x,y].
    """
    if answer is None:
        return None
    yx = coord_order == "yx"
    obj = _loads(answer)
    if isinstance(obj, dict):
        box = obj.get("boxes")
        if _all_numbers(box, 4):
            box = [float(v) for v in box]

            def _pts(key):
                pts = obj.get(key) or []
                out = []
                if isinstance(pts, (list, tuple)):
                    for p in pts:
                        if _all_numbers(p, 2):
                            out.append([float(p[0]), float(p[1])])
                return out

            pos, neg = _pts("positive_points"), _pts("negative_points")
            if yx:
                box = _swap_box(box)
                pos = [_swap_point(p) for p in pos]
                neg = [_swap_point(p) for p in neg]
            return box, pos, neg

    # Fallback: strict JSON failed (e.g. flattened point lists). The box drives SAM,
    # so recover it via regex; points are best-effort (dropped if unrecoverable).
    return _parse_seg_lenient(answer, coord_order)


def _parse_seg_lenient(answer: str, coord_order: str = "xy"):
    """Recover box (required) and best-effort points from malformed seg JSON."""
    box_m = re.search(r'"?boxes"?\s*:\s*\[([^\]]*)\]', answer)
    if not box_m:
        return None
    box_nums = _FLOAT_RE.findall(box_m.group(1))
    if len(box_nums) < 4:
        return None
    box = [float(v) for v in box_nums[:4]]

    def _pts(key):
        # Grab the substring after the key up to the next key or end, then pair numbers.
        m = re.search(r'"?' + key + r'"?\s*:\s*(.*?)(?="?(?:positive_points|negative_points|boxes)"?\s*:|\}|$)',
                      answer, re.DOTALL)
        if not m:
            return []
        nums = [float(v) for v in _FLOAT_RE.findall(m.group(1))]
        return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]

    pos, neg = _pts("positive_points"), _pts("negative_points")
    if coord_order == "yx":
        box = _swap_box(box)
        pos = [_swap_point(p) for p in pos]
        neg = [_swap_point(p) for p in neg]
    return box, pos, neg


def parse_bbox_list(answer: str, coord_order: str = "xy") -> Optional[List[List[float]]]:
    """Parse a list of [x1,y1,x2,y2] boxes (spot-the-difference).

    coord_order="yx" swaps each box [y1,x1,y2,x2] -> xyxy before returning.
    """
    if answer is None:
        return None
    obj = _loads(answer)
    boxes = []
    if isinstance(obj, (list, tuple)):
        # Case A: list of 4-number boxes.
        for b in obj:
            if _all_numbers(b, 4):
                boxes.append([float(v) for v in b])
        # Case B: a single flat 4-number box.
        if not boxes and _all_numbers(obj, 4):
            boxes = [[float(v) for v in obj]]
    if not boxes:
        return None
    return [_swap_box(b) for b in boxes] if coord_order == "yx" else boxes
