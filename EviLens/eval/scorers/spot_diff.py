"""Spot-the-difference scorer: greedy IoU>=0.5 matching -> precision/recall/F1.

All predicted boxes must lie on the LEFT panel (x in [0,500]); boxes that spill
past x=500 are treated as invalid and dropped before matching.
"""
from .. import parsing, config
from . import register, ScoreResult
from .geometry import iou_2d


def _valid_left_panel(box) -> bool:
    # x_min and x_max must both be within the left half (0-500).
    return box[0] <= 500 and box[2] <= 500


@register("spot_diff")
def score(answer_text, sample, ctx=None) -> ScoreResult:
    coord_order = (ctx or {}).get("coord_order", "xy")
    preds = parsing.parse_bbox_list(answer_text, coord_order=coord_order)
    gts = sample.gt.get("bboxes") or []
    n_gt = len(gts)
    if preds is None:
        return ScoreResult(
            score=0.0, parse_ok=False,
            components={"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_gt": n_gt, "n_pred": 0},
            pred=None,
        )
    valid = [p for p in preds if _valid_left_panel(p)]
    n_pred = len(valid)

    # Greedy matching by descending IoU; each GT and pred used at most once.
    pairs = []
    for pi, p in enumerate(valid):
        for gi, g in enumerate(gts):
            iou = iou_2d(p, g)
            if iou >= config.IOU_THRESHOLD:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    tp = 0
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ScoreResult(
        score=f1,
        parse_ok=True,
        components={
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "n_gt": n_gt, "n_pred": n_pred,
        },
        pred=valid,
    )
