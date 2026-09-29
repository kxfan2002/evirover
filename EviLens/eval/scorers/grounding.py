"""Grounding scorer: single-box IoU, with Acc@0.5 aggregated later."""
from .. import parsing, config
from . import register, ScoreResult
from .geometry import iou_2d


@register("grounding")
def score(answer_text, sample, ctx=None) -> ScoreResult:
    coord_order = (ctx or {}).get("coord_order", "xy")
    pred = parsing.parse_bbox(answer_text, coord_order=coord_order)
    gt = sample.gt.get("bbox")
    if pred is None:
        return ScoreResult(score=0.0, parse_ok=False, components={"iou": 0.0}, pred=None)
    iou = iou_2d(pred, gt)
    return ScoreResult(
        score=iou,
        parse_ok=True,
        components={"iou": iou, "hit@0.5": float(iou >= config.IOU_THRESHOLD)},
        pred=pred,
    )
