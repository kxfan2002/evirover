"""Counting scorer: exact integer match (0/1). MAE recorded as a side metric."""
from .. import parsing
from . import register, ScoreResult


@register("counting")
def score(answer_text, sample, ctx=None) -> ScoreResult:
    pred = parsing.parse_int(answer_text)
    gt = sample.gt.get("answer")
    if pred is None or gt is None:
        return ScoreResult(score=0.0, parse_ok=False, components={}, pred=pred)
    correct = float(int(pred) == int(gt))
    return ScoreResult(
        score=correct,
        parse_ok=True,
        components={"correct": correct, "abs_err": abs(int(pred) - int(gt))},
        pred=pred,
    )
