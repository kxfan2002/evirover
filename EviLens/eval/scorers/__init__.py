"""Scorer registry. Each scorer maps a parsed model answer + GT to a ScoreResult."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class ScoreResult:
    score: float                                  # primary per-sample score in [0,1]
    parse_ok: bool                                # did the answer parse into the expected shape
    components: Dict[str, Any] = field(default_factory=dict)  # extra metrics for reporting
    pred: Any = None                              # normalized prediction (for logging)


# family -> callable(answer_text: str, sample, ctx) -> ScoreResult
SCORER_REGISTRY: Dict[str, Callable] = {}


def register(family: str):
    def deco(fn):
        SCORER_REGISTRY[family] = fn
        return fn
    return deco


def get_scorer(family: str) -> Callable:
    if family not in SCORER_REGISTRY:
        raise ValueError(f"no scorer registered for family: {family}")
    return SCORER_REGISTRY[family]


# Import submodules to trigger registration.
from . import grounding, counting, spot_diff, segmentation  # noqa: E402,F401
