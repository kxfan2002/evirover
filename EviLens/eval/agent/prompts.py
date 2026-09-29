"""Agent-path prompts: full-tool system prompt + per-task user turns.

The agent path must match the distribution the checkpoint was SFT'd on, so:
  - System prompt = unified_system_prompt.txt verbatim (the full <tools> version),
    NOT the tool-stripped QA prompt in eval/prompts.py.
  - User turn = the exact per-task/source template the SFT data used, filled with
    the record's description (and image size / diff count where the template needs
    it), so wording, framing, and answer-format instructions are identical.

The mapping is keyed on (task, _source_file). Unknown sources fall back to a
sensible per-task default; the chosen template key is returned for the trajectory
log so the mapping stays auditable.
"""
from __future__ import annotations

import os
from typing import Tuple

from PIL import Image

# Path to the exact SFT system prompt (full tool version), at the repo root.
_SYS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "unified_system_prompt.txt",
)


def _load_system_prompt() -> str:
    path = os.path.abspath(_SYS_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"agent system prompt not found: {path}\n"
            "unified_system_prompt.txt must sit at the repo root — the agent path "
            "requires the verbatim full-tool prompt, and substituting the "
            "tool-stripped QA prompt would silently change the protocol.")
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


SYSTEM_PROMPT = _load_system_prompt()


# -- user-turn templates (verbatim heads/tails from the SFT data) -----------

# -- union answer-format tails (SHARED verbatim with SFT data and eval/prompts.py) --
# These strings must stay byte-identical across the SFT user turns, eval/prompts.py
# FAMILY_ANSWER_HINT, and the templates below, so train/eval wording is aligned.
_HINT_GROUNDING = (
    "Answer with a single bounding box for the target, normalized to 0-1000, "
    "in the format:\n<answer>[x_min, y_min, x_max, y_max]</answer>"
)
_HINT_SEGMENTATION = (
    "Answer with a segmentation prompt as a JSON object containing one bounding box, "
    "3 positive points (clearly INSIDE the target), and 3 negative points "
    "(clearly OUTSIDE the target), all normalized to 0-1000, in the format:\n"
    '<answer>{{"boxes":[x_min,y_min,x_max,y_max],"positive_points":[[x,y],[x,y],[x,y]],'
    '"negative_points":[[x,y],[x,y],[x,y]]}}</answer>'
)
_HINT_COUNTING = (
    "Answer with a single integer in the format:\n<answer>N</answer>"
)
_HINT_SPOT_DIFF = (
    "Answer with a list of bounding boxes on the LEFT panel only "
    "(x coordinates in 0-500), normalized to 0-1000, with one bounding box per "
    "difference (each box tightly around a single difference) and exactly {n} boxes, "
    "in the format:\n<answer>[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]</answer>"
)

_GROUNDING_PERSON = (
    "Find the bounding box for the person described below.\n\n"
    "Description:\n{desc}\n\n" + _HINT_GROUNDING
)
_GROUNDING_CHARACTER = (
    "Find the bounding box for the anime/game/illustrated character described below.\n\n"
    "Description:\n{desc}\n\n" + _HINT_GROUNDING
)
_GROUNDING_TARGET = (
    "Find the bounding box for the described target in the image.\n\n"
    "Description:\n{desc}\n\n" + _HINT_GROUNDING
)
_SEG_PERSON = (
    "Prepare segmentation prompts for the person described below.\n\n"
    "Original image size: width={w}px, height={h}px.\n\n"
    "Description:\n{desc}\n\n" + _HINT_SEGMENTATION
)
_SEG_CHARACTER = (
    "Prepare segmentation prompts for the anime/game/illustrated character described below.\n\n"
    "Original image size: width={w}px, height={h}px.\n\n"
    "Description:\n{desc}\n\n" + _HINT_SEGMENTATION
)
_SPOT_DIFF = (
    "This is a find-the-difference image. There are exactly **{n}** differences on "
    "the **left** panel. Compare the left and right halves to find all of them, then "
    "provide bounding boxes for each difference on the left.\n\n"
    "Hint: find {n} differences\n\n" + _HINT_SPOT_DIFF
)

# _source_file -> whether the subject is an anime/game/illustrated character.
_CHARACTER_SOURCES = {"anime_grounding", "anime_seg"}


def _image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w, h)


def build_user_text(sample) -> Tuple[str, str]:
    """Return (user_text, template_key) for a Sample, matching SFT wording."""
    raw = sample.raw or {}
    # Data-selection path: use the RL-training parquet's `question` VERBATIM so the
    # rollout prompt is byte-identical to what training feeds (avoids template
    # mis-selection across sources). Set by pool_dataset.load_pool_samples().
    if raw.get("_verbatim_question") and raw.get("question"):
        return raw["question"], "qa_verbatim"
    task = sample.task
    src = (sample.raw or {}).get("_source_file", "")
    desc = (sample.description or "").strip()

    if task == "counting":
        # Counting: the raw question plus the shared integer answer-format hint.
        return f"{desc}\n\n{_HINT_COUNTING}", "counting_raw"

    if task == "spot_diff":
        gt = sample.gt.get("bboxes") or []
        n = len(gt) if gt else 0
        # Fall back to any count embedded in the description if GT is unavailable.
        return _SPOT_DIFF.format(n=n if n else 1), f"spot_diff_n{n}"

    if task == "grounding_bbox":
        if src in _CHARACTER_SOURCES:
            return _GROUNDING_CHARACTER.format(desc=desc), "grounding_character"
        if sample.subcategory == "localization":
            return _GROUNDING_TARGET.format(desc=desc), "grounding_target"
        # recognition of real people (real_grounding, nano_grounding)
        return _GROUNDING_PERSON.format(desc=desc), "grounding_person"

    if task == "segmentation":
        w = (sample.raw or {}).get("image_width") or 0
        h = (sample.raw or {}).get("image_height") or 0
        if not (w and h):
            w, h = _image_size(sample.image_path)
        if src in _CHARACTER_SOURCES:
            return _SEG_CHARACTER.format(w=w, h=h, desc=desc), "seg_character"
        return _SEG_PERSON.format(w=w, h=h, desc=desc), "seg_person"

    # Unknown task: send the raw description.
    return desc, "raw"
