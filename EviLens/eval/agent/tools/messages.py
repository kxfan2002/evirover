"""Every failure string the model can see, in one place.

Kept byte-identical to the training-side copy of this file: evaluation and
training must show the model the same tool feedback, or the same checkpoint is
not comparable between them. Centralising them also makes the set reviewable —
changing a string changes the model's input distribution.

Three constraints when writing one:
  1. Short. These go into the context window, and failures are a double-digit
     fraction of all tool calls. Aim for one line, ~25 tokens.
  2. Actionable. The model should know what to do next. "unknown image_id:
     'IMG_3'" is not actionable unless the valid ids are listed.
  3. An infrastructure failure must read differently from a genuine empty
     result. If both say "No results.", the model learns that a broken search
     backend means the question has no answer. Group B says "service error /
     rate-limited / network error", group C says "no results for ... Try
     different wording". Downstream code should rely on the error_class field
     rather than on the wording.

The `--- xxx result ---` / `--- end xxx result ---` envelope is kept because the
training data used it.

Groups: A=model_error  B=infra_error  C=empty_result  D=budget  E=agent layer
"""
from __future__ import annotations

from typing import Any, Iterable

# ===========================================================================
# A. model_error -- the model's own mistake. Do not retry; return something it can act on.
# ===========================================================================

# A1 Malformed `boxes`: missing, or not four numbers.
A1_BAD_BOXES = "bad boxes: need 4 numbers [x0,y0,x1,y1] in 0-1000"

# A2 Unknown image_id. Lookup is an exact string match, so IMG_3 vs IMG_03 fails.
#    List every valid id, untruncated: the model cannot guess them.
A2_UNKNOWN_IMAGE_ID = "unknown image_id {got!r}. Available: {available}"

# A3 Box degenerates to zero area after clamping to the image bounds. Say which
#    dimension collapsed, and that clamping caused it.
A3_EMPTY_BOX = "box {box} is empty after clamping ({which} 0). Need x0<x1 and y0<y1."

# A4 Missing required argument.
A4_MISSING_ARG = "[{tool}] {arg} is required."

# A5 Unknown tool name; the message lists the available tools.
A5_UNKNOWN_TOOL = "unknown tool {got!r}."

# A6 Source image file missing or corrupt.
A6_IMAGE_UNREADABLE = "cannot read image {image_id}. Use another image_id."

# A7 compare_lr box is not inside the left panel. Coordinates are normalized over
#    the whole image, so the left panel is x in 0-500; a box crossing x=500 has no
#    counterpart on the right. Make the model fix it rather than silently cropping.
A7_NOT_LEFT_PANEL = (
    "compare_lr needs a box on the LEFT panel, but got x_max={x_max:g}. "
    "Coordinates are normalized 0-1000 over the FULL side-by-side image, so the "
    "left panel is x in 0-500. Give a box with x_max <= 500."
)


# ===========================================================================
# B. infra_error -- infrastructure failure, nothing to do with the model.
#    Distinguished from group C by wording; the hard guarantee is error_class.
# ===========================================================================

# B1 Rate limited (429) and still failing after retries.
B1_RATE_LIMITED = "[{tool}] rate-limited, {tries} retries exhausted."

# B2 5xx/4xx other than 429. The response body is noise; leave it out.
B2_SERVICE_ERROR = "[{tool}] service error HTTP {code}."

# B3 Network failure (connection refused / timeout). Do not inline the traceback.
B3_NETWORK_ERROR = "[{tool}] network error."

# B4 Missing API key. Classed as infra_error so it shows up in the metrics instead
#    of looking like "this question has no answer".
B4_NO_KEY = "[{tool}] unavailable: no API key."

# B5 browse fetch failed. Return the failure directly, without summarizing it.
B5_FETCH_FAILED = "[browse] fetch failed ({reason})."

# B6 image_search upload failed. The call is over; do not attempt the search.
B6_UPLOAD_FAILED = "[image_search] could not upload the image."


# ===========================================================================
# C. empty_result -- a legitimate state of the world. The model should rephrase,
#    not conclude the question is unanswerable.
# ===========================================================================

C1_NO_RESULTS = '[{tool}] no results for "{query}". Try different wording.'

C2_NO_IMAGE_RESULTS = "[{tool}] no visually similar images found."


# ===========================================================================
# D. budget -- a resource constraint, not a failure.
# ===========================================================================

# D1 Serper quota exhausted.
D1_SERPER_BUDGET = "[{tool}] search quota for this task is used up. Answer with what you have."

# D2 One turn left. Without this reminder the loop just ends with no answer, which
#    is the main source of unparseable rollouts. Reserve the last turn for answering.
D2_LAST_TURN = (
    "You have one turn left. Do NOT call any more tools. Based only on what you have "
    "gathered, give your best final answer now, enclosed in <answer></answer> tags."
)

# D3 Length budget exhausted.
D3_LENGTH_BUDGET = (
    "You have reached the response length budget. Do NOT call any more tools. Based only "
    "on the information gathered so far, give your best final answer now, enclosed in "
    "<answer></answer> tags."
)


# ===========================================================================
# E. agent layer -- the tool protocol itself broke.
# ===========================================================================

# E1 <tool_call> did not contain valid JSON.
E1_JSON_PARSE = "[Json Parse Error]: Tool call is not a valid JSON."

# E2 Unexpected exception while running a tool. Keep the exception type for
#    debugging, drop the message body.
E2_TOOL_EXCEPTION = "[{tool}] failed ({exc_type}). Try a different tool or arguments."


# ===========================================================================
# F. SAM3, on the verify_mask side.
# ===========================================================================

# F1 No segment_fn wired in. Degrading silently to a drawn box would let the model
#    believe it is looking at a mask, so say so explicitly.
F1_SAM_DISABLED = "[verify_mask] mask unavailable; showing bbox overlay only, NOT a mask."

# F2 SAM3 call failed. Again, do not let the model think it sees a mask.
F2_SAM_FAILED = "[verify_mask] mask computation failed; showing bbox overlay only, NOT a mask."


# ---------------------------------------------------------------------------
def fmt_available(items: Iterable[Any], limit: int | None = None) -> str:
    """Render available options as a short string. limit=None means no truncation:
    A2 must list every image_id, or the model still cannot pick the right one."""
    items = [str(x) for x in items]
    if not items:
        return "(none)"
    if limit is None or len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", ... (+{len(items) - limit})"
