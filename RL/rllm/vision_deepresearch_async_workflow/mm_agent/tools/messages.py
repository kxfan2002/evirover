from __future__ import annotations

from typing import Any, Iterable

# ===========================================================================
# ===========================================================================

A1_BAD_BOXES = "bad boxes: need 4 numbers [x0,y0,x1,y1] in 0-1000"

A2_UNKNOWN_IMAGE_ID = "unknown image_id {got!r}. Available: {available}"

A3_EMPTY_BOX = "box {box} is empty after clamping ({which} 0). Need x0<x1 and y0<y1."

A4_MISSING_ARG = "[{tool}] {arg} is required."

A5_UNKNOWN_TOOL = "unknown tool {got!r}."

A6_IMAGE_UNREADABLE = "cannot read image {image_id}. Use another image_id."

A7_NOT_LEFT_PANEL = (
    "compare_lr needs a box on the LEFT panel, but got x_max={x_max:g}. "
    "Coordinates are normalized 0-1000 over the FULL side-by-side image, so the "
    "left panel is x in 0-500. Give a box with x_max <= 500."
)


# ===========================================================================
# ===========================================================================

B1_RATE_LIMITED = "[{tool}] rate-limited, {tries} retries exhausted."

B2_SERVICE_ERROR = "[{tool}] service error HTTP {code}."

B3_NETWORK_ERROR = "[{tool}] network error."

B4_NO_KEY = "[{tool}] unavailable: no API key."

B5_FETCH_FAILED = "[browse] fetch failed ({reason})."

B6_UPLOAD_FAILED = "[image_search] could not upload the image."

B7_IMAGE_DOWNLOAD_FAILED = "[{tool}] found images but all {n} download(s) failed (network error)."


# ===========================================================================
# ===========================================================================

C1_NO_RESULTS_QUOTED = (
    "[{tool}] 0 results. Every quoted phrase must appear verbatim on one page. "
    "Retry with fewer quotes."
)

C1_NO_RESULTS_PLAIN = "[{tool}] 0 results. Try broader or more common terms."

C1_NO_RESULTS = C1_NO_RESULTS_PLAIN

C2_NO_IMAGE_RESULTS = "[{tool}] no visually similar images found."

C2_NO_IMAGE_RESULTS_QUOTED = (
    "[{tool}] no images found. Every quoted phrase must appear verbatim. "
    "Retry with fewer quotes."
)


# ===========================================================================
# ===========================================================================

D1_SERPER_BUDGET = "[{tool}] search quota for this task is used up. Answer with what you have."

D2_LAST_TURN = (
    "You have one turn left. Do NOT call any more tools. Based only on what you have "
    "gathered, give your best final answer now, enclosed in <answer></answer> tags."
)

D4_LEFT_PANEL = (
    "Your answer contains boxes on the RIGHT panel (x > 500). Only boxes on the LEFT "
    "panel are accepted: every box must satisfy 0 <= x <= 500. If you located a "
    "difference on the right panel, subtract 500 from both x coordinates to map it "
    "back to the left panel. Re-output your complete final answer now, with ALL boxes "
    "on the left panel, enclosed in <answer></answer> tags. This is your only chance."
)

D3_LENGTH_BUDGET = (
    "You have reached the response length budget. Do NOT call any more tools. Based only "
    "on the information gathered so far, give your best final answer now, enclosed in "
    "<answer></answer> tags."
)


# ===========================================================================
# ===========================================================================

E1_JSON_PARSE = "[Json Parse Error]: Tool call is not a valid JSON."

E2_TOOL_EXCEPTION = "[{tool}] failed ({exc_type}). Try a different tool or arguments."


# ===========================================================================
# ===========================================================================

F1_SAM_DISABLED = "[verify_mask] mask unavailable; showing bbox overlay only, NOT a mask."

F2_SAM_FAILED = "[verify_mask] mask computation failed; showing bbox overlay only, NOT a mask."


# ---------------------------------------------------------------------------
def fmt_available(items: Iterable[Any], limit: int | None = None) -> str:
    items = [str(x) for x in items]
    if not items:
        return "(none)"
    if limit is None or len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", ... (+{len(items) - limit})"
