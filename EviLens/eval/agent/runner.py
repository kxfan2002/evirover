"""The multi-turn agent loop.

Per sample:
  1. system = full-tool prompt; user = SFT-style task turn + ORIGINAL image.
  2. Call the model. Parse its turn:
       - <tool_call> -> execute the tool, append <tool_response> (+ any images) as
         a user turn, loop.
       - <answer>    -> stop; this is the final answer.
  3. Cap total tool calls and turns; on cap/parse-failure, take the last <answer>
     seen if any, else empty.
  4. Score the final <answer> with the SAME scorer the QA path uses, and emit a
     row in the identical schema report.py consumes (plus a `trajectory` field).

Image interleaving: messages carry text with <image> placeholders and a parallel
list of image paths; build_openai_messages() converts each turn into OpenAI
content parts (text split around <image>, images as base64 data URLs) so the
model sees images exactly where the text references them — matching SFT.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import config, parsing
from ..scorers import get_scorer
from . import crop_diag
from . import prompts
from .client import AgentChatClient
from .image_store import ImageStore, image_to_data_url
from .tools import messages as MSG
from .tools.base import ToolContext, format_tool_response, SerperBudget

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


# -- message construction ---------------------------------------------------

class Turn:
    """One conversation turn: role + text (with <image> markers) + image paths."""

    def __init__(self, role: str, text: str, image_paths: Optional[List[str]] = None):
        self.role = role
        self.text = text
        self.image_paths = image_paths or []


@lru_cache(maxsize=4096)
def _encode_or_note(path: str) -> Optional[str]:
    """Data URL for one image, or None if it cannot be decoded.

    A search tool can save bytes that are not a decodable image (an HTML error
    page, a truncated download). Decoding happens here, at payload-build time,
    which is *outside* the per-tool try/except in the rollout loop — so a single
    bad download used to raise all the way out and kill the whole episode with
    finish_reason=api_error, scoring the sample 0. Failures are returned instead
    of raised so the caller can degrade to a text note; the cache keeps a known
    bad file from being re-decoded on every subsequent turn.
    """
    try:
        return image_to_data_url(path)
    except Exception as e:  # noqa: BLE001 — one unreadable file must not end the rollout
        print(f"  [image] undecodable, skipped: {path} ({type(e).__name__}: {e})")
        return None


def _image_part(path: str) -> Dict[str, Any]:
    url = _encode_or_note(path)
    if url is None:
        return {"type": "text", "text": "[image unavailable: file could not be decoded]"}
    return {"type": "image_url", "image_url": {"url": url}}


def build_openai_messages(turns: List[Turn]) -> List[Dict[str, Any]]:
    """Turn the conversation into OpenAI chat messages with interleaved images."""
    messages: List[Dict[str, Any]] = []
    for t in turns:
        if not t.image_paths:
            messages.append({"role": t.role, "content": t.text})
            continue
        # Split text on <image> markers; splice image parts in order.
        segments = t.text.split("<image>")
        parts: List[Dict[str, Any]] = []
        for i, seg in enumerate(segments):
            if seg:
                parts.append({"type": "text", "text": seg})
            if i < len(segments) - 1 and i < len(t.image_paths):
                parts.append(_image_part(t.image_paths[i]))
        # Any extra images (marker/count mismatch) get appended at the end.
        for j in range(len(segments) - 1, len(t.image_paths)):
            parts.append(_image_part(t.image_paths[j]))
        messages.append({"role": t.role, "content": parts})
    return messages


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first well-formed {name, arguments} tool call, or None."""
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    blob = m.group(1).strip()
    try:
        obj = json.loads(blob)
    except Exception:
        # Tolerate trailing commentary after the JSON object.
        try:
            depth, end = 0, None
            for i, ch in enumerate(blob):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            obj = json.loads(blob[:end]) if end else None
        except Exception:
            obj = None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments")
    if not isinstance(args, dict):
        args = obj.get("parameters") if isinstance(obj.get("parameters"), dict) else {}
    return {"name": str(obj["name"]), "arguments": args}


# Tools whose ORIGINAL-frame box counts as a candidate answer region.
_CANDIDATE_TOOL_NAMES = {"crop", "verify", "verify_part", "crop_image"}
_MAX_RECAP_CANDIDATES = 8


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _dedupe_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse near-identical regions (the loops repeat the same box many times),
    keeping the LAST occurrence of each distinct region and preserving order."""
    out: List[Dict[str, Any]] = []
    for c in reversed(cands):
        if not any(_iou(c["box"], o["box"]) > 0.7 for o in out):
            out.append(c)
        if len(out) >= _MAX_RECAP_CANDIDATES:
            break
    return list(reversed(out))


def _commit_phase(turns: List[Turn], client: AgentChatClient,
                  candidates: List[Dict[str, Any]],
                  answered: bool) -> tuple:
    """Re-present the model's own candidate regions and ask it to commit.

    No ground truth is consulted — the candidate list comes purely from the boxes the
    model aimed its own tools at. The crops themselves are already in context; this
    turn just re-surfaces the coordinates so the final answer can reference the BEST
    region rather than whichever one happened to be last.
    Returns (revised|None, recap_note_for_trajectory).
    """
    cands = _dedupe_candidates(candidates)
    if not cands:
        return None, "[commit phase: no candidates]"
    lines = [f"  {i+1}. {c['tool']} -> [{c['box'][0]}, {c['box'][1]}, {c['box'][2]}, {c['box'][3]}]"
             for i, c in enumerate(cands)]
    lead = ("You have already given an answer. Before it is accepted, review the regions "
            "you examined during this episode:" if answered else
            "You are out of budget and must answer now. These are the regions you examined "
            "during this episode:")
    body = (f"{lead}\n" + "\n".join(lines) + "\n\n"
            "Identify which of these regions actually contains the target, then output the "
            "final tight bounding box for the target itself (it may be a sub-region of the "
            "region you pick — do not just copy a region's coordinates unless it is already "
            "tight). Do NOT call any more tools. Reply with only your final answer in the "
            "required <answer>...</answer> format.")
    turns.append(Turn("user", f"<tool_response>\n{body}\n</tool_response>"))
    try:
        resp = client.chat(build_openai_messages(turns))
    except Exception as e:  # noqa: BLE001 — a failed commit phase must not break the rollout
        return None, f"[commit phase: request failed: {type(e).__name__}]"
    raw = resp.assistant_text()
    turns.append(Turn("assistant", raw))
    m = _ANSWER_RE.search(raw)
    return ({"raw": raw, "answer": m.group(1).strip() if m else None},
            f"[commit phase: {len(cands)} candidates re-presented]")


# -- the loop ---------------------------------------------------------------

def run_agent_sample(
    sample,
    client: AgentChatClient,
    tools: Dict[str, Any],
    output_dir: str,
    max_turns: int = 34,
    max_tool_calls: int = 30,
    sam_ctx: Optional[Dict[str, Any]] = None,
    serper_per_sample: int = 0,
    final_recap: str = "off",
) -> Dict[str, Any]:
    """Run one sample through the agent loop and return a scored row."""
    # A rollout that spends every turn on a tool needs one turn to receive the
    # "budget exhausted" nudge and at least one more to act on it. Equal budgets
    # leave none, so those rollouts end at max_turns with no answer and score 0 —
    # a silent ~10-13% loss that looks like a capability drop, not a config bug.
    if max_turns < max_tool_calls + 2:
        print(f"  [agent] WARNING: max_turns={max_turns} <= max_tool_calls+1="
              f"{max_tool_calls + 1}: tool-exhausted rollouts get no turn to answer.")
    store = ImageStore(output_dir, sample_run_id=_safe_id(sample.id))
    store.add_original(sample.image_path)
    # Fresh per-sample Serper budget (>0 enables per-sample capping; each sample
    # gets its own allotment so image search is fair and deterministic instead of
    # a run-wide first-come race). 0 => leave the tools' shared/run-wide budget.
    per_sample_budget = SerperBudget(serper_per_sample) if serper_per_sample and serper_per_sample > 0 else None
    ctx = ToolContext(sample=sample, image_store=store, output_dir=output_dir,
                      serper_budget=per_sample_budget)

    user_text, template_key = prompts.build_user_text(sample)
    turns: List[Turn] = [
        Turn("system", prompts.SYSTEM_PROMPT),
        Turn("user", f"<image>\n{user_text}", [sample.image_path]),
    ]

    trajectory: List[Dict[str, Any]] = []
    crop_boxes: List[Optional[List[float]]] = []   # original-frame crop boxes, for miss diagnosis
    candidates: List[Dict[str, Any]] = []          # original-frame regions proposed, for the commit phase
    final_answer: Optional[str] = None
    finish_reason = "answered"
    n_tool_calls = 0
    no_action_streak = 0        # consecutive turns with neither tool_call nor answer

    for _turn in range(max_turns):
        resp = client.chat(build_openai_messages(turns))
        assistant_text = resp.assistant_text()
        turns.append(Turn("assistant", assistant_text))
        trajectory.append({"role": "assistant", "content": assistant_text,
                           "finish_reason": resp.finish_reason})

        answer = _ANSWER_RE.search(assistant_text)
        tool_call = _parse_tool_call(assistant_text)

        # An <answer> ends the episode (even if a tool call also appears, the
        # answer takes precedence — the model is done).
        if answer:
            final_answer = answer.group(1).strip()
            finish_reason = "answered"
            break

        if tool_call is None:
            # Neither a tool_call nor an answer: nudge once, then give up. Without
            # a cap, a model stuck emitting unclosed <think> burns the full token
            # budget every turn until max_turns. The streak is consecutive, so an
            # idle turn after a successful tool call starts over.
            no_action_streak += 1
            if no_action_streak > 1:
                finish_reason = "empty_response" if not assistant_text.strip() else "no_action"
                trajectory.append({"role": "user", "content": f"[stop: {finish_reason}]"})
                break
            turns.append(Turn("user",
                "<tool_response>\nNo <tool_call> or <answer> found. Emit exactly one "
                "tool call, or your final <answer>.\n</tool_response>"))
            trajectory.append({"role": "user", "content": "[nudge: no action]"})
            continue
        no_action_streak = 0

        if n_tool_calls >= max_tool_calls:
            # Must stay word-for-word identical to the training-side signal
            # (MSG.D2_LAST_TURN): a model trained on one phrasing and evaluated on
            # another is being tested out of distribution.
            finish_reason = "tool_budget_exhausted"
            turns.append(Turn("user", f"<tool_response>\n{MSG.D2_LAST_TURN}\n</tool_response>"))
            trajectory.append({"role": "user", "content": "[tool budget exhausted]"})
            continue

        # Execute the tool.
        name = tool_call["name"]
        args = tool_call["arguments"]
        n_tool_calls += 1
        # Record the model's DECLARED localization crop (original-frame only) BEFORE
        # execution — a box aimed at the wrong place is a wrong-place signal even if the
        # crop itself errors. Used by crop_diag to split misses (wrong_crop vs imprecise).
        if name in crop_diag.CROP_TOOL_NAMES:
            crop_boxes.append(crop_diag.crop_box_from_args(args))
        # Candidate set for the commit phase: every ORIGINAL-frame region the model
        # aimed a look/verify tool at. These are its own proposals, so no GT is used.
        if name in _CANDIDATE_TOOL_NAMES:
            _cb = crop_diag.crop_box_from_args(args)
            if _cb is not None:
                candidates.append({"tool": name, "box": [round(float(v)) for v in _cb]})
        tool = tools.get(name)
        if tool is None:
            body = (f"--- {name} result ---\nUnknown tool: {name!r}. Available: "
                    f"{', '.join(sorted(tools))}.\n--- end {name} result ---")
            turns.append(Turn("user", f"<tool_response>\n{body}\n</tool_response>"))
            trajectory.append({"role": "tool", "name": name, "error": "unknown_tool"})
            continue
        try:
            result = tool.call(args, ctx)
        except Exception as e:  # noqa: BLE001
            body = f"--- {name} result ---\n[Error] {type(e).__name__}: {e}\n--- end {name} result ---"
            turns.append(Turn("user", f"<tool_response>\n{body}\n</tool_response>"))
            trajectory.append({"role": "tool", "name": name, "args": _short(args), "error": str(e)[:200]})
            continue

        resp_text = format_tool_response(result)
        img_paths = [im.path for im in result.images]
        turns.append(Turn("user", resp_text, img_paths))
        trajectory.append({
            "role": "tool", "name": name, "args": _short(args),
            "n_images": len(img_paths),
            "text": result.text[:1000],
        })
    else:
        finish_reason = "max_turns"

    # -- commit phase -------------------------------------------------------
    # Models tend to answer with the last region they examined rather than the best
    # one, and a rollout that hits max_turns emits no answer at all. Re-present the
    # candidate regions the model itself proposed and let it choose.
    #   "rescue" : only when the loop produced no answer
    #   "always" : also let an answered rollout revise
    if final_recap != "off" and candidates:
        need_rescue = final_answer is None
        if need_rescue or final_recap == "always":
            revised, recap_note = _commit_phase(
                turns, client, candidates, answered=(final_answer is not None))
            trajectory.append({"role": "user", "content": recap_note})
            if revised is not None:
                trajectory.append({"role": "assistant", "content": revised["raw"],
                                   "finish_reason": "commit_phase"})
                if revised["answer"] is not None:
                    final_answer = revised["answer"]
                    finish_reason = "rescued" if need_rescue else "revised"

    # Score the final answer with the same scorer the QA path uses.
    row = _score_row(sample, final_answer, finish_reason, sam_ctx)
    row["template_key"] = template_key
    row["n_tool_calls"] = n_tool_calls
    row["trajectory"] = trajectory
    row["images"] = [im.rel_path for im in store.images]

    # Crop-quality miss diagnosis (grounding only): WHY did this rollout miss?
    # Splits misses into no_crop / wrong_crop / imprecise_final so the crop-shaping RL
    # run can be judged on its causal lever. Never let a diag error break a rollout.
    if sample.task == "grounding_bbox":
        try:
            gt_bbox = sample.gt.get("bbox")
            comp = row.setdefault("components", {})
            iou = float(comp.get("iou", 0.0))
            if crop_diag.crop_box_from_args({"boxes": gt_bbox}) is not None:
                miss_type = crop_diag.classify_grounding_miss(iou, gt_bbox, crop_boxes)
                comp["miss_type"] = miss_type
                comp["crop_best_containment"] = crop_diag.best_containment(gt_bbox, crop_boxes)
                comp["n_orig_crops"] = sum(1 for c in crop_boxes if c is not None)
                row["miss_type"] = miss_type
        except Exception:  # noqa: BLE001 — diagnosis must never break scoring
            pass
    return row


def _score_row(sample, answer: Optional[str], finish_reason: str,
               sam_ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = {
        "id": sample.id, "file": sample.file, "family": sample.family,
        "subcategory": sample.subcategory, "task": sample.task,
    }
    scorer = get_scorer(config.TASK_SCORER[sample.task])
    try:
        res = scorer(answer, sample, ctx=sam_ctx)
        score, parse_ok, components, pred = res.score, res.parse_ok, res.components, res.pred
    except Exception as e:  # noqa: BLE001
        score, parse_ok, components, pred = 0.0, False, {"scorer_error": str(e)[:200]}, None
    return {
        **base,
        "score": float(score), "parse_ok": bool(parse_ok),
        "components": components, "pred": pred,
        "answer": answer, "content": answer,
        "reasoning_content": None,
        "finish_reason": finish_reason,
        "gt": sample.gt,
    }


def _safe_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id))[:120]


def _short(args: Any) -> Any:
    """Trim large arg values (e.g. python code) for the trajectory log."""
    if isinstance(args, dict):
        out = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 400:
                out[k] = v[:400] + f"...(+{len(v) - 400} chars)"
            else:
                out[k] = v
        return out
    return args
