from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image
import re
from collections import Counter
import json5

# rLLM imports
from rllm.engine.rollout import RolloutEngine
from rllm.engine.agent_workflow_engine import _resolve_task_key
from vision_deepresearch_async_workflow.mm_agent.tools import messages as MSG

# Constants from original DeepResearch
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = int(os.environ.get("VDR_MAX_LLM_CALL_PER_RUN", "50"))


def _is_spot_diff(task) -> bool:
    if not isinstance(task, dict):
        return False
    return "spot_diff" in (str(task.get("task") or "") + "|" +
                           str(task.get("family") or "") + "|" +
                           str(task.get("source_file") or "")).lower()


def _has_right_panel_box(answer_text: str) -> bool:
    try:
        import json as _json
        import re as _re
        m = _re.search(r"\[.*\]", str(answer_text or ""), _re.S)
        if not m:
            return False
        boxes = _json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return False
    if not (isinstance(boxes, list) and boxes):
        return False
    if not isinstance(boxes[0], (list, tuple)):
        boxes = [boxes]
    for b in boxes:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                if float(b[0]) > 500 or float(b[2]) > 500:
                    return True
            except Exception:  # noqa: BLE001
                continue
    return False


def _load_max_calls_by_task() -> dict:
    raw = os.environ.get("VDR_MAX_LLM_CALL_BY_TASK", "")
    if not raw.strip():
        return {}
    try:
        return {str(k): int(v) for k, v in json.loads(raw).items()}
    except Exception as exc:  # noqa: BLE001
        print(f"[deepresearch_agent] failed to parse VDR_MAX_LLM_CALL_BY_TASK, ignoring: {exc}")
        return {}


MAX_LLM_CALL_BY_TASK = _load_max_calls_by_task()


def _load_empty_search_gate() -> int:
    raw = os.environ.get("VDR_EMPTY_SEARCH_GATE_N", "0")
    try:
        return max(0, int(raw))
    except Exception:  # noqa: BLE001
        print(f"[deepresearch_agent] failed to parse VDR_EMPTY_SEARCH_GATE_N ({raw!r}), treating it as off")
        return 0


_EMPTY_SEARCH_GATE_N = _load_empty_search_gate()
MAX_PROMPT_LENGTH_PER_RUN = 64000
MAX_RESPONSE_LENGTH_PER_RUN = 4096
# Plan B: tokens reserved below the training truncation window so the forced
# final-answer turn (plus one-turn image-lag) still fits inside the window.
# Env-overridable for tuning without code edits.
ANSWER_RESERVE_TOKENS = int(os.environ.get("VDR_ANSWER_RESERVE_TOKENS", "3500"))

# --- Dynamic per-turn generation budget (kills the cumulative_mm truncation) ---
# Every completion's max_tokens is clamped to min(MAX_RESPONSE_LENGTH_PER_RUN,
# remaining-window), so NO single turn (natural OR forced-answer) can push the
# image-expanded cumulative response past data.max_response_length -> the trainer
# never has to drop trailing turns (which was silently deleting <answer>).
#   remaining = window - resp_used(engine) - lagged_reserve - FRAMING_MARGIN_TOKENS
# resp_used is one TOOL TURN behind (the just-appended turn only lands in the NEXT
# engine prompt_length), so lagged_reserve = self._pending_lag = the EXACT token
# cost of that lagged turn -- image AND text -- measured with the trainer's own
# parser (_trailing_tool_tokens). Text matters: a browse page / multi-result search
# runs up to ~1.3k tok; reserving only images (the old behavior) let text-heavy
# trajectories overrun the window by exactly that lagged text. IMAGE_RESERVE_TOKENS
# is now only the per-image fallback upper bound (Qwen3-VL 2.16M px -> ~2109 merged
# vision tokens < 2176) used when the parser is unavailable.
#
# There is NO engine<->trainer per-image tokenizer drift: the rollout builds
# prompt_ids with the SAME Qwen3-VL processor as the trainer (verl_engine.py:69),
# so resp_used is already in trainer token space. FRAMING_MARGIN_TOKENS is now only
# a small fixed cushion for chat-template framing (the tool-response text itself is
# reserved exactly by _pending_lag). The real overflow bug was the FORCED-ANSWER
# GATE reading resp_used WITHOUT this one-turn lag, so it fired a turn too late and
# let extra crops/text pile past the window; the gate now adds the lagged reserve
# (see _run).
IMAGE_RESERVE_TOKENS = int(os.environ.get("VDR_IMAGE_RESERVE_TOKENS", "2176"))
FRAMING_MARGIN_TOKENS = int(os.environ.get("VDR_FRAMING_MARGIN_TOKENS", "512"))
MIN_ANSWER_TOKENS = int(os.environ.get("VDR_MIN_ANSWER_TOKENS", "256"))

DEEPRESEARCH_SYSTEM_PROMPT_TEXT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """


DEEPRESEARCH_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "crop_and_search", "description": "Crop some important local regions from an image and perform reverse image / visual search to identify objects, text, organizations, or other visual elements.", "parameters": {"type": "object", "properties": {"image_id": {"type": "string", "description": "The path or unique identifier of the image to analyze."}, "bbox": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "description": "Bounding box coordinates [x1, y1, x2, y2]."}, "minItems": 1, "description": "One or more important local regions to be cropped from the image."}, "goal": {"type": "string", "description": "The specific purpose of the visual search."}}, "required": ["image_id", "bbox", "goal"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """


def today_date():
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().date().strftime("%Y-%m-%d")


def _load_default_system_prompt() -> str:
    """Load the SFT full-tool prompt synchronized from final/unified_system_prompt.txt."""
    prompt_path = Path(__file__).with_name("mm_agent") / "unified_system_prompt.txt"
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        return f"{prompt}\n\nCurrent date: {today_date()}"
    except Exception as exc:  # noqa: BLE001
        print(f"[Prompt] Failed to load {prompt_path}: {exc}; falling back to built-in prompt.")
        return DEEPRESEARCH_SYSTEM_PROMPT + today_date()


def analyze_repetition_ngram(text: str, n: int = 30, threshold: float = 0.5):
    """
    Use N-grams to detect repetition in a string.

    Args:
        text (str): Input text to analyze.
        n (int): N-gram window size (default 10).
            - For long repetitive sequences, 10-20 is recommended.
        threshold (float): Distinct-N threshold (0~1).
            - Values below this indicate heavy repetition (default 0.5).

    Returns:
        bool: True if repetition is detected, False otherwise.
    """
    if not text or len(text) < n:
        print("text is too short, cannot analyze.")
        return False

    # 1. Generate N-grams (character-level sliding window).
    # List comprehension: slice from index i to i+n.
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]

    total_count = len(ngrams)
    if total_count == 0:
        return False

    # 2. Count frequencies.
    ngram_counts = Counter(ngrams)
    unique_count = len(ngram_counts)

    # 3. Compute Distinct-N (unique count / total count).
    # Repetitive text is typically < 0.4.
    distinct_ratio = unique_count / total_count

    # 4. Determine repetition.
    is_repetitive = distinct_ratio < threshold

    return is_repetitive


def count_words(text: str) -> int:
    # Match segments that look like English words.
    # Rule: starts and ends with a letter, may contain letters, apostrophes, or hyphens.
    pattern = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
    words = pattern.findall(text)
    return len(words)


def max_consecutive_wait_run(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"[A-Za-z]+", text.lower())
    run = best = 0
    for w in words:
        if w == "wait":
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def build_text_completion_prompt(
    messages: list[dict], allow_special: bool = True
) -> str:
    """
    Build text completion prompt from messages list.
    Adapted from qwen_agent.utils.utils.build_text_completion_prompt

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        allow_special: Whether to allow special tokens (for compatibility)

    Returns:
        Formatted prompt string
    """
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt_parts = []

    # Handle system message
    if messages and messages[0]["role"] == "system":
        sys_content = messages[0]["content"]
        prompt_parts.append(f"{im_start}system\n{sys_content}{im_end}")
        messages = messages[1:]

    # Ensure chat completes with assistant
    if messages and messages[-1]["role"] != "assistant":
        messages = messages + [{"role": "assistant", "content": ""}]

    # Format each message
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"{im_start}{role}\n{content}{im_end}")

    return "\n".join(prompt_parts)


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent adapted from Tongyi DeepResearch.

    This agent implements the core reasoning loop with tool calling capabilities,
    using rLLM's OpenAI engine for model inference.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        system_prompt: str | None = None,
        default_max_tries: int = 3,
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Args:
            rollout_engine: rLLM OpenAI engine for model inference
            tools: Dictionary of available tools {tool_name: tool_instance}
            system_prompt: Optional custom system prompt
        """
        self.rollout_engine = rollout_engine
        self.tools = tools or {}
        self.system_prompt = system_prompt
        # Configuration from original DeepResearch
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        self.default_max_tries = default_max_tries

        # Smart context management using actual API consumption
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # EXACT token cost of the trailing tool turn the engine has not counted yet
        # (one-turn lag); set at append time, consumed by the length gate/budget.
        self._pending_lag = 0

        self.max_prompt_tokens = MAX_PROMPT_LENGTH_PER_RUN
        self.max_response_tokens = MAX_RESPONSE_LENGTH_PER_RUN

        # Plan B: cumulative-response budget == training truncation window
        # (config.data.max_response_length), so the agent stops generating within
        # the SAME window the trainer keeps -> unifies generation with truncation.
        self.response_window = None
        _cfg = getattr(rollout_engine, "config", None)
        try:
            self.response_window = int(_cfg.data.max_response_length)
        except Exception:  # noqa: BLE001
            self.response_window = None
        self.answer_reserve = ANSWER_RESERVE_TOKENS

    def _prepare_tools_for_run(
        self,
        *,
        images: list | None = None,
        image_path: Any = None,
        task: Any = None,
        uid: Any = None,
        output_dir: str | None = None,
    ) -> None:
        """Initialize per-run state for tools that need shared image context."""
        seen_runtimes = set()
        for tool in self.tools.values():
            runtime = getattr(tool, "runtime", None)
            if runtime is not None:
                key = id(runtime)
                if key in seen_runtimes:
                    continue
                runtime.start_run(
                    image_path=image_path,
                    images=images,
                    task=task,
                    uid=uid,
                    output_dir=output_dir,
                )
                seen_runtimes.add(key)
            elif hasattr(tool, "start_run"):
                tool.start_run(
                    image_path=image_path,
                    images=images,
                    task=task,
                    uid=uid,
                    output_dir=output_dir,
                )

    def sanity_check_output(self, content: str) -> bool:
        """Check if the model output contains the expected thinking structure."""
        return "<think>" in content and "</think>" in content

    def _dynamic_response_budget(self, messages: list[dict] | None = None) -> int:
        """max_tokens for the next completion = min(per-turn cap, remaining window).

        remaining = response_window - resp_used - lagged_reserve - FRAMING_MARGIN_TOKENS
          * resp_used  : image-EXPANDED response tokens committed so far, measured
                         from the engine's own prompt/completion counts (accurate
                         for every prior turn except the most recent tool turn,
                         which the engine only counts on the NEXT call).
                         Same Qwen3-VL processor as the trainer -> no per-image drift.
          * lagged_reserve : self._pending_lag = EXACT token cost of that most recent
                         tool turn -- image AND text -- so a large text tool-response
                         (browse page / multi-result search) is reserved too, not
                         just images (a tool turn can also return several images).
          * FRAMING_MARGIN_TOKENS: small fixed cushion for chat-template framing.
        Guarantees resp_used + actual_last_turn + budget <= window - FRAMING_MARGIN,
        i.e. tokenize_and_mask_cumulative_mm never drops a trailing (answer) turn.
        Before the first engine response (no baseline yet) we fall back to the cap.
        """
        cap = int(self.max_response_tokens)
        init_len = getattr(self, "_initial_prompt_len", None)
        if not self.response_window or init_len is None:
            return cap
        resp_used = (
            int(self.total_prompt_tokens or 0)
            + int(self.total_completion_tokens or 0)
            - int(init_len)
        )
        # The tool turn just appended (messages[-1]) is not yet in resp_used (the
        # engine counts it on the NEXT call = one-turn lag). Reserve its EXACT
        # image+text token cost (self._pending_lag, measured at append time with
        # the SAME parser the trainer uses) -- not a worst-case constant, so small
        # crops waste nothing and large text tool-responses are no longer missed.
        # No engine<->trainer drift (rollout prompt_ids come from the same Qwen3-VL
        # processor, verl_engine.py:69).
        lagged_reserve = int(getattr(self, "_pending_lag", 0) or 0)
        remaining = (
            int(self.response_window)
            - resp_used
            - lagged_reserve
            - FRAMING_MARGIN_TOKENS
        )
        return int(max(MIN_ANSWER_TOKENS, min(cap, remaining)))

    def _trailing_tool_tokens(self, msg) -> int:
        """EXACT image-expanded token cost of a just-appended trailing turn (a tool
        response OR an injected reminder), computed with the SAME parser the trainer
        uses so it matches tokenize_and_mask_cumulative_mm to the token. Covers the
        one-turn lag for BOTH image AND text payloads: the engine only counts an
        appended turn on the NEXT call, so a large text tool-response (a browse page
        or multi-result search, measured up to ~1.3k tok here) would otherwise be
        invisible to the gate/budget for one turn -- exactly the residual that let
        text-heavy trajectories overrun the window. Falls back to a size-based upper
        bound if the parser is unavailable or errors, so the reserve never silently
        vanishes (text is estimated at ~3 chars/token, images at IMAGE_RESERVE each)."""
        if not isinstance(msg, dict):
            return 0
        n_imgs = len(msg["images"]) if isinstance(msg.get("images"), list) else 0
        parser = getattr(self.rollout_engine, "chat_parser", None)
        if parser is not None and hasattr(parser, "_encode_segment_mm"):
            try:
                text = parser.parse([msg], add_generation_prompt=False, is_first_msg=False)
                ids, _, _ = parser._encode_segment_mm(text, [msg])
                return int(len(ids))
            except Exception:  # noqa: BLE001
                pass
        content = msg.get("content", "")
        text_est = len(content) // 3 if isinstance(content, str) else 0
        return IMAGE_RESERVE_TOKENS * n_imgs + text_est

    async def call_server(
        self, messages: list[dict], max_tries: Optional[int] = None, **kwargs
    ):
        """Call rollout engine once; assumes XML ReAct format."""
        try:
            # Dynamic per-turn budget: clamp this completion so the image-expanded
            # cumulative response cannot exceed the training window (no truncation).
            # Pass messages so the reserve scales with the trajectory's image count
            # (per-image processor drift) and the trailing tool turn's image count.
            budget = self._dynamic_response_budget(messages)
            # Force per-round limits from DeepResearchAgent without local token estimation.
            if hasattr(self.rollout_engine, "max_prompt_length"):
                self.rollout_engine.max_prompt_length = int(self.max_prompt_tokens)
            if hasattr(self.rollout_engine, "max_response_length"):
                self.rollout_engine.max_response_length = int(budget)

            kwargs.pop("max_new_tokens", None)
            kwargs["max_tokens"] = int(budget)
            response = await self.rollout_engine.get_model_response(
                messages=messages, **kwargs
            )

            if hasattr(response, "prompt_length") and hasattr(
                response, "completion_length"
            ):
                self.total_prompt_tokens += response.prompt_length
                self.total_completion_tokens += response.completion_length

            # This turn's generation hit the per-call max_tokens cap (verl_engine.py
            # sets finish_reason="length" when completion_ids >= max_tokens). Sticky
            # for the whole episode: even if a later turn recovers with a clean
            # <answer>, the truncated turn's tokens are still in the trajectory and
            # must not be reinforced if the episode ends up reward=1 (see reward
            # capping in deepresearch_workflow.py).
            if getattr(response, "finish_reason", None) == "length":
                self._had_length_truncation = True

            # The prompt just sent to the engine counted any previously-lagged tool
            # image(s), so clear the pending one-turn-lag reserve.
            self._pending_lag = 0
            return response
        except Exception as exc:  # noqa: BLE001
            print(f"call_server failed: {exc}")
            raise

    def record_token_usage(self, response) -> None:
        """Record the latest prompt/completion token count from rollout engine."""
        prompt_tokens = getattr(response, "prompt_length", None)
        completion_tokens = getattr(response, "completion_length", None)

        if prompt_tokens is not None:
            try:
                self.total_prompt_tokens = int(prompt_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_prompt_tokens = 0

        if completion_tokens is not None:
            try:
                self.total_completion_tokens = int(completion_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_completion_tokens = 0

    def get_total_tokens_used(self) -> int:
        """Return the latest prompt + completion token usage reported by the engine."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def _estimate_prompt_tokens(self, messages: list[dict]) -> int:
        """Estimate prompt length for the next call using the rollout engine's tokenizer."""
        tokenizer = getattr(self.rollout_engine, "tokenizer", None)
        chat_parser = getattr(self.rollout_engine, "chat_parser", None)

        if tokenizer is None or chat_parser is None:
            return self.total_prompt_tokens

        try:
            prompt = chat_parser.parse(
                messages,
                add_generation_prompt=True,
                is_first_msg=True,
                tools=[],
                accumulate_reasoning=getattr(
                    self.rollout_engine, "accumulate_reasoning", False
                ),
            )
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return len(token_ids)
        except Exception as exc:  # noqa: BLE001
            print(f"[TokenEstimator] Failed to estimate prompt tokens: {exc}")
            return self.total_prompt_tokens

    def _normalize_tool_result(self, result: Any) -> tuple[str, list[Any]]:
        """Normalize tool outputs into text plus optional multimodal images.

        Supported return shapes:
        - str: plain text observation
        - dict: {"text"/"content"/"result": ..., "images"/"image": ...}
        - tuple/list: (text, images)
        - object: attributes such as .text/.content/.result and .images/.image

        Images are kept in the message-level format already consumed by the
        existing Qwen-VL chat parser: e.g. {"image": "/path/to/image.png"}.
        """

        text: Any = result
        images: Any = None

        if isinstance(result, (tuple, list)) and len(result) == 2:
            text, images = result
        elif isinstance(result, dict):
            text = (
                result.get("text")
                or result.get("content")
                or result.get("result")
                or result.get("message")
                or ""
            )
            images = result.get("images", result.get("image"))
        else:
            text = (
                getattr(result, "text", None)
                or getattr(result, "content", None)
                or getattr(result, "result", None)
                or result
            )
            images = getattr(result, "images", None)
            if images is None:
                images = getattr(result, "image", None)

        if images is None:
            normalized_images: list[Any] = []
        elif isinstance(images, list):
            normalized_images = images
        else:
            normalized_images = [images]

        normalized_images = [
            image if isinstance(image, dict) else {"image": image}
            for image in normalized_images
            if image is not None
        ]

        return str(text), normalized_images

    def _build_tool_response_message(self, result: Any) -> dict:
        """Build a ReAct tool response message, preserving returned images."""

        text, images = self._normalize_tool_result(result)
        message = {
            "role": "user",
            "content": f"<tool_response>\n{text}\n</tool_response>",
        }
        if images:
            message["images"] = images
        return message

    def _build_result(
        self,
        *,
        question: str,
        answer: str | None,
        messages: list[dict],
        prediction: str,
        termination: str,
        rounds: int,
        start_time: float,
        tool_error_classes: list | None = None,
        # next_prompt_tokens: int | None = None,
    ) -> dict:
        """Assemble result payload with token usage metadata."""
        token_usage = {
            "prompt": self.total_prompt_tokens,
            "completion": self.total_completion_tokens,
            "max_prompt": self.max_prompt_tokens,
        }

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": rounds,
            "time_taken": time.time() - start_time,
            "token_usage": token_usage,
            "tool_error_classes": list(tool_error_classes or []),
            "had_length_truncation": bool(getattr(self, "_had_length_truncation", False)),
        }
        return result

    async def _run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Main reasoning loop adapted from original DeepResearch.

        This is the core ReAct implementation that handles:
        - Multi-turn conversation
        - Tool calling and execution
        - Context length management
        - Termination conditions

        Args:
            question: The research question to answer
            answer: Ground truth answer (for evaluation)
            images: List of image data URLs (base64 encoded)

        Returns:
            Dictionary with results including messages, prediction, and termination reason
        """
        start_time = time.time()
        run_task = kwargs.pop("task", None)
        run_uid = kwargs.pop("uid", None)
        tool_output_dir = kwargs.pop("tool_output_dir", None)
        self._prepare_tools_for_run(
            images=images,
            image_path=image_path,
            task=run_task,
            uid=run_uid,
            output_dir=tool_output_dir,
        )

        system_prompt = (
            self.system_prompt + today_date() if self.system_prompt else _load_default_system_prompt()
        )

        if images:
            user_message = {
                "role": "user",
                "content": question,
                "images": images,
            }
        else:
            user_message = {"role": "user", "content": question}

        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]

        if not images:
            messages = [
                {"role": "system", "content": system_prompt},
                user_message,
            ]

        num_llm_calls_available = MAX_LLM_CALL_BY_TASK.get(
            _resolve_task_key(run_task), self.max_llm_calls
        )
        round = 0
        termination = None
        prediction = ""
        consecutive_infra_error = 0
        consecutive_model_error = 0
        format_bad_count = 0
        consecutive_empty_search = 0
        self._initial_prompt_len = None
        self._pending_lag = 0
        force_answer_pending = False
        tool_error_classes: List[str] = []
        force_answer_reason = ""
        left_panel_retry_used = False

        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1

            # Get model response from rollout engine
            try:
                response = await self.call_server(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001
                prediction = "call_server failed"
                termination = "error"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                    tool_error_classes=tool_error_classes,
                )

            # Synchronize token usage with rollout engine feedback
            self.record_token_usage(response)
            if self._initial_prompt_len is None:
                # First-turn prompt length == initial (system + question + source image),
                # image-expanded on the healthy path. Response budget is measured
                # relative to this so it matches cumulative_mm's prompt/response split.
                self._initial_prompt_len = int(self.total_prompt_tokens or 0)

            # Extract text content (may be None for pure function calling)
            content = (
                response.text if hasattr(response, "text") and response.text else ""
            )

            if "<tool_call>" in content:
                # Extract tool name for display
                if "python" in content.lower() and "<code>" in content:
                    pass
                elif '"name":' in content:
                    try:
                        tool_text = content.split("<tool_call>")[1].split(
                            "</tool_call>"
                        )[0]
                        tool_data = json5.loads(tool_text)
                        tool_name = tool_data.get("name", "Unknown")
                        if "arguments" in tool_data:
                            args_str = str(tool_data["arguments"])
                            pass
                        else:
                            pass
                    except Exception:
                        pass
                else:
                    pass

            # Clean up content if it contains tool_response
            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]

            if max_consecutive_wait_run(content) >= 8:
                messages.append(
                    {"role": "assistant", "content": content.strip(), "step_error": False}
                )
                prediction = ""
                termination = "wait_loop_detected"
                print("VDR_WAIT_GATE_FIRED")
                break

            # Only XML ReAct tool calls are supported.
            if "<tool_call>" in content and "</tool_call>" in content:
                #
                if "<answer>" in content and "</answer>" in content:
                    messages.append(
                        {"role": "assistant", "content": content.strip(), "step_error": False}
                    )
                    prediction = ""
                    termination = "mixed_tool_and_answer"
                    print("VDR_MIXED_GATE_FIRED")
                    break
                if force_answer_pending:
                    messages.append(
                        {"role": "assistant", "content": content.strip(), "step_error": False}
                    )
                    prediction = ""
                    termination = f"{force_answer_reason}_no_answer"
                    break
                # ReAct text format path
                assistant_message = {
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                }
                messages.append(assistant_message)
                tool_error = False
                step_err_cls = None

                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[
                    0
                ]
                # Legacy PythonInterpreter used a <code> block. The SFT protocol
                # uses JSON: {"name": "python", "arguments": {"code": "..."}},
                # so only take the legacy branch when a real <code> block exists.
                if "<code>" in tool_call_text and "</code>" in tool_call_text:
                    try:
                        code_raw = (
                            content.split("<tool_call>")[1]
                            .split("</tool_call>")[0]
                            .split("<code>")[1]
                            .split("</code>")[0]
                            .strip()
                        )
                        result = await self.execute_python(code_raw)
                        if isinstance(result, str) and result.startswith(
                            (
                                "Python execution error:",
                                "PythonInterpreter tool not available",
                                "PythonInterpreter tool is not callable",
                            )
                        ):
                            tool_error = True
                            tool_error_classes.append("model_error")
                            step_err_cls = "model_error"
                    except Exception:
                        result = (
                            "[Python Interpreter Error]: Python code formatting error."
                        )
                        tool_error = True
                        tool_error_classes.append("model_error")
                        step_err_cls = "model_error"
                else:
                    try:
                        # Parse JSON tool call
                        tool_call = json5.loads(tool_call_text)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})
                        if not isinstance(tool_args, dict):
                            tool_args = {}
                    except Exception:
                        result = MSG.E1_JSON_PARSE
                        tool_error_classes.append("model_error")
                        tool_error = True
                        step_err_cls = "model_error"
                    else:
                        result, call_ok, err_cls = await self.custom_call_tool(tool_name, tool_args)
                        if not call_ok and err_cls not in ("budget", "empty_result"):
                            tool_error = True
                            step_err_cls = err_cls
                        if err_cls:
                            tool_error_classes.append(err_cls)
                        if tool_name in ("text_search", "text_search_image"):
                            if err_cls == "empty_result":
                                consecutive_empty_search += 1
                            elif call_ok:
                                consecutive_empty_search = 0

                if tool_error:
                    assistant_message["step_error"] = True
                    assistant_message["err_cls"] = step_err_cls

                # Add tool response in ReAct format. If a tool returns images,
                # preserve them as real multimodal inputs for the next turn.
                messages.append(self._build_tool_response_message(result))
                # Measure this (still-uncounted) tool turn's EXACT token cost for the
                # length gate/budget; offloaded so the processor pass does not block
                # the shared rollout event loop.
                self._pending_lag = await asyncio.to_thread(
                    self._trailing_tool_tokens, messages[-1])
                if assistant_message["step_error"]:
                    if step_err_cls == "infra_error":
                        consecutive_infra_error += 1
                        consecutive_model_error = 0
                    elif step_err_cls == "model_error":
                        consecutive_model_error += 1
                        consecutive_infra_error = 0
                else:
                    consecutive_infra_error = 0
                    consecutive_model_error = 0
                if _EMPTY_SEARCH_GATE_N and consecutive_empty_search >= _EMPTY_SEARCH_GATE_N:
                    prediction = ""
                    termination = "consecutive_empty_search"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                        tool_error_classes=tool_error_classes,
                    )
                if consecutive_infra_error >= 3:
                    print(
                        f"VDR_INFRA_GATE_FIRED consecutive_infra_error={consecutive_infra_error} "
                        f"classes={tool_error_classes[-3:]}"
                    )
                    prediction = "Too many consecutive infra errors."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    tool_error_classes=tool_error_classes,
                    )
                if consecutive_model_error >= 3:
                    print(
                        f"VDR_MODELERR_GATE_FIRED consecutive_model_error={consecutive_model_error}"
                    )
                    prediction = ""
                    termination = "model_error_loop"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    tool_error_classes=tool_error_classes,
                    )

            # Check for final answer AFTER processing tools
            # This allows o3 to execute tools even when it includes answer in same message
            elif "<answer>" in content and "</answer>" in content:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": False,
                    }
                )
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
                termination = "answer"

                if _is_spot_diff(run_task) and _has_right_panel_box(prediction):
                    if not left_panel_retry_used:
                        left_panel_retry_used = True
                        messages.append({"role": "user", "content": MSG.D4_LEFT_PANEL})
                        force_answer_pending = True
                        force_answer_reason = "left_panel"
                        num_llm_calls_available = max(num_llm_calls_available, 1)
                        termination = None
                        prediction = ""
                        self._pending_lag = await asyncio.to_thread(
                            self._trailing_tool_tokens, messages[-1])
                        print("VDR_LEFTPANEL_GATE_FIRED")
                        continue
                    prediction = ""
                    termination = "left_panel_violation"
                break

            # Priority 3: No tool call and answer, just reasoning or format error
            else:
                is_repetitive = analyze_repetition_ngram(content)
                is_overlong = count_words(content) > 2500
                if is_repetitive and is_overlong:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.strip(),
                            "step_error": True,
                        }
                    )
                    prediction = "Repetition response"
                    termination = "repetition_detected"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    tool_error_classes=tool_error_classes,
                    )

                observation = "Error: Invalid content format. Content must contain <tool_call> or <answer> tags. Let's try again."
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": True,
                    }
                )
                messages.append({"role": "user", "content": observation})
                format_bad_count += 1
                if format_bad_count >= 2:
                    print(f"VDR_FORMAT_GATE_FIRED format_bad_count={format_bad_count}")
                    prediction = ""
                    termination = "bad_format_no_answer"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    tool_error_classes=tool_error_classes,
                    )

            # Plan B: cumulative-length gate — unify generation with the training
            # truncation window. When the image-expanded response so far nears the
            # window, force ONE final-answer turn (no more tools) so the trajectory
            # ends WITH its answer INSIDE the window the trainer keeps. The forced
            # turn's max_tokens is set by _dynamic_response_budget() (= remaining
            # window, no fixed 4096), which GUARANTEES the answer fits without the
            # trainer dropping trailing turns; answer_reserve here only decides HOW
            # EARLY to stop calling tools (leaving room for a real answer).
            if (
                termination is None
                and not force_answer_pending
                and self.response_window
                and self._initial_prompt_len is not None
                and num_llm_calls_available >= 1
            ):
                resp_so_far = (
                    int(self.total_prompt_tokens or 0)
                    + int(self.total_completion_tokens or 0)
                    - self._initial_prompt_len
                )
                # The tool turn just appended is not yet in the engine's prompt_length
                # (one-turn lag). Add its EXACT token cost (self._pending_lag) so the
                # gate fires BEFORE the true image-expanded length overruns the window
                # -- but no earlier than the real length requires (no padding).
                effective_resp_so_far = resp_so_far + int(getattr(self, "_pending_lag", 0) or 0)
                if effective_resp_so_far >= self.response_window - self.answer_reserve:
                    messages.append({"role": "user", "content": MSG.D3_LENGTH_BUDGET})
                    force_answer_pending = True
                    force_answer_reason = "length_budget"
                    num_llm_calls_available = min(num_llm_calls_available, 1)
                    print(
                        f"VDR_LEN_GATE_FIRED resp_so_far={resp_so_far} "
                        f"window={self.response_window} reserve={self.answer_reserve}"
                    )

            if (
                termination is None
                and not force_answer_pending
                and num_llm_calls_available == 1
                and "<answer>" not in content
            ):
                messages.append({"role": "user", "content": MSG.D2_LAST_TURN})
                force_answer_pending = True
                force_answer_reason = "turn_budget"
                self._pending_lag = await asyncio.to_thread(
                    self._trailing_tool_tokens, messages[-1])
                print(f"VDR_TURN_GATE_FIRED round={round} max_calls={self.max_llm_calls}")

            # Determine whether another round is feasible
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                prediction = f"No answer found after {self.max_llm_calls} rounds."
                termination = f"answer not found after {self.max_llm_calls} rounds"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                    tool_error_classes=tool_error_classes,
                )

        #
        if termination is None:
            last_message_content = (
                messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
            )
            if last_message_content and "<answer>" in last_message_content:
                prediction = last_message_content.split("<answer>")[1].split("</answer>")[0]
                termination = "answer"
            else:
                prediction = "No answer found."
                termination = "answer not found"
                if num_llm_calls_available == 0:
                    termination = "exceed available llm calls"

        result = self._build_result(
            question=question,
            answer=answer,
            messages=messages,
            prediction=prediction,
            termination=termination,
            rounds=round,
            start_time=start_time,
                    tool_error_classes=tool_error_classes,
        )

        print("\n🏁 DeepResearch completed:")
        print(f"   Rounds: {round}")
        print(f"   Time: {result['time_taken']:.1f}s")
        print(f"   Termination: {termination}")
        print(
            "   Token usage: prompt={prompt}, completion={completion}, max_prompt={max_prompt}".format(
                prompt=self.total_prompt_tokens,
                completion=self.total_completion_tokens,
                max_prompt=self.max_prompt_tokens,
            )
        )
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> str:
        """
        Execute tool calls with the available tools.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        if tool_name not in self.tools:
            return (MSG.A5_UNKNOWN_TOOL.format(got=tool_name), False, "model_error")
        if True:
            try:
                # Call the tool
                if hasattr(self.tools[tool_name], "call"):
                    # Async tool
                    if asyncio.iscoroutinefunction(self.tools[tool_name].call):
                        result = await self.tools[tool_name].call(**tool_args)
                    else:
                        result = self.tools[tool_name].call(**tool_args)
                elif callable(self.tools[tool_name]):
                    # Direct callable
                    result = self.tools[tool_name](**tool_args)
                else:
                    result = MSG.E2_TOOL_EXCEPTION.format(tool=tool_name, exc_type="NotCallable")
                    return (result, False, "model_error")

                if isinstance(result, dict):
                    return (result, bool(result.get("ok", True)), result.get("error_class"))
                return (result, True, None)

            except Exception as e:
                return (MSG.E2_TOOL_EXCEPTION.format(tool=tool_name, exc_type=type(e).__name__),
                        False, "model_error")

    async def execute_python(self, code: str) -> str:
        """
        Execute Python code using the PythonInterpreter tool.

        Args:
            code: Python code to execute

        Returns:
            Execution result as string
        """
        if "PythonInterpreter" in self.tools:
            try:
                # Use the PythonInterpreter tool
                tool = self.tools["PythonInterpreter"]
                if hasattr(tool, "call"):
                    if asyncio.iscoroutinefunction(tool.call):
                        result = await tool.call(code=code)
                    else:
                        result = tool.call(code=code)
                    return str(result)
                else:
                    return "PythonInterpreter tool is not callable"
            except Exception as e:
                return f"Python execution error: {e}"
        else:
            return "PythonInterpreter tool not available"

    def reset(self):
        """Reset the agent state (for compatibility with rLLM workflow)."""
        # Reset token counters for each new task
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._pending_lag = 0
        self._had_length_truncation = False

    async def run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Public interface for running the agent.

        Args:
            question: Research question to answer
            answer: Ground truth answer (optional, for evaluation)

        Returns:
            Result dictionary
        """
        # Reset token counters for each new run
        self.reset()
        return await self._run(question, answer, images, image_path, **kwargs)


DeepResearchAgent = MultiTurnReactAgent
