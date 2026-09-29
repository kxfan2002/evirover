"""Multi-turn, image-interleaved chat client for the agent loop.

Like eval.client.ChatClient but for a running conversation: it takes a full
`messages` list (system + user/assistant/tool turns, some carrying images as
base64 data-URL content parts) and returns one assistant message. The runner
owns the message list and image interleaving; this class only does the HTTP
call, retries, and reasoning/content merging.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .. import config


@dataclass
class AgentResponse:
    content: str
    reasoning_content: str
    finish_reason: Optional[str]
    retries: int
    raw_message: Dict[str, Any]

    def assistant_text(self) -> str:
        """The text used to parse <tool_call>/<answer>.

        If content already carries <think>, use it as-is. Otherwise fold any
        separate reasoning_content into a <think> block so downstream parsing
        (which scans the whole string) still finds tool calls/answers that some
        reasoning models emit only in reasoning_content.
        """
        content = (self.content or "").strip()
        reasoning = (self.reasoning_content or "").strip()
        if "<think" in content.lower():
            return content
        if reasoning and "<tool_call>" not in content and "<answer>" not in content.lower():
            return f"<think>\n{reasoning}\n</think>\n{content}".strip()
        if reasoning and content:
            return f"{content}\n{reasoning}"
        return content or reasoning


class AgentChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = config.DEFAULT_MAX_TOKENS,
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_p: float = config.DEFAULT_TOP_P,
        top_k: int = config.DEFAULT_TOP_K,
        timeout: int = config.DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = config.DEFAULT_MAX_RETRIES,
        max_retries_429: int = config.DEFAULT_MAX_RETRIES_429,
        # Separate retry budget for empty (zero-token) responses.
        max_retries_empty: int = config.DEFAULT_MAX_RETRIES_EMPTY,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retries_429 = max_retries_429
        self.max_retries_empty = max_retries_empty

    def chat(self, messages: List[Dict[str, Any]]) -> AgentResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        # Pinned explicitly so all models are compared at identical sampling.
        # (None => omit, for endpoints that reject a given field.)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Optional[Exception] = None
        attempt = 0
        rl_attempt = 0
        empty_attempt = 0
        while attempt < self.max_retries:
            try:
                resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 429:
                    last_err = RuntimeError(f"HTTP 429: {resp.text[:200]}")
                    rl_attempt += 1
                    if rl_attempt <= self.max_retries_429:
                        time.sleep(min(5 * rl_attempt, 60))
                        continue
                    raise last_err
                if resp.status_code >= 500:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]
                finish = choice.get("finish_reason")
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                # Retry regardless of finish_reason: the common case is "stop", the
                # model having sampled EOS as its first token. Has its own budget.
                if not (content.strip() or reasoning.strip()):
                    empty_attempt += 1
                    if empty_attempt <= self.max_retries_empty:
                        time.sleep(min(2 ** empty_attempt, 10))
                        continue
                    # Budget exhausted: return the empty response rather than
                    # raising. This is model behaviour, not an API fault.
                return AgentResponse(
                    content=content,
                    reasoning_content=reasoning,
                    finish_reason=finish,
                    retries=attempt,
                    raw_message=msg,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"agent chat failed after {self.max_retries} tries: {last_err}")


class MockAgentClient:
    """Offline stand-in for --dry-run: exercises the full agent loop with no API.

    First call emits a local-tool call (crop) so image feedback is tested; the
    second call emits a family-appropriate <answer>. State is per-conversation,
    keyed on the number of assistant turns already present in `messages`.
    """

    def __init__(self, *args, **kwargs):
        self.model = "mock-agent"

    def chat(self, messages):
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        # Reconstruct the user task text (first user turn) to pick an answer shape.
        user_text = ""
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    user_text = c
                elif isinstance(c, list):
                    user_text = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                break

        if n_assistant == 0:
            content = ("<think>mock: inspect a region first</think>\n<tool_call>\n"
                       "{\"name\": \"crop\", \"arguments\": {\"image_id\": \"ORIGINAL\", "
                       "\"boxes\": [100, 100, 400, 400]}}\n</tool_call>")
            return AgentResponse(content=content, reasoning_content="",
                                 finish_reason="stop", retries=0, raw_message={"content": content})

        if "segmentation prompt JSON" in user_text or "positive_points" in user_text:
            ans = ("{\"boxes\":[100,100,400,400],\"positive_points\":[[200,200],[250,250],"
                   "[300,300]],\"negative_points\":[[10,10],[20,20],[30,30]]}")
        elif "find-the-difference" in user_text or "LEFT panel" in user_text:
            ans = "[[100,100,150,150],[200,200,260,260]]"
        elif "How many" in user_text or user_text.strip().endswith("?"):
            ans = "1"
        else:
            ans = "[100, 100, 400, 400]"
        content = f"<think>mock: answer</think>\n<answer>{ans}</answer>"
        return AgentResponse(content=content, reasoning_content="",
                             finish_reason="stop", retries=0, raw_message={"content": content})
