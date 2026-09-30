"""LLM summarizer for the browse tool (OpenAI-compatible chat endpoint).

The SFT `browse` tool summarized each fetched page with an LLM (via an internal
API) before returning it to the agent. This eval reproduces that behavior against
a *configurable* OpenAI-compatible model instead
of the internal API, so no internal dependency is needed.

When no endpoint/model is configured, `build_tools` leaves the browse tool without
a summarizer and it degrades to returning raw truncated page text.
"""
from __future__ import annotations

import time
from typing import Optional

import requests


class SummaryClient:
    """Minimal OpenAI-compatible chat client used only to summarize page text.

    `summarize(prompt)` returns the model's answer. Reasoning models (e.g.
    qwen3.5) put the answer in `content` and their scratch-work in
    `reasoning_content`; we take `content`, falling back to `reasoning_content`
    only if `content` is empty. `max_tokens` is generous because reasoning tokens
    are billed against the completion budget.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        timeout: int = 90,
        max_retries: int = 3,
    ):
        if not base_url or not model:
            raise ValueError("SummaryClient requires base_url and model")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key or ""
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def summarize(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                text = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
                if text:
                    return text
                raise RuntimeError("empty summary from model")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"summarize failed after {self.max_retries} tries: {last_err}")
