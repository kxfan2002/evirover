"""text_search -> Perplexity Search API (POST https://api.perplexity.ai/search).

This is the *raw* search endpoint (not Sonar): it returns a structured
results[] array (title, url, snippet, date) with no LLM synthesis, matching the
separated search/browse design the model was trained on. Output is wrapped in
the same numbered-list envelope the SFT `text_search` tool produced.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

from . import messages as MSG
from .base import (ERR_EMPTY, ERR_INFRA, ERR_MODEL, ToolContext,
                   ToolResult, http_post_json)

PPLX_SEARCH_URL = "https://api.perplexity.ai/search"
SERPER_SEARCH_URL = "https://google.serper.dev/search"


class SerperTextSearchTool:

    name = "text_search"

    def __init__(self, serper_key: str = ""):
        self.serper_key = serper_key or ""

    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_query = args.get("query")
        if raw_query is None and isinstance(args.get("queries"), list) and args["queries"]:
            raw_query = args["queries"][0]
        query = str(raw_query or "").strip()
        top_k = max(1, min(10, int(args.get("top_k", 6))))

        def wrap(msg: str, ok: bool, cls=None, body: str = "") -> ToolResult:
            return ToolResult(
                text=f"--- text search result for [{query}] ---\n{body or msg}\n"
                     f"--- end text search result ---",
                ok=ok, error_class=cls,
            )

        if not query:
            return wrap(MSG.A4_MISSING_ARG.format(tool="text_search", arg="query"),
                        False, ERR_MODEL)
        if not self.serper_key:
            return wrap(MSG.B4_NO_KEY.format(tool="text_search"), False, ERR_INFRA)

        data, err, code = http_post_json(
            SERPER_SEARCH_URL,
            {"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
            {"q": query, "num": top_k},
        )
        if err == "rate_limited":
            return wrap(MSG.B1_RATE_LIMITED.format(tool="text_search", tries=10), False, ERR_INFRA)
        if err == ERR_INFRA:
            return wrap(
                MSG.B2_SERVICE_ERROR.format(tool="text_search", code=code) if code
                else MSG.B3_NETWORK_ERROR.format(tool="text_search"),
                False, ERR_INFRA,
            )

        organic = [x for x in (data.get("organic") or []) if isinstance(x, dict)][:top_k]
        if not organic:
            tmpl = MSG.C1_NO_RESULTS_QUOTED if '"' in query else MSG.C1_NO_RESULTS_PLAIN
            return wrap(tmpl.format(tool="text_search"), False, ERR_EMPTY)

        lines = []
        for idx, item in enumerate(organic, 1):
            title = item.get("title") or "Untitled"
            link = item.get("link") or item.get("url") or ""
            snippet = " ".join(str(item.get("snippet") or item.get("description") or "").split())
            lines.append(f"{idx}. [{title}]({link})")
            if snippet:
                lines.append(f"   {snippet}")
        return ToolResult(
            text=f"--- text search result for [{query}] ---\n" + "\n".join(lines) +
                 "\n--- end text search result ---",
            raw=organic,
        )


class TextSearchTool:
    name = "text_search"

    def __init__(self, pplx_key: str = ""):
        self.pplx_key = pplx_key or ""

    def _results(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            r = data.get("results")
            if isinstance(r, list):
                return [x for x in r if isinstance(x, dict)]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_query = args.get("query")
        if raw_query is None and isinstance(args.get("queries"), list) and args["queries"]:
            raw_query = args["queries"][0]
        query = str(raw_query or "").strip()
        top_k = max(1, min(10, int(args.get("top_k", 6))))
        if not query:
            return ToolResult(text="[text_search] query is required.")
        if not self.pplx_key:
            return ToolResult(text=(
                f"--- text search result for [{query}] ---\n"
                "text_search unavailable: no Perplexity API key configured.\n"
                "--- end text search result ---"
            ))

        headers = {"Authorization": f"Bearer {self.pplx_key}", "Content-Type": "application/json"}
        payload = {"query": query, "max_results": top_k}
        results: List[Dict[str, Any]] = []
        error = ""
        for attempt in range(5):
            try:
                resp = requests.post(PPLX_SEARCH_URL, headers=headers, json=payload, timeout=45)
                if resp.status_code == 429:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    break
                results = self._results(resp.json())[:top_k]
                break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                time.sleep(min(2 ** attempt, 10))

        if not results:
            return ToolResult(text=(
                f"--- text search result for [{query}] ---\n"
                f"No results. {error}\n"
                "--- end text search result ---"
            ))

        lines = [f"--- text search result for [{query}] ---"]
        for idx, item in enumerate(results, 1):
            title = item.get("title") or "Untitled"
            link = item.get("url") or item.get("link") or ""
            snippet = " ".join(str(item.get("snippet") or item.get("description") or "").split())
            lines.append(f"{idx}. [{title}]({link})")
            if snippet:
                lines.append(f"   {snippet}")
        lines.append("--- end text search result ---")
        return ToolResult(text="\n".join(lines), raw=results)
