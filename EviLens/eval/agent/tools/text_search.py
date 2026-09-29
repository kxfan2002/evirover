"""text_search backends. Two providers, ONE identical output envelope, so the
only thing that differs between them is the underlying search index/ranking —
letting an eval isolate "which search API" as a clean variable.

- TextSearchTool        -> Perplexity Search API (POST api.perplexity.ai/search).
  Raw search endpoint (not Sonar): returns results[] (title, url, snippet, date)
  with no LLM synthesis, matching the separated search/browse design the model
  was trained on.
- SerperTextSearchTool   -> Serper web search (POST google.serper.dev/search).
  Returns organic[] (title, link, snippet); mapped into the SAME numbered-list
  envelope. Uncapped by design (Perplexity is uncapped in the baseline), so the
  scarce image-tool Serper budget is untouched and the two backends are compared
  on result quality alone.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

import requests

from . import messages as MSG
from .base import (ERR_EMPTY, ERR_INFRA, ERR_MODEL, ToolContext,
                   ToolResult, http_post_json)

PPLX_SEARCH_URL = "https://api.perplexity.ai/search"
SERPER_SEARCH_URL = "https://google.serper.dev/search"


def _parse_query(args: Dict[str, Any]) -> str:
    raw_query = args.get("query")
    if raw_query is None and isinstance(args.get("queries"), list) and args["queries"]:
        raw_query = args["queries"][0]
    return str(raw_query or "").strip()


def _format_results(query: str, results: List[Dict[str, Any]]) -> str:
    """Render a results list in the numbered-list envelope the SFT tool produced.

    Shared by both backends so the model sees a byte-identical format regardless
    of provider. Each item may key its url as 'url' or 'link', snippet as
    'snippet' or 'description'.
    """
    lines = [f"--- text search result for [{query}] ---"]
    for idx, item in enumerate(results, 1):
        title = item.get("title") or "Untitled"
        link = item.get("url") or item.get("link") or ""
        snippet = " ".join(str(item.get("snippet") or item.get("description") or "").split())
        lines.append(f"{idx}. [{title}]({link})")
        if snippet:
            lines.append(f"   {snippet}")
    lines.append("--- end text search result ---")
    return "\n".join(lines)


def _no_results(query: str, error: str) -> ToolResult:
    return ToolResult(text=(
        f"--- text search result for [{query}] ---\n"
        f"No results. {error}\n"
        "--- end text search result ---"
    ))


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
        query = _parse_query(args)
        top_k = max(1, min(10, int(args.get("top_k", 7))))
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
            return _no_results(query, error)
        return ToolResult(text=_format_results(query, results), raw=results)


class SerperTextSearchTool:
    """text_search backed by Serper /search (drop-in for TextSearchTool).

    Uncapped: uses the Serper key directly but does NOT touch the image-tool
    SerperBudget, so an all-Serper eval keeps image search fair while text search
    stays as freely available as Perplexity is in the baseline.
    """
    name = "text_search"

    def __init__(self, serper_key: str = ""):
        self.serper_key = serper_key or ""

    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_query = args.get("query")
        if raw_query is None and isinstance(args.get("queries"), list) and args["queries"]:
            raw_query = args["queries"][0]
        query = str(raw_query or "").strip()
        # ---- always strip double quotes from the query ----
        # Quotes mean exact-phrase matching, and a fine-tuned model can drift into
        # quoting every descriptive phrase, which recalls almost nothing on real
        # pages. Quotes have never helped this tool, so they are removed.
        if '"' in query:
            query = re.sub(r"\s+", " ", query.replace('"', " ")).strip()
        top_k = max(1, min(10, int(args.get("top_k", 10))))

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
            # A missing key is infra_error, not an empty result: otherwise a key
            # that expires mid-run is invisible in the metrics.
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
            # A genuine miss -- a legitimate world state, kept distinct from the
            # failures above so the model rephrases instead of giving up.
            return wrap(MSG.C1_NO_RESULTS.format(tool="text_search", query=query), False, ERR_EMPTY)

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
