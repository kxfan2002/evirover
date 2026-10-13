"""browse -> self-fetch a URL, then LLM-summarize it against the query.

Mirrors the SFT browse tool end-to-end: fetch strategy (direct request -> free
Jina -> authenticated Jina), then summarize the cleaned page text with an LLM
answering the query (configurable OpenAI-compatible model, default Novita
qwen/qwen3.5-27b). Falls back to raw truncated text when no summarizer is set.
"""
from __future__ import annotations

import os
from html.parser import HTMLParser
from typing import Any, Dict, Optional

import requests

from . import messages as MSG
from .base import ERR_INFRA, ERR_MODEL, ToolContext, ToolResult
from .summarizer import SummaryClient

DEFAULT_JINA_API_KEY = os.getenv("JINA_API_KEY") or ""

JINA_HTTPS_PROXY = os.getenv("JINA_HTTPS_PROXY") or ""
_JINA_PROXIES = {"http": JINA_HTTPS_PROXY, "https": JINA_HTTPS_PROXY} if JINA_HTTPS_PROXY else None
MIN_FETCH_CHARS = 300
MAX_RETURN_CHARS = max(0, int(os.getenv("MM_BROWSE_MAX_OUTPUT_CHARS", "3000")))
MAX_FETCH_TOKENS = int(os.getenv("MM_BROWSE_MAX_FETCH_TOKENS", "24000"))
_CHARS_PER_TOKEN_FALLBACK = 3.2


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            text = " ".join((data or "").split())
            if text:
                self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def _extract(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            tag.decompose()
        chunks = []
        title = soup.find("title")
        if title and title.get_text(strip=True):
            chunks.append("Title: " + title.get_text(" ", strip=True))
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            chunks.append("Description: " + str(meta.get("content")).strip())
        for node in soup.find_all(["h1", "h2", "h3", "p", "li", "td", "th"]):
            text = node.get_text(" ", strip=True)
            if text and len(text) >= 20:
                chunks.append(text)
        return "\n".join(chunks)
    except Exception:
        parser = _TextExtractor()
        parser.feed(html)
        return parser.text()


def _direct_fetch(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=40)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "text" not in ctype and "html" not in ctype and "json" not in ctype:
        return resp.text[:120000]
    return _extract(resp.text)[:120000]


def _jina_fetch(url: str, api_key: str = "") -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(f"https://r.jina.ai/{url}", headers=headers,
                        timeout=60, proxies=_JINA_PROXIES)
    resp.raise_for_status()
    return resp.text[:120000]


class BrowseTool:
    name = "browse"

    def __init__(self, summarizer: Optional[SummaryClient] = None,
                 max_content_tokens: int = MAX_FETCH_TOKENS):
        self.summarizer = summarizer
        self.max_content_tokens = max_content_tokens

    def _truncate_content(self, text: str) -> str:
        return text[: int(self.max_content_tokens * _CHARS_PER_TOKEN_FALLBACK)]

    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = str(args.get("url") or "").strip()
        query = str(args.get("query") or "").strip() or "Extract the information relevant to the current task."
        if not url:
            return ToolResult(text=MSG.A4_MISSING_ARG.format(tool="browse", arg="url"),
                              ok=False, error_class=ERR_MODEL)

        source = ""
        fetch_method = "direct"
        errors = []
        for method, fn in (
            ("direct", lambda: _direct_fetch(url)),
            ("jina", lambda: _jina_fetch(url)),
            ("jina_auth", lambda: _jina_fetch(url, DEFAULT_JINA_API_KEY) if DEFAULT_JINA_API_KEY else ""),
        ):
            try:
                text = fn()
                if len(text.strip()) >= MIN_FETCH_CHARS:
                    source, fetch_method = text, method
                    break
                errors.append(f"{method}: too little content")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{method}: {exc}")

        if not source.strip():
            reason = (errors[-1].split(":", 1)[0] if errors else "unknown")
            return ToolResult(
                text=(f"--- browse result ---\nurl: {url}\n"
                      f"{MSG.B5_FETCH_FAILED.format(reason=reason)}\n"
                      "--- end browse result ---"),
                ok=False, error_class=ERR_INFRA)

        source = source.strip()

        if self.summarizer is not None:
            content = self._truncate_content(source)
            prompt = (
                "Read the webpage content below and answer the query in English.\n"
                "Return concise useful evidence. If the page does not contain relevant "
                "information, say so clearly.\n\n"
                f"URL: {url}\n"
                f"Query: {query}\n\n"
                f"--- webpage content ---\n{content}\n--- end webpage content ---"
            )
            try:
                summary = self.summarizer.summarize(prompt).strip()
                text = (
                    f"--- browse result ---\n"
                    f"url: {url}\n"
                    f"query: {query}\n"
                    f"fetch_method: {fetch_method}\n"
                    f"summary:\n{summary}\n"
                    f"--- end browse result ---"
                )
                return ToolResult(text=text, raw={"url": url, "query": query,
                                                  "fetch_method": fetch_method, "summarized": True})
            except Exception as _exc:  # noqa: BLE001
                import sys as _sys
                print(f"[browse] summarize failed, falling back to raw text: {_exc!r}",
                      file=_sys.stderr, flush=True)
                summary_error = f"{type(_exc).__name__}: {_exc}"
        else:
            summary_error = None

        body = source[:MAX_RETURN_CHARS]
        trunc_note = (
            f"\n[content truncated to {MAX_RETURN_CHARS} chars — refine your query or "
            "browse a more specific URL if you need more detail.]"
            if len(source) > MAX_RETURN_CHARS else ""
        )
        text = (
            f"--- browse result ---\n"
            f"url: {url}\n"
            f"query: {query}\n"
            f"fetch_method: {fetch_method}\n"
            f"content:\n{body}{trunc_note}\n"
            f"--- end browse result ---"
        )
        return ToolResult(text=text, raw={"url": url, "query": query,
                                          "fetch_method": fetch_method, "summarized": False,
                                          "summary_error": summary_error})
