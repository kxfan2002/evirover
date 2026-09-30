"""browse -> self-fetch a URL, then LLM-summarize it against the query.

Mirrors the SFT browse tool end-to-end: fetch strategy (direct request -> free
Jina -> authenticated Jina), then summarize the cleaned page text with an LLM
answering the query (SFT did this via an internal API; here it's a configurable
OpenAI-compatible model, injected as `summarizer`).

If no summarizer is configured (or a summarization call fails), browse degrades
gracefully to returning the raw cleaned page text (truncated).
"""
from __future__ import annotations

import os
from html.parser import HTMLParser
from typing import Any, Dict, Optional

import requests

from . import messages as MSG
from .base import ERR_INFRA, ERR_MODEL, ToolContext, ToolResult
from ..summarizer import SummaryClient

DEFAULT_JINA_API_KEY = os.getenv("JINA_API_KEY", "")
MIN_FETCH_CHARS = 300
MAX_RETURN_CHARS = 6000
# Cap the fetched page text handed to the summarizer. Honored by TOKENS when a
# tokenizer is available (language-independent); otherwise by a char fallback.
MAX_FETCH_TOKENS = 24000
# ~3.2 chars/token measured on English web text (Qwen tokenizer); used only when
# no tokenizer is available so browse never over-feeds the summarizer.
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


def _classify(errors: list) -> str:
    """Reduce the fetch attempts to one label that says why, not to the name of
    whichever method happened to run last.

    The categories are the ones that lead to different actions:
      unreachable -> network or proxy problem, not a key problem
      auth        -> 401/403, check the key and its quota
      rate_limit  -> 429, back off or rotate the key
      not_found   -> 4xx, the page is gone; the model should try another URL
      empty       -> connected, but the body is too short (anti-bot or JS page)
    The last exception type is appended so nothing is lost without feeding the
    model three tracebacks.
    """
    blob = " | ".join(errors).lower()
    if "too little content" in blob and not any(
            k in blob for k in ("connection", "timeout", "resolve", "unreachable", "proxy")):
        return "empty_content"
    for pat, tag in (
        (("newconnectionerror", "connectionerror", "max retries", "failed to resolve",
          "name or service not known", "connection refused", "proxyerror", "timed out",
          "timeout"), "unreachable"),
        (("401", "403", "unauthorized", "forbidden", "invalid api key"), "auth"),
        (("429", "rate limit", "too many requests"), "rate_limit"),
        (("404", "410", "not found"), "not_found"),
    ):
        if any(p in blob for p in pat):
            return tag
    last = errors[-1] if errors else "unknown"
    return last.split(":", 2)[-1].strip()[:40] or "unknown"


def _jina_fetch(url: str, api_key: str = "") -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.text[:120000]


class BrowseTool:
    name = "browse"

    def __init__(self, summarizer: Optional[SummaryClient] = None,
                 tokenizer: Any = None, max_content_tokens: int = MAX_FETCH_TOKENS):
        # When None, browse returns raw cleaned page text (no LLM summarization).
        self.summarizer = summarizer
        # HF tokenizer (optional): when set, fetched page text is truncated to
        # `max_content_tokens` TOKENS before summarization (language-independent).
        self.tokenizer = tokenizer
        self.max_content_tokens = max_content_tokens

    def _truncate_content(self, text: str) -> str:
        """Cap the page text to max_content_tokens tokens (or a char fallback)."""
        if self.tokenizer is not None:
            try:
                ids = self.tokenizer.encode(text)
                if len(ids) > self.max_content_tokens:
                    ids = ids[: self.max_content_tokens]
                    return self.tokenizer.decode(ids)
                return text
            except Exception:  # noqa: BLE001 — fall back to char cap on any tokenizer error
                pass
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
            # Return on fetch failure without going through the summarizer. The
            # reason must explain why it failed rather than name the last method
            # tried: collapsing unreachable, auth and empty into one label sends
            # debugging in entirely the wrong direction.
            reason = _classify(errors)
            return ToolResult(ok=False, error_class=ERR_INFRA, text=(
                f"--- browse result ---\nurl: {url}\n"
                f"{MSG.B5_FETCH_FAILED.format(reason=reason)}\n"
                "--- end browse result ---"
            ))

        source = source.strip()

        # Preferred path: LLM-summarize the page against the query (SFT parity).
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
            except Exception as exc:  # noqa: BLE001 — never let a summarizer hiccup kill the browse
                body = source[:MAX_RETURN_CHARS]
                text = (
                    f"--- browse result ---\n"
                    f"url: {url}\n"
                    f"query: {query}\n"
                    f"fetch_method: {fetch_method}\n"
                    f"[summarizer error: {exc}; returning raw page text]\n"
                    f"content:\n{body}\n"
                    f"--- end browse result ---"
                )
                return ToolResult(text=text, raw={"url": url, "query": query,
                                                  "fetch_method": fetch_method, "summarized": False})

        # Fallback path: no summarizer configured -> raw cleaned page text.
        body = source[:MAX_RETURN_CHARS]
        text = (
            f"--- browse result ---\n"
            f"url: {url}\n"
            f"query: {query}\n"
            f"fetch_method: {fetch_method}\n"
            f"content:\n{body}\n"
            f"--- end browse result ---"
        )
        return ToolResult(text=text, raw={"url": url, "query": query,
                                          "fetch_method": fetch_method, "summarized": False})
