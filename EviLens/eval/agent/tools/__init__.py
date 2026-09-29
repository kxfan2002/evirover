"""Agent tools, keyed by name (matching tools.json).

Local tools (no network): crop, verify, verify_part, verify_mask, compare_lr, python.
Network tools: text_search (Perplexity), browse (self-fetch), text_search_image
and image_search (Serper, budget-gated).

build_tools() wires them with the shared runtime config (search keys, Serper
budget, SAM segment fn). Unavailable network tools are still registered but
return a graceful "unavailable" tool response so the agent can proceed.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from .base import Tool, ToolContext, ToolResult, ToolImage, format_tool_response, SerperBudget


def build_tools(
    pplx_key: str = "",
    serper_key: str = "",
    serper_budget: Optional[SerperBudget] = None,
    segment_fn: Optional[Callable] = None,
    enable_python: bool = True,
    summary_base_url: str = "",
    summary_api_key: str = "",
    summary_model: str = "",
    browse_tokenizer_path: str = "",
    browse_max_tokens: int = 24000,
    text_search_backend: str = "perplexity",
) -> Dict[str, Tool]:
    from .crop_verify import (CropTool, VerifyTool, VerifyPartTool, VerifyMaskTool,
                              CompareLRTool)
    from .text_search import TextSearchTool, SerperTextSearchTool
    from .browse import BrowseTool
    from .search_image import TextSearchImageTool, ImageSearchTool
    from .python_tool import PythonTool

    # browse summarizer (SFT parity): configured OpenAI-compatible model, else raw text.
    summarizer = None
    if summary_base_url and summary_model:
        from ..summarizer import SummaryClient
        summarizer = SummaryClient(
            base_url=summary_base_url, api_key=summary_api_key, model=summary_model,
        )

    # Optional tokenizer so browse caps fetched page text by TOKENS (24k) rather
    # than chars (language-independent). Loaded once; shared read-only across the
    # thread pool (HF fast tokenizers are safe for concurrent encode/decode).
    browse_tokenizer = None
    if browse_tokenizer_path:
        try:
            from transformers import AutoTokenizer
            browse_tokenizer = AutoTokenizer.from_pretrained(
                browse_tokenizer_path, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[agent] browse tokenizer load failed ({exc}); using char fallback")

    # text_search backend: Perplexity (default) or Serper /search. Serper backend
    # is uncapped (does not consume the image-tool budget) so an all-Serper eval
    # compares providers on result quality, not availability.
    if text_search_backend == "serper":
        text_tool = SerperTextSearchTool(serper_key=serper_key)
    else:
        text_tool = TextSearchTool(pplx_key=pplx_key)

    tools = [
        CropTool(),
        VerifyTool(),
        VerifyPartTool(),
        VerifyMaskTool(segment_fn=segment_fn),
        CompareLRTool(),
        text_tool,
        BrowseTool(summarizer=summarizer, tokenizer=browse_tokenizer,
                   max_content_tokens=browse_max_tokens),
        TextSearchImageTool(serper_key=serper_key, budget=serper_budget),
        ImageSearchTool(serper_key=serper_key, budget=serper_budget),
    ]
    if enable_python:
        tools.append(PythonTool())
    return {t.name: t for t in tools}


__all__ = [
    "Tool", "ToolContext", "ToolResult", "ToolImage",
    "format_tool_response", "SerperBudget", "build_tools",
]
