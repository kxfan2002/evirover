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
    text_search_backend: str = "serper",
    summary_base_url: str = "",
    summary_api_key: str = "",
    summary_model: str = "",
) -> Dict[str, Tool]:
    from .crop_verify import (CropTool, VerifyTool, VerifyPartTool, VerifyMaskTool,
                              CompareLRTool)
    from .text_search import SerperTextSearchTool, TextSearchTool
    from .browse import BrowseTool
    from .search_image import TextSearchImageTool, ImageSearchTool
    from .python_tool import PythonTool
    from .summarizer import SummaryClient

    if text_search_backend == "perplexity":
        text_tool = TextSearchTool(pplx_key=pplx_key)
    else:
        text_tool = SerperTextSearchTool(serper_key=serper_key)

    summarizer = None
    if summary_base_url and summary_model and summary_api_key:
        summarizer = SummaryClient(
            base_url=summary_base_url,
            api_key=summary_api_key,
            model=summary_model,
        )

    tools = [
        CropTool(),
        VerifyTool(),
        VerifyPartTool(),
        VerifyMaskTool(segment_fn=segment_fn),
        CompareLRTool(),
        text_tool,
        BrowseTool(summarizer=summarizer),
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
