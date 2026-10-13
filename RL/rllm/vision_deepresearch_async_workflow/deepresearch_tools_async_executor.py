from vision_deepresearch_async_workflow.mm_agent.adapter import (
    build_mm_deepresearch_tools,
)
from vision_deepresearch_async_workflow.tools.shared import DeepResearchTool

# Tool registry aligned with mm_agent/tools.json.
# The concrete implementations are copied from final/Eval/eval/agent/tools and
# adapted to Vision-DeepResearch's async DeepResearchTool interface.
DEEPRESEARCH_TOOLS = build_mm_deepresearch_tools()


def get_tool(name: str) -> DeepResearchTool:
    """Get a tool by name."""
    return DEEPRESEARCH_TOOLS.get(name)


def get_all_tools() -> dict[str, DeepResearchTool]:
    """Get all available tools."""
    return DEEPRESEARCH_TOOLS.copy()


# Local-only tools: no network / external API (no search, browse, image_search).
# Used for the no-search demo so rollout never hits Serper/Perplexity/OSS and the
# reward stays purely local (bbox IoU / spot_diff F1). crop/verify/verify_part
# return follow-up IMAGE turns, so episodes are genuinely multi-turn+multimodal
# and exercise the image-aware training forward.
LOCAL_TOOL_NAMES = ("crop", "verify", "verify_part", "verify_mask", "compare_lr", "python")


def get_local_tools() -> dict[str, DeepResearchTool]:
    """Get only the local (no-network) tools."""
    return {n: t for n, t in DEEPRESEARCH_TOOLS.items() if n in LOCAL_TOOL_NAMES and t is not None}


__all__ = [
    "DeepResearchTool",
    "build_mm_deepresearch_tools",
    "DEEPRESEARCH_TOOLS",
    "get_tool",
    "get_all_tools",
    "get_local_tools",
    "LOCAL_TOOL_NAMES",
]
