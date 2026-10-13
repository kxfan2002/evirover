"""Adapters that expose the SFT/Eval multimodal tools through the
Vision-DeepResearch DeepResearchTool interface.

The copied Eval tools use call(args, ctx). Vision-DeepResearch uses
await tool.call(**tool_args). This module bridges the two without changing the
Eval source package.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from contextvars import ContextVar
from typing import Any, Dict, Optional

from PIL import Image

from vision_deepresearch_async_workflow.tools.shared import DeepResearchTool

from .image_store import ImageStore
from .tools import build_tools, SerperBudget
from .tools.base import ToolContext, ToolResult


def _safe_id(value: Any) -> str:
    text = str(value or "run")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:120] or "run"


def _default_output_root() -> str:
    return (
        os.getenv("MM_AGENT_IMAGE_CACHE")
        or os.getenv("IMAGE_CROP_CACHE")
        or "/tmp/vision_deepresearch_mm_tools"
    )


def _load_tool_specs() -> Dict[str, Dict[str, Any]]:
    spec_path = Path(__file__).with_name("tools.json")
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        return {item["name"]: item for item in data.get("tools", []) if isinstance(item, dict) and item.get("name")}
    except Exception:  # noqa: BLE001
        return {}


def _materialize_image_ref(image_ref: Any, output_dir: Path, sample_run_id: str) -> Optional[str]:
    """Return a local path for a path/data-url/PIL/dict image reference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image_ref, dict):
        if image_ref.get("path"):
            image_ref = image_ref["path"]
        elif image_ref.get("image"):
            image_ref = image_ref["image"]
        elif image_ref.get("url"):
            image_ref = image_ref["url"]
        elif image_ref.get("data"):
            image_ref = image_ref["data"]
        elif image_ref.get("bytes") is not None:
            try:
                img = Image.open(BytesIO(image_ref["bytes"])).convert("RGB")
                out = output_dir / f"{sample_run_id}-original-input.jpg"
                img.save(out, "JPEG", quality=95)
                return str(out)
            except Exception:  # noqa: BLE001
                return None

    if isinstance(image_ref, Image.Image):
        out = output_dir / f"{sample_run_id}-original-input.jpg"
        img = image_ref.convert("RGB") if image_ref.mode != "RGB" else image_ref
        img.save(out, "JPEG", quality=95)
        return str(out)

    if isinstance(image_ref, str):
        if os.path.exists(image_ref):
            return image_ref
        if image_ref.startswith("data:image/"):
            try:
                _, encoded = image_ref.split(",", 1)
                img = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
                out = output_dir / f"{sample_run_id}-original-input.jpg"
                img.save(out, "JPEG", quality=95)
                return str(out)
            except Exception:  # noqa: BLE001
                return None

    return None


#
#
_CURRENT_CTX: "ContextVar[Optional[ToolContext]]" = ContextVar("mm_tool_ctx", default=None)
_CURRENT_BUDGET: "ContextVar[Optional[SerperBudget]]" = ContextVar("mm_serper_budget", default=None)


class MMToolRuntime:

    def __init__(self, output_root: Optional[str] = None):
        self.output_root = output_root or _default_output_root()
        self.serper_budget_limit = int(os.getenv("SERPER_BUDGET_LIMIT", "25"))

    def start_run(
        self,
        *,
        image_path: Any = None,
        images: Optional[list[Any]] = None,
        task: Any = None,
        uid: Any = None,
        output_dir: Optional[str] = None,
    ) -> None:
        sample_run_id = _safe_id(uid or (task or {}).get("id") if isinstance(task, dict) else uid)
        run_key = f"{sample_run_id}:{id(images)}:{str(image_path)[:200]}"
        cur = _CURRENT_CTX.get()
        if cur is not None and getattr(cur, "_run_key", None) == run_key:
            return

        out_root = Path(output_dir or self.output_root) / sample_run_id
        store = ImageStore(str(out_root), sample_run_id=sample_run_id)

        original_ref = image_path
        if original_ref is None and images:
            original_ref = images[0]
        original_path = _materialize_image_ref(original_ref, out_root / "inputs", sample_run_id)
        if original_path:
            store.add_original(original_path)

        ctx = ToolContext(sample=task or {}, image_store=store, output_dir=str(out_root))
        ctx._run_key = run_key
        _CURRENT_CTX.set(ctx)
        _CURRENT_BUDGET.set(SerperBudget(self.serper_budget_limit))

    def require_ctx(self) -> ToolContext:
        ctx = _CURRENT_CTX.get()
        if ctx is None:
            self.start_run(uid="lazy")
            ctx = _CURRENT_CTX.get()
        assert ctx is not None
        return ctx

    @staticmethod
    def current_budget() -> Optional[SerperBudget]:
        return _CURRENT_BUDGET.get()


class MMToolAdapter(DeepResearchTool):
    """Expose an Eval-style tool as an async Vision-DeepResearch tool."""

    def __init__(self, tool: Any, runtime: MMToolRuntime, spec: Optional[Dict[str, Any]] = None):
        self.tool = tool
        self.runtime = runtime
        name = getattr(tool, "name", (spec or {}).get("name", "unknown_tool"))
        parameters = _parameters_from_spec(spec or {})
        super().__init__(
            name=name,
            description=(spec or {}).get("description", name),
            parameters=parameters,
        )

    async def call(self, **kwargs) -> Dict[str, Any]:
        ctx = self.runtime.require_ctx()
        # tool.call does synchronous CPU work (PIL crop/decode, python exec, image
        # I/O). Offload to a thread so it doesn't block the shared event loop that
        # dispatches generation requests to the 8 vLLM engines.
        try:
            result = await asyncio.to_thread(self.tool.call, kwargs, ctx)
        except Exception as exc:  # noqa: BLE001
            from .tools import messages as MSG
            from .tools.base import ERR_MODEL, ToolResult
            result = ToolResult(
                text=MSG.E2_TOOL_EXCEPTION.format(tool=self.name, exc_type=type(exc).__name__),
                ok=False, error_class=ERR_MODEL,
            )
        return _result_to_message_payload(result)


def _parameters_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    required = []
    for name, meta in (spec.get("parameters") or {}).items():
        typ = meta.get("type", "string")
        if typ.startswith("array"):
            schema: Dict[str, Any] = {"type": "array"}
        elif typ == "integer":
            schema = {"type": "integer"}
        elif typ == "number":
            schema = {"type": "number"}
        else:
            schema = {"type": "string"}
        if meta.get("description"):
            schema["description"] = meta["description"]
        props[name] = schema
        if meta.get("required"):
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def _result_to_message_payload(result: Any) -> Dict[str, Any]:
    if isinstance(result, ToolResult):
        return {
            "text": result.text,
            "images": [{"image": im.path} for im in result.images],
            "raw": result.raw,
            "ok": result.ok,
            "error_class": result.error_class,
        }
    text = getattr(result, "text", None)
    images = getattr(result, "images", None)
    if text is not None:
        return {
            "text": str(text),
            "images": [
                {"image": getattr(im, "path", im)}
                for im in (images or [])
                if im is not None
            ],
        }
    return {"text": str(result), "images": []}


def build_mm_deepresearch_tools(
    *,
    pplx_key: Optional[str] = None,
    serper_key: Optional[str] = None,
    serper_budget_limit: Optional[int] = None,
    output_root: Optional[str] = None,
    segment_fn: Any = None,
    enable_python: bool = True,
) -> Dict[str, MMToolAdapter]:
    specs = _load_tool_specs()
    runtime = MMToolRuntime(output_root=output_root)
    pplx_key = pplx_key if pplx_key is not None else (os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY") or "")
    serper_key = serper_key if serper_key is not None else (os.getenv("SERPER_API_KEY") or "")
    if serper_budget_limit is None:
        serper_budget_limit = int(os.getenv("SERPER_BUDGET_LIMIT", "25"))
    budget = SerperBudget(serper_budget_limit)

    text_search_backend = os.getenv("VDR_TEXT_SEARCH_BACKEND", "serper").strip().lower()

    if segment_fn is None and os.getenv("VDR_SAM_PREINIT", "0").lower() in {"1", "true", "yes"}:
        from rllm.rewards.mm_deepresearch_reward import segment_with_sam3

        segment_fn = segment_with_sam3

    eval_tools = build_tools(
        pplx_key=pplx_key,
        serper_key=serper_key,
        serper_budget=budget,
        segment_fn=segment_fn,
        enable_python=enable_python,
        text_search_backend=text_search_backend,
        summary_base_url=os.getenv("SUMMARY_BASE_URL", "https://api.novita.ai/v3/openai"),
        summary_api_key=(os.getenv("SUMMARY_API_KEY") or os.getenv("NOVITA_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN", "")),
        summary_model=os.getenv("SUMMARY_MODEL", "qwen/qwen3.5-27b"),
    )
    return {
        name: MMToolAdapter(tool, runtime=runtime, spec=specs.get(name))
        for name, tool in eval_tools.items()
    }
