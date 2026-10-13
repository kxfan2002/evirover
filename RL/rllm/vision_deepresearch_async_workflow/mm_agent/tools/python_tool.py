"""python -> execute model-written Python for calculation / image analysis.

Per tools.json: pre-injects `original_image` (PIL), `IMG_001..N` (PIL, prior
tool images), and `num_images` (int). Captures print() output; if the code
assigns a PIL image to `_result_image`, it is registered as a new IMG_NNN and
returned. Code with no output (no print, no _result_image) is treated invalid.

The code is executed in an ISOLATED worker process (see python_sandbox.py), not
in-process: model-authored code can segfault a C extension or hang the GIL, and
this tool runs on the training driver's event-loop thread pool, so an in-process
crash/hang would take down the whole RL run. Isolation turns those into a
tool-error the rollout can see. A hard per-call timeout also bounds hangs.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..image_store import image_to_data_url  # noqa: F401  (available if code needs it)
from . import messages as MSG
from .base import ERR_MODEL, ToolContext, ToolImage, ToolResult
from .python_sandbox import run_code_isolated


def _image_paths(ctx: ToolContext) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """Collect (original_path, [(image_id, path), ...]) for the worker to reopen."""
    original = ctx.image_store.find("ORIGINAL")
    original_path = original.path if original else None
    items = [(it.image_id, it.path) for it in ctx.image_store.images
             if it.image_id.startswith("IMG_")]
    return original_path, items


class PythonTool:
    name = "python"

    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(
                text="--- python result ---\n"
                     + MSG.A4_MISSING_ARG.format(tool="python", arg="code")
                     + "\n--- end python result ---",
                ok=False, error_class=ERR_MODEL)

        original_path, img_items = _image_paths(ctx)
        printed, error, image_bytes = run_code_isolated(code, original_path, img_items)

        images: List[ToolImage] = []
        img_line = ""
        if image_bytes:
            try:
                result_img = Image.open(io.BytesIO(image_bytes))
                stored = ctx.image_store.save_pil(result_img, "python")
                images.append(ToolImage(stored.image_id, stored.rel_path, stored.path))
                img_line = f"\n{stored.image_id}: <image>"
            except Exception as exc:  # noqa: BLE001
                error = error or f"failed to register result image: {exc}"

        printed = (printed or "").strip()
        if error:
            body = f"[Error] {error}" if not error.startswith("[Error]") else error
            if printed:
                body = f"{printed}\n{body}"
        elif printed or img_line:
            body = printed or "(no text output)"
        else:
            body = "[Error] code produced no output (no print, no _result_image)."

        text = f"--- python result ---\n{body}{img_line}\n--- end python result ---"
        failed = bool(error) or body.startswith("[Error]")
        return ToolResult(text=text, images=images, raw={"has_error": bool(error)},
                          ok=not failed, error_class=ERR_MODEL if failed else None)
