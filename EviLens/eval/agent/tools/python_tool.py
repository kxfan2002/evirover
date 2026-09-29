"""python -> execute model-written Python for calculation / image analysis.

Per tools.json: pre-injects `original_image` (PIL), `IMG_001..N` (PIL, prior
tool images), and `num_images` (int). Captures print() output; if the code
assigns a PIL image to `_result_image`, it is registered as a new IMG_NNN and
returned. Code with no output (no print, no _result_image) is treated invalid.

Runs in-process (not sandboxed against malicious code) — acceptable here since
the code comes from the model under evaluation, not an untrusted user. A
per-call limit avoids hangs.
"""
from __future__ import annotations

import io
import contextlib
from typing import Any, Dict

from PIL import Image, ImageDraw

from ..image_store import image_to_data_url  # noqa: F401  (available if code needs it)
from . import messages as MSG
from .base import ERR_MODEL, ToolContext, ToolImage, ToolResult


def _preinject(ctx: ToolContext) -> Dict[str, Any]:
    env: Dict[str, Any] = {}
    imgs = ctx.image_store.images
    original = ctx.image_store.find("ORIGINAL")
    if original:
        env["original_image"] = Image.open(original.path).convert("RGB")
    n = 0
    for item in imgs:
        if item.image_id.startswith("IMG_"):
            try:
                env[item.image_id] = Image.open(item.path).convert("RGB")
                n += 1
            except Exception:  # noqa: BLE001
                pass
    env["num_images"] = n
    return env


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

        import numpy as np  # local imports so --no-sam / minimal envs still import the module
        try:
            import cv2  # noqa: F401
        except Exception:  # noqa: BLE001
            cv2 = None
        try:
            import scipy  # noqa: F401
        except Exception:  # noqa: BLE001
            scipy = None
        import math
        import json as _json
        import copy as _copy

        g: Dict[str, Any] = {
            "np": np, "numpy": np, "Image": Image, "ImageDraw": ImageDraw,
            "math": math, "json": _json, "cv2": cv2, "scipy": scipy, "copy": _copy,
        }
        g.update(_preinject(ctx))
        g["_result_image"] = None

        buf = io.StringIO()
        error = ""
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, g)  # noqa: S102 — model-authored code, in-process by design
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        printed = buf.getvalue().strip()
        result_img = g.get("_result_image")
        images = []
        img_line = ""
        if isinstance(result_img, Image.Image):
            stored = ctx.image_store.save_pil(result_img, "python")
            images.append(ToolImage(stored.image_id, stored.rel_path, stored.path))
            img_line = f"\n{stored.image_id}: <image>"

        if error:
            body = f"[Error] {error}"
            if printed:
                body = f"{printed}\n{body}"
        elif printed or img_line:
            body = printed or "(no text output)"
        else:
            body = "[Error] code produced no output (no print, no _result_image)."

        text = f"--- python result ---\n{body}{img_line}\n--- end python result ---"
        failed = bool(error) or body.startswith("[Error]")
        # An execution error must mark the result as failed, so python reports
        # failures the same way every other tool does.
        return ToolResult(text=text, images=images, raw={"has_error": bool(error)},
                          ok=not failed, error_class=ERR_MODEL if failed else None)
