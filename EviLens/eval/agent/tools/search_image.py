"""Serper-backed image tools (budget-gated).

- text_search_image -> Serper /images: keyword image search, returns image
  references (downloads each result and registers it as an IMG_NNN with an
  <image> the model can see).
- image_search       -> Serper /lens: reverse image search on ORIGINAL (or a
  crop of it). Uploads the query image to OSS to get a public URL, then returns
  text evidence (page titles/urls/snippets) — no result images, matching the
  SFT tool's text-only evidence output.

Both share a scarce Serper budget (SerperBudget). When the key is missing or the
budget is exhausted, they return a graceful "unavailable" tool response so the
agent loop continues instead of erroring.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from ..image_store import crop_image, norm_xyxy_to_pixels
from . import messages as MSG
from .base import (ERR_BUDGET, ERR_EMPTY, ERR_INFRA, ERR_MODEL, SerperBudget,
                   ToolContext, ToolImage, ToolResult, http_post_json)
from . import oss_upload

SERPER_IMAGES_URL = "https://google.serper.dev/images"
SERPER_LENS_URL = "https://google.serper.dev/lens"


def _download(url: str, save_dir: str, prefix: str) -> Optional[str]:
    import hashlib
    from pathlib import Path
    from urllib.parse import urlparse

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        ext = ".jpg"
    out = Path(save_dir) / f"{prefix}{hashlib.md5(url.encode()).hexdigest()}{ext}"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0",
                                          "Accept": "image/*,*/*;q=0.8"}, timeout=20, stream=True)
        resp.raise_for_status()
        data = b""
        for chunk in resp.iter_content(8192):
            data += chunk
            if len(data) > 12 * 1024 * 1024:
                return None
        if len(data) < 64:
            return None
        out.write_bytes(data)
        # verify it opens
        with Image.open(out) as im:
            im.verify()
        return str(out)
    except Exception:  # noqa: BLE001
        return None


class TextSearchImageTool:
    name = "text_search_image"

    def __init__(self, serper_key: str = "", budget: Optional[SerperBudget] = None):
        self.serper_key = serper_key or ""
        self.budget = budget

    def _unavailable(self, query: str, reason: str) -> ToolResult:
        return ToolResult(text=(
            f"--- image search result for [{query}] ---\n"
            f"text_search_image unavailable: {reason}\n"
            "--- end image search result ---"
        ))

    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query") or "").strip()
        top_k = max(1, min(10, int(args.get("top_k", 5))))

        def wrap(msg, cls):
            return ToolResult(
                text=f"--- image search result for [{query}] ---\n{msg}\n"
                     f"--- end image search result ---",
                ok=False, error_class=cls)

        if not query:
            return wrap(MSG.A4_MISSING_ARG.format(tool="text_search_image", arg="query"), ERR_MODEL)
        if not self.serper_key:
            return wrap(MSG.B4_NO_KEY.format(tool="text_search_image"), ERR_INFRA)
        if self.budget is not None and not self.budget.try_spend():
            # Budget exhaustion is a design constraint, not a fault; classed apart.
            return wrap(MSG.D1_SERPER_BUDGET.format(tool="text_search_image"), ERR_BUDGET)

        data, err, code = http_post_json(
            SERPER_IMAGES_URL,
            {"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
            {"q": query, "num": top_k})
        if err == "rate_limited":
            return wrap(MSG.B1_RATE_LIMITED.format(tool="text_search_image", tries=10), ERR_INFRA)
        if err == ERR_INFRA:
            return wrap(MSG.B2_SERVICE_ERROR.format(tool="text_search_image", code=code) if code
                        else MSG.B3_NETWORK_ERROR.format(tool="text_search_image"), ERR_INFRA)
        items = [x for x in (data.get("images") or []) if isinstance(x, dict)]

        lines = [f"--- image search result for [{query}] ---"]
        tool_images: List[ToolImage] = []
        raw = []
        count = 0
        for item in items:
            if count >= top_k:
                break
            img_url = str(item.get("imageUrl") or item.get("thumbnailUrl") or item.get("image_url") or "")
            if not img_url:
                continue
            local = _download(img_url, str(ctx.image_store.image_dir),
                              prefix=f"{ctx.image_store.sample_run_id}-tsi-")
            if not local:
                continue
            stored = ctx.image_store.add_external_file(local, "text_search_image")
            count += 1
            title = item.get("title") or "image"
            page_url = item.get("link") or item.get("source") or ""
            lines += [f"{stored.image_id}: <image>", f"  title: {title}", f"  url: {img_url}"]
            if page_url:
                lines.append(f"  page_url: {page_url}")
            tool_images.append(ToolImage(stored.image_id, stored.rel_path, stored.path))
            raw.append(item)
        if count == 0:
            return wrap(MSG.C2_NO_IMAGE_RESULTS.format(tool="text_search_image"), ERR_EMPTY)
        lines.append("--- end image search result ---")
        return ToolResult(text="\n".join(lines), images=tool_images, raw=raw)


class ImageSearchTool:
    name = "image_search"

    def __init__(self, serper_key: str = "", budget: Optional[SerperBudget] = None):
        self.serper_key = serper_key or ""
        self.budget = budget

    def _unavailable(self, reason: str) -> ToolResult:
        return ToolResult(text=(
            "--- image-to-image search results ---\n"
            f"image_search unavailable: {reason}\n"
            "--- end image-to-image search results ---"
        ))

    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        image_id = str(args.get("image_id") or "ORIGINAL")
        top_k = max(1, min(10, int(args.get("top_k", 3))))
        boxes = args.get("boxes")

        def wrap(msg, cls):
            return ToolResult(
                text=f"--- image-to-image search results ---\n{msg}\n"
                     f"--- end image-to-image search results ---",
                ok=False, error_class=cls)

        if not self.serper_key:
            return wrap(MSG.B4_NO_KEY.format(tool="image_search"), ERR_INFRA)

        item = ctx.image_store.find(image_id)
        if not item:
            # List every available id, untruncated (see messages.A2).
            avail = MSG.fmt_available([i.image_id for i in ctx.image_store.images])
            return wrap(MSG.A2_UNKNOWN_IMAGE_ID.format(got=image_id, available=avail), ERR_MODEL)

        # Optionally crop before searching.
        tmp_crop = None
        query_path = item.path
        crop_note = ""
        if isinstance(boxes, (list, tuple)) and len(boxes) == 4:
            try:
                cropped = crop_image(item.path, [float(x) for x in boxes], padding=0.0, scale=1)
                fd, tmp_crop = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                cropped.save(tmp_crop, "JPEG", quality=92)
                query_path = tmp_crop
                crop_note = f"cropped to boxes={list(boxes)}"
            except Exception:  # noqa: BLE001  crop failure is not fatal; search the
                                              # whole image instead
                crop_note = "crop failed; searched full image"

        def cleanup():
            if tmp_crop:
                try:
                    os.unlink(tmp_crop)
                except OSError:
                    pass

        if self.budget is not None and not self.budget.try_spend():
            cleanup()
            return wrap(MSG.D1_SERPER_BUDGET.format(tool="image_search"), ERR_BUDGET)

        # ---- upload ----
        # Upload and search are deliberately in separate try blocks. Sharing one
        # would let an upload failure fall through to "No results.", making a broken
        # image host indistinguishable from a genuine miss.
        oss_key = None
        try:
            public_url, oss_key = oss_upload.upload(query_path)
        except Exception:  # noqa: BLE001
            if self.budget is not None:
                self.budget.refund()      # never reached Serper; give the quota back
            oss_upload.delete(oss_key)
            cleanup()
            return wrap(MSG.B6_UPLOAD_FAILED, ERR_INFRA)

        try:
            data, err, code = http_post_json(
                SERPER_LENS_URL,
                {"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
                {"url": public_url}, timeout=60)
        finally:
            oss_upload.delete(oss_key)
            cleanup()

        if err == "rate_limited":
            return wrap(MSG.B1_RATE_LIMITED.format(tool="image_search", tries=10), ERR_INFRA)
        if err == ERR_INFRA:
            return wrap(MSG.B2_SERVICE_ERROR.format(tool="image_search", code=code) if code
                        else MSG.B3_NETWORK_ERROR.format(tool="image_search"), ERR_INFRA)

        organic = [x for x in (data.get("organic") or data.get("results") or []) if isinstance(x, dict)]
        if not organic:
            return wrap(MSG.C2_NO_IMAGE_RESULTS.format(tool="image_search"), ERR_EMPTY)

        lines = ["--- image-to-image search results ---"]
        if crop_note:
            lines.append(crop_note)
        lines.append(f"num_results: {len(organic)}")
        for i, r in enumerate(organic[:top_k], 1):
            title = r.get("title") or "result"
            link = r.get("link") or r.get("url") or ""
            snippet = " ".join(str(r.get("snippet") or "").split())
            lines.append(f"[{i}] {title} — {link}")
            if snippet:
                lines.append(f"    {snippet}")
        lines.append("--- end image-to-image search results ---")
        return ToolResult(text="\n".join(lines), raw={"organic": organic})
