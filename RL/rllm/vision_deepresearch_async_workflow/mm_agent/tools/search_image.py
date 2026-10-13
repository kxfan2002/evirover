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


def _current_budget(fallback):
    try:
        from ..adapter import MMToolRuntime
        b = MMToolRuntime.current_budget()
        if b is not None:
            return b
    except Exception:
        pass
    return fallback


def _download(url: str, save_dir: str, prefix: str, timeout: int = 8) -> Optional[str]:
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
                                          "Accept": "image/*,*/*;q=0.8"}, timeout=timeout, stream=True)
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

    def call(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query") or "").strip()
        top_k = max(1, min(10, int(args.get("top_k", 3))))

        def wrap(msg, cls):
            return ToolResult(
                text=f"--- image search result for [{query}] ---\n{msg}\n"
                     f"--- end image search result ---",
                ok=False, error_class=cls)

        if not query:
            return wrap(MSG.A4_MISSING_ARG.format(tool="text_search_image", arg="query"), ERR_MODEL)
        if not self.serper_key:
            return wrap(MSG.B4_NO_KEY.format(tool="text_search_image"), ERR_INFRA)
        _budget = _current_budget(self.budget)
        if _budget is not None and not _budget.try_spend():
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
        dl_attempted = 0
        dl_failed = 0
        for item in items:
            if count >= top_k:
                break
            primary = str(item.get("imageUrl") or item.get("image_url") or "")
            thumb = str(item.get("thumbnailUrl") or "")
            candidates = [u for u in (primary, thumb) if u]
            candidates = list(dict.fromkeys(candidates))
            if not candidates:
                continue
            dl_attempted += 1
            local = None
            used_url = ""
            for cand in candidates:
                local = _download(cand, str(ctx.image_store.image_dir),
                                  prefix=f"{ctx.image_store.sample_run_id}-tsi-")
                if local:
                    used_url = cand
                    break
            if not local:
                dl_failed += 1
                continue
            stored = ctx.image_store.add_external_file(local, "text_search_image")
            count += 1
            title = item.get("title") or "image"
            page_url = item.get("link") or item.get("source") or ""
            lines += [f"{stored.image_id}: <image>", f"  title: {title}", f"  url: {used_url}"]
            if page_url:
                lines.append(f"  page_url: {page_url}")
            tool_images.append(ToolImage(stored.image_id, stored.rel_path, stored.path))
            raw.append(item)
        if count == 0:
            if dl_attempted > 0 and dl_failed == dl_attempted:
                return wrap(MSG.B7_IMAGE_DOWNLOAD_FAILED.format(
                    tool="text_search_image", n=dl_failed), ERR_INFRA)
            tmpl = (MSG.C2_NO_IMAGE_RESULTS_QUOTED if '"' in query
                    else MSG.C2_NO_IMAGE_RESULTS)
            return wrap(tmpl.format(tool="text_search_image"), ERR_EMPTY)
        lines.append("--- end image search result ---")
        return ToolResult(text="\n".join(lines), images=tool_images, raw=raw)


class ImageSearchTool:
    name = "image_search"

    def __init__(self, serper_key: str = "", budget: Optional[SerperBudget] = None):
        self.serper_key = serper_key or ""
        self.budget = budget

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
            except Exception:
                crop_note = "crop failed; searched full image"

        def cleanup():
            if tmp_crop:
                try:
                    os.unlink(tmp_crop)
                except OSError:
                    pass

        _budget = _current_budget(self.budget)
        if _budget is not None and not _budget.try_spend():
            cleanup()
            return wrap(MSG.D1_SERPER_BUDGET.format(tool="image_search"), ERR_BUDGET)

        oss_key = None
        try:
            public_url, oss_key = oss_upload.upload(query_path)
        except Exception as _exc:  # noqa: BLE001
            import sys as _sys
            import traceback as _tb
            print(f"[image_search] OSS upload FAILED path={query_path!r} "
                  f"exc={_exc!r}", file=_sys.stderr, flush=True)
            _tb.print_exc(file=_sys.stderr)
            if _budget is not None:
                _budget.refund()
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
