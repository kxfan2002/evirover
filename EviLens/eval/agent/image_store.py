"""Per-sample image store for the agent loop.

Adapted from raw_data/peragent/label_sft/implement/image_store.py, trimmed to
what the eval needs. Tracks ORIGINAL plus tool-produced images (crop/verify/
search results) under stable IDs (IMG_001, IMG_002, ...) matching the
tools.json image_id convention, and turns any of them into base64 data URLs for
the OpenAI-compatible request.
"""
from __future__ import annotations

import base64
import io
import mimetypes
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageDraw


@dataclass
class StoredImage:
    image_id: str        # ORIGINAL | IMG_001 | IMG_002 ...
    path: str            # absolute local path
    rel_path: str        # path relative to the run output dir (for logging)
    role: str            # original | crop | verify | text_search_image | ...
    source: str = ""     # source path / url, if any


class ImageStore:
    """Holds every image visible to the agent for one sample run.

    Unlike the reference, all tool-produced images share a single IMG_NNN
    counter (not per-prefix), so numbering follows the order images appear in the
    record, matching the rule in tools.json.
    """

    def __init__(self, output_dir: str, sample_run_id: str):
        self.output_dir = Path(output_dir)
        self.sample_run_id = sample_run_id
        self.image_dir = self.output_dir / "cached_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.images: List[StoredImage] = []
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"IMG_{self._counter:03d}"

    def _rel(self, path: Path) -> str:
        return "./" + os.path.relpath(path, self.output_dir)

    @staticmethod
    def _safe_ext(path: str) -> str:
        ext = Path(path).suffix.lower()
        return ext if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"} else ".jpg"

    # -- registration -------------------------------------------------------

    def add_original(self, source_path: str) -> StoredImage:
        ext = self._safe_ext(source_path)
        dest = self.image_dir / f"{self.sample_run_id}-original{ext}"
        if not dest.exists():
            shutil.copy2(source_path, dest)
        item = StoredImage("ORIGINAL", str(dest), self._rel(dest), "original", source_path)
        self.images.append(item)
        return item

    # Whole-image review products answer "is this the right place", not "what is
    # here", so they need no native resolution -- and they dominate visual token
    # cost. crop and verify_part are the opposite and must never be downscaled.
    FULL_VIEW_ROLES = ("verify", "verify_mask")

    @staticmethod
    def _full_view_max_side() -> int:
        v = os.environ.get("VDR_EVAL_VERIFY_MAX_SIDE", "1536")
        n = int(v)
        return n if n > 0 else 10 ** 9

    def save_pil(self, image: Image.Image, role: str) -> StoredImage:
        image_id = self._next_id()
        dest = self.image_dir / f"{self.sample_run_id}-{image_id.lower()}.jpg"
        if image.mode != "RGB":
            image = image.convert("RGB")
        if role in self.FULL_VIEW_ROLES:
            cap = self._full_view_max_side()
            w, h = image.size
            if max(w, h) > cap:
                s = cap / float(max(w, h))
                # The box is drawn on the original at a line width proportional to
                # the long edge, so it survives the downscale.
                image = image.resize((max(1, int(w * s)), max(1, int(h * s))),
                                     Image.Resampling.LANCZOS)
        image.save(dest, "JPEG", quality=95)
        item = StoredImage(image_id, str(dest), self._rel(dest), role)
        self.images.append(item)
        return item

    def add_external_file(self, source_path: str, role: str) -> StoredImage:
        """Register an already-downloaded image file (e.g. a search result)."""
        ext = self._safe_ext(source_path)
        image_id = self._next_id()
        dest = self.image_dir / f"{self.sample_run_id}-{image_id.lower()}{ext}"
        if os.path.abspath(source_path) != os.path.abspath(dest) and not dest.exists():
            shutil.copy2(source_path, dest)
        item = StoredImage(image_id, str(dest), self._rel(dest), role, source_path)
        self.images.append(item)
        return item

    def find(self, image_id: str) -> Optional[StoredImage]:
        for item in self.images:
            if item.image_id == image_id:
                return item
        return None


def _default_max_side() -> int:
    """Delivery long-side cap. 0 / negative = no cap (send at native resolution).

    1536 is NOT a protocol constant anyone chose for the benchmark — it is an
    implementation default that silently became one. Measured on the 155
    localization images (2026-08-31): 87.7% get downscaled, median factor 0.361,
    worst 0.162, and 46.6% of GT boxes end up occupying LESS THAN ONE 32x32
    visual token. Meanwhile every commercial baseline was run through the qa
    path, which sends the ORIGINAL file bytes uncapped — so the comparison
    tables were 1536-vs-native without that ever being stated.

    Raising this is only half the change: the processor's own area cap
    (preprocessor_config.json, ours=2.16M vs upstream 16.78M) clamps whatever
    gets through. See VDR_EVAL_MM_MAX_PIXELS in serve.py.
    """
    v = os.environ.get("VDR_EVAL_MAX_SIDE")
    if v is None:
        return 1536
    n = int(v)
    return n if n > 0 else 10 ** 9


@lru_cache(maxsize=2048)
def image_to_data_url(path: str, max_side: int = 0, quality: int = 90) -> str:
    """Load an image, downscale the long side to <=max_side, return a JPEG data URL.

    max_side=0 -> read VDR_EVAL_MAX_SIDE (default 1536). Explicit callers still win.

    MEMOIZED: the agent runner rebuilds the full message list every turn, so it
    re-requests the data URL for every image already in the conversation on every
    turn — O(turns x images) decodes per episode. Image files are immutable once
    written (original copied once, crops saved once), so the data URL is a pure
    function of (path, max_side, quality); caching it collapses the per-episode
    PIL decode/resize/encode work from O(N^2) to O(N) with byte-identical output.
    Bounded LRU (2048) so memory stays bounded across a long sampling run.
    """
    if max_side <= 0:
        max_side = _default_max_side()
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as probe:
        w, h = probe.size
    if max(w, h) <= max_side:
        # When no resize is needed, send the original bytes. Re-encoding preserves
        # resolution but not fidelity, and localization turns on exactly those
        # pixels. This also keeps the agent path byte-identical to the QA path.
        mime, _ = mimetypes.guess_type(path)
        with open(path, "rb") as f:
            return f"data:{mime or 'image/jpeg'};base64," + base64.b64encode(f.read()).decode("utf-8")
    with Image.open(path) as img:
        if img.mode in ("RGBA", "P", "L", "LA"):
            img = img.convert("RGB")
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# -- geometry helpers (0-1000 normalized xyxy -> pixels) --------------------

def _clamp_norm_xyxy(bbox: Iterable[float]) -> List[float]:
    vals = [max(0.0, min(1000.0, float(v))) for v in bbox]
    if len(vals) != 4:
        raise ValueError("bbox must have 4 values")
    return vals


def norm_xyxy_to_pixels(bbox_xyxy: Iterable[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = _clamp_norm_xyxy(bbox_xyxy)
    return [
        max(0, min(width - 1, int(round(x1 * width / 1000.0)))),
        max(0, min(height - 1, int(round(y1 * height / 1000.0)))),
        max(0, min(width - 1, int(round(x2 * width / 1000.0)))),
        max(0, min(height - 1, int(round(y2 * height / 1000.0)))),
    ]


def crop_image(image_path: str, bbox_norm: Iterable[float], padding: float = 0.1,
               scale: int = 2) -> Image.Image:
    """Crop a normalized-0-1000 xyxy region, pad, and ADAPTIVELY upscale for detail.

    Default is adaptive: apply the `scale`x LANCZOS upscale only when the crop's
    native long side < VDR_EVAL_CROP_SCALE_THRESH (default 768), else 1x. The 768
    threshold matches the image_to_data_url 1536 long-side delivery cap (2x768=1536),
    beyond which the 2x is downscaled back away = wasted tokens. A/B on localization:
    adaptive matches blanket-2x accuracy (<1sigma) while cutting ~17% of crop tokens;
    the 2x patch-grid boost only ever mattered for the small final commit crop.

    VDR_EVAL_CROP_SCALE=N forces a fixed Nx (disables adaptive) for ablation
    (=2 restores the old blanket-2x behavior, =1 the scale-1 floor).
    """
    _env_scale = os.environ.get("VDR_EVAL_CROP_SCALE")
    forced_scale = int(_env_scale) if _env_scale is not None else None
    _thresh = int(os.environ.get("VDR_EVAL_CROP_SCALE_THRESH", "768"))
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = norm_xyxy_to_pixels(bbox_norm, w, h)
        pad_x = int((x2 - x1) * max(0.0, padding))
        pad_y = int((y2 - y1) * max(0.0, padding))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid crop bbox")
        cropped = img.crop((x1, y1, x2, y2))
        _sz_log = os.environ.get("VDR_CROP_SIZE_LOG")
        if _sz_log:
            try:
                with open(_sz_log, "a") as _f:
                    _f.write(f"{cropped.width},{cropped.height}\n")
            except OSError:
                pass
        eff_scale = scale
        if forced_scale is not None:
            eff_scale = forced_scale
        elif scale > 1 and max(cropped.width, cropped.height) >= _thresh:
            eff_scale = 1              # coarse crop: 2x would be clipped by 1536 cap -> native
        if eff_scale > 1:
            cropped = cropped.resize((cropped.width * eff_scale, cropped.height * eff_scale),
                                     Image.Resampling.LANCZOS)
        return cropped


def compare_lr_image(image_path: str, bbox_norm: Iterable[float],
                     padding: float = 0.1) -> Image.Image:
    """Side-by-side crop of a LEFT-panel box and its corresponding RIGHT-panel region.

    For spot-the-difference images (two panels side by side), a difference can only
    be found by looking at the SAME location in both panels. A plain crop cannot do
    this: the left region and its right counterpart are 500 (normalized) apart, so
    a single box wide enough to contain both spans more than half the image and is
    effectively unzoomed. This builds that pair directly.

    Both halves are cut at IDENTICAL pixel dimensions — the box is clamped to the
    left panel FIRST and the right half then reuses that exact width/height, so the
    two sides stay pixel-comparable. (crop_image cannot be reused: its padding and
    per-side clamping would silently give the two halves different sizes.)

    Upscaling matches crop_image's adaptive rule but is judged on the STITCHED long
    side, since the composite is what gets delivered under the 1536 cap.
    """
    _env_scale = os.environ.get("VDR_EVAL_CROP_SCALE")
    forced_scale = int(_env_scale) if _env_scale is not None else None
    _thresh = int(os.environ.get("VDR_EVAL_CROP_SCALE_THRESH", "768"))
    sep = 6
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        half = w // 2
        x1, y1, x2, y2 = norm_xyxy_to_pixels(bbox_norm, w, h)
        pad_x = int((x2 - x1) * max(0.0, padding))
        pad_y = int((y2 - y1) * max(0.0, padding))
        # Clamp to the LEFT panel only; the right half mirrors these exact bounds.
        x1 = max(0, x1 - pad_x)
        x2 = min(half, x2 + pad_x)
        y1 = max(0, y1 - pad_y)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid crop bbox")
        left = img.crop((x1, y1, x2, y2))
        right = img.crop((x1 + half, y1, x2 + half, y2))
        if left.size != right.size:
            raise ValueError(
                f"invalid crop bbox: left {left.size} != right {right.size}")

        bw, bh = left.size
        out = Image.new("RGB", (bw * 2 + sep, bh), (255, 0, 255))
        out.paste(left, (0, 0))
        out.paste(right, (bw + sep, 0))

        eff_scale = 2
        if forced_scale is not None:
            eff_scale = forced_scale
        elif max(out.width, out.height) >= _thresh:
            eff_scale = 1
        if eff_scale > 1:
            out = out.resize((out.width * eff_scale, out.height * eff_scale),
                             Image.Resampling.LANCZOS)
        return out


def verify_image(image_path: str, bbox_norm: Iterable[float]) -> Image.Image:
    """Draw an unlabeled red rectangle at the normalized bbox on the full image."""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = norm_xyxy_to_pixels(bbox_norm, w, h)
        draw = ImageDraw.Draw(img)
        line_w = max(3, int(round(max(w, h) / 180)))
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=line_w)
        return img.copy()
