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
import os
import shutil
from dataclasses import dataclass
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

    def save_pil(self, image: Image.Image, role: str) -> StoredImage:
        image_id = self._next_id()
        dest = self.image_dir / f"{self.sample_run_id}-{image_id.lower()}.jpg"
        if image.mode != "RGB":
            image = image.convert("RGB")
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


def image_to_data_url(path: str, max_side: int = 1536, quality: int = 90) -> str:
    """Load an image, downscale the long side to <=max_side, return a JPEG data URL."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as img:
        if img.mode in ("RGBA", "P", "L", "LA"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
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


# Qwen3-VL image processor area cap (preprocessor_config.json max_pixels). Crops are
# delivered to the processor BY PATH, so any upscale beyond this cap is downscaled
# right back away -> wasted vision tokens. Overridable for ablation / config drift.
_CROP_MAX_PIXELS = int(os.environ.get("VDR_CROP_MAX_PIXELS", "2160000"))


def crop_image(image_path: str, bbox_norm: Iterable[float], padding: float = 0.1,
               scale: int = 2) -> Image.Image:
    """Crop a normalized-0-1000 xyxy region, pad, and ADAPTIVELY upscale for detail.

    The `scale`x LANCZOS upscale sharpens the vision patch-grid of small commit crops
    (validated: dropping it costs ~2.5-2.7sigma on small-target localization). But for
    a large crop the upscaled area exceeds the processor's max_pixels cap and is simply
    downscaled back -> pure token waste. So we apply the full `scale`x ONLY when the
    upscaled crop still fits under the cap (native area <= max_pixels/scale^2); larger
    crops are delivered at native resolution. A/B on localization: this matches
    blanket-2x accuracy (<1sigma) while cutting ~17% of crop-image tokens.

    VDR_CROP_SCALE=N forces a fixed Nx (disables adaptive) for ablation.
    """
    forced = os.environ.get("VDR_CROP_SCALE")
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
        if forced is not None:
            eff_scale = int(forced)
        elif scale > 1 and cropped.width * cropped.height * (scale * scale) <= _CROP_MAX_PIXELS:
            eff_scale = scale          # upscale fully realized under the cap -> keep it
        else:
            eff_scale = 1              # would be clipped by the cap anyway -> native res
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

    Upscaling follows this side's adaptive rule (area vs _CROP_MAX_PIXELS), judged
    on the STITCHED image since the composite is what reaches the model.
    """
    forced = os.environ.get("VDR_CROP_SCALE")
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

        if forced is not None:
            eff_scale = int(forced)
        elif out.width * out.height * 4 <= _CROP_MAX_PIXELS:
            eff_scale = 2
        else:
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
