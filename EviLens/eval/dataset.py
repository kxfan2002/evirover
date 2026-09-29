"""Load benchmark jsonl, resolve image paths, load GT (incl. RGBA mask fix)."""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from . import config


@dataclass
class Sample:
    id: str
    family: str              # grounding | segmentation | counting
    subcategory: str         # recognition | localization | "" (grounding only)
    task: str                # grounding_bbox | spot_diff | segmentation | counting
    file: str                # category file, e.g. grounding/recognition
    image_path: str          # absolute
    description: str
    gt: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def load_gt_mask(mask_abs_path: Optional[str]) -> Optional[np.ndarray]:
    """Load a GT mask (absolute path) as a boolean array (H, W).

    Benchmark masks are RGBA where the binary mask lives in the RGB channels
    (0/255) and the alpha channel is noise, so we read the R channel > 127.
    Single-channel (mode L) fallback masks are read directly. Returns None if
    the path is missing or the file is absent.
    """
    if not mask_abs_path or not os.path.isfile(mask_abs_path):
        return None
    im = Image.open(mask_abs_path)
    arr = np.array(im)
    if arr.ndim == 2:
        return arr > 127
    # RGB(A): use the red channel; ignore alpha.
    return arr[..., 0] > 127


def _resolve_mask(d: dict) -> Optional[str]:
    """Absolute GT mask path. All seg masks live under benchmark/masks/ and are
    referenced by mask_path_benchmark."""
    mb = d.get("mask_path_benchmark")
    return config.mask_path(mb) if mb else None


def _build_gt(d: dict, task: str) -> Dict[str, Any]:
    """Extract the GT payload the scorer for this task needs."""
    gt: Dict[str, Any] = {}
    if task == "grounding_bbox":
        gt["bbox"] = d.get("bbox_1000_xyxy")
    elif task == "spot_diff":
        gt["bboxes"] = d.get("bbox_1000_xyxy")
    elif task == "counting":
        gt["answer"] = d.get("answer")
    elif task == "segmentation":
        gt["mask_path"] = _resolve_mask(d)
        gt["bbox"] = d.get("bbox_1000_xyxy")
        gt["image_width"] = d.get("image_width")
        gt["image_height"] = d.get("image_height")
    return gt


def _samples_from_file(name: str, meta: Dict[str, str], limit: Optional[int]) -> List[Sample]:
    family = meta["family"]
    jsonl = os.path.join(config.BENCH_DIR, f"{name}.jsonl")
    samples: List[Sample] = []
    with open(jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Per-record fields written by restructure_benchmark.py.
            task = d.get("task")
            subcategory = d.get("subcategory", meta.get("subcategory", ""))
            tag = d.get("image_tag")
            img_abs = config.image_path(d["image_path"], tag)
            samples.append(
                Sample(
                    id=d["id"],
                    family=family,
                    subcategory=subcategory,
                    task=task,
                    file=name,
                    image_path=img_abs,
                    description=d.get("description", ""),
                    gt=_build_gt(d, task),
                    raw=d,
                )
            )
            if limit and len(samples) >= limit:
                break
    return samples


def load_samples(files: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Sample]:
    """Load samples for the given benchmark files (default: config.DEFAULT_FILES)."""
    files = files or list(config.DEFAULT_FILES)
    out: List[Sample] = []
    for name in files:
        if name not in config.BENCH_FILES:
            raise ValueError(f"unknown benchmark file: {name}")
        out.extend(_samples_from_file(name, config.BENCH_FILES[name], limit))
    return out
