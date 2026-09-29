"""SAM3 CUDA backend: turn a box (+ optional points) into a binary mask.

Mirrors the verified inference path in raw_data/sam3/segment_nano_bbox.py:
  build_sam3_image_model(device="cuda", ..., enable_inst_interactivity=True)
  -> Sam3Processor -> processor.set_image -> model.predict_inst(box=, point_coords=, ...)

Loaded lazily as a process-wide singleton and guarded by a lock (single GPU,
serial inference).
"""
import os
import sys
import threading
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from . import config

# Make the `sam3` package importable. If it is pip-installed nothing is needed;
# a plain clone of facebook/sam3 is supported by pointing SAM3_REPO at it. No
# fallback path is guessed — a missing sam3 surfaces as a loud ImportError below.
_SAM3_REPO = os.environ.get("SAM3_REPO", "")
if _SAM3_REPO and _SAM3_REPO not in sys.path:
    sys.path.insert(0, _SAM3_REPO)

_LOCK = threading.Lock()
_BACKEND: Optional["Sam3Backend"] = None


class Sam3Backend:
    def __init__(self, checkpoint: str = config.SAM3_CHECKPOINT, device: str = ""):
        # Check the weights before anything else. Without this, a missing checkpoint
        # runs the whole segmentation family as sam_error / iou=0 with parse_ok=True,
        # which reads as "the model got it wrong".
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {checkpoint}\n"
                "The segmentation family needs it. Download facebook/sam3's sam3.pt and set\n"
                "  export SAM3_CHECKPOINT=/path/to/sam3.pt\n"
                "or pass --no-sam to skip segmentation mask scoring entirely.")
        # A bare "cuda" resolves to GPU 0 here, so parallel jobs would stack their
        # SAM3 copies on one card. But build_sam3_image_model only accepts
        # "cuda"/"cpu" -- given "cuda:N" it silently leaves the weights on the CPU.
        # So select the device with set_device and hand the builder "cuda".
        device = device or os.getenv("EVAL_SAM_DEVICE") or f"cuda:{os.getenv('EVAL_SAM_GPU', '0')}"
        print(f"[sam3] loading backend -> {device} (pid={os.getpid()})", flush=True)
        import torch as _t
        _idx = None
        if device.startswith("cuda") and _t.cuda.is_available():
            _idx = int(device.split(":", 1)[1]) if ":" in device else _t.cuda.current_device()
            _t.cuda.set_device(_idx)
        device = "cuda" if _idx is not None else "cpu"
        self.device_index = _idx
        import torch  # local import so --no-sam runs need no torch/CUDA
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        self.device_type = device          # "cuda"/"cpu", for autocast in segment()
        # Do not enter autocast here: it and cuda.set_device are thread-local, so a
        # context entered at construction only covers whichever thread triggered the
        # lazy load. segment() establishes both per call instead.
        # tf32 is global, not thread-local, so it stays.
        if device == "cuda" and torch.cuda.is_available():
            if torch.cuda.get_device_properties(self.device_index or 0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.model = build_sam3_image_model(
            device=device,
            checkpoint_path=checkpoint,
            load_from_HF=False,
            compile=False,
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(self.model, device=device)

    def segment(
        self,
        image_path: str,
        box_px: Sequence[float],
        pos_pts_px: Optional[List[List[float]]] = None,
        neg_pts_px: Optional[List[List[float]]] = None,
    ) -> np.ndarray:
        """Return a boolean mask (H, W) for the given box + point prompts.

        Coordinates are in pixel space. Returns an all-False mask on failure.

        The device and autocast contexts are established on the calling thread;
        both are thread-local, so they cannot be set once in __init__.
        """
        torch = self.torch
        if self.device_index is not None:
            with torch.cuda.device(self.device_index), \
                    torch.autocast(self.device_type, dtype=torch.bfloat16):
                return self._segment(image_path, box_px, pos_pts_px, neg_pts_px)
        return self._segment(image_path, box_px, pos_pts_px, neg_pts_px)

    def _segment(
        self,
        image_path: str,
        box_px: Sequence[float],
        pos_pts_px: Optional[List[List[float]]] = None,
        neg_pts_px: Optional[List[List[float]]] = None,
    ) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        empty = np.zeros((h, w), dtype=bool)

        # Clip box to image bounds.
        x1, y1, x2, y2 = box_px
        x1 = max(0.0, min(float(x1), w - 1))
        y1 = max(0.0, min(float(y1), h - 1))
        x2 = max(0.0, min(float(x2), w - 1))
        y2 = max(0.0, min(float(y2), h - 1))
        if x2 <= x1 or y2 <= y1:
            return empty
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        pos_pts = pos_pts_px or []
        neg_pts = neg_pts_px or []
        point_coords = None
        point_labels = None
        if pos_pts or neg_pts:
            pts = [[float(p[0]), float(p[1])] for p in pos_pts + neg_pts]
            labels = [1] * len(pos_pts) + [0] * len(neg_pts)
            point_coords = np.array(pts, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

        state = self.processor.set_image(image)
        kwargs = dict(box=box, multimask_output=False)
        if point_coords is not None:
            kwargs["point_coords"] = point_coords
            kwargs["point_labels"] = point_labels
        masks, scores, _ = self.model.predict_inst(state, **kwargs)
        mask = np.asarray(masks[0])
        if mask.dtype != bool:
            mask = mask > 0.5
        return mask


def get_backend() -> Sam3Backend:
    """Lazily build the process-wide SAM3 backend."""
    global _BACKEND
    with _LOCK:
        if _BACKEND is None:
            _BACKEND = Sam3Backend()
    return _BACKEND


def segment(image_path, box_px, pos_pts_px=None, neg_pts_px=None) -> np.ndarray:
    """Thread-safe serial segmentation entry point."""
    backend = get_backend()
    with _LOCK:
        return backend.segment(image_path, box_px, pos_pts_px, neg_pts_px)
