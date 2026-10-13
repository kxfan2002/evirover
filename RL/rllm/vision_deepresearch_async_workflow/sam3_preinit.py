
from __future__ import annotations

import os
import tempfile
import time

import numpy as np


def _enabled() -> bool:
    return os.getenv("VDR_SAM_PREINIT", "0").lower() in {"1", "true", "yes"}


_DONE = False


def preinit_sam3() -> None:
    global _DONE
    if _DONE:
        return
    if not _enabled():
        print("[sam3_preinit] VDR_SAM_PREINIT is off, skipping (segmentation will score 0 -- make sure that is intended)")
        return

    import torch
    from PIL import Image

    gpu = int(os.getenv("VDR_SAM_GPU", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    before = torch.cuda.memory_allocated(gpu) if torch.cuda.is_available() else 0

    from rllm.rewards.mm_deepresearch_reward import _sam_checkpoint, segment_with_sam3

    ckpt = _sam_checkpoint()
    if not os.path.exists(ckpt):
        raise RuntimeError(f"[sam3_preinit] SAM3 weights not found: {ckpt} (set SAM3_CHECKPOINT)")

    tmp_dir = os.getenv("VDR_TMP") or tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    probe = os.path.join(tmp_dir, "_sam3_preinit_probe.png")
    if not os.path.exists(probe):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[64:192, 64:192] = 255
        Image.fromarray(arr).save(probe)

    os.environ.setdefault("MM_REWARD_SAM_DEVICE", f"cuda:{gpu}")

    t0 = time.time()
    mask = segment_with_sam3(probe, [64, 64, 192, 192], [[128, 128]], [[8, 8]])
    dt = time.time() - t0

    if mask is None or not hasattr(mask, "shape"):
        raise RuntimeError("[sam3_preinit] SAM3 returned an invalid mask, refusing to start")

    if torch.cuda.is_available():
        resident = (torch.cuda.memory_allocated(gpu) - before) / 2**30
        reserved = torch.cuda.memory_reserved(gpu) / 2**30
        print(
            f"[sam3_preinit] OK  pid={os.getpid()}  gpu={gpu}  load + first call {dt:.1f}s  "
            f"resident {resident:.2f} GiB  reserved {reserved:.2f} GiB  ckpt={ckpt}"
        )
    else:
        print(f"[sam3_preinit] OK (CPU)  load + first call {dt:.1f}s  ckpt={ckpt}")

    _DONE = True
