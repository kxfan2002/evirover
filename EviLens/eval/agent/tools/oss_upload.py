"""Upload a local image to Aliyun OSS and return a public URL.

image_search (reverse image) needs a publicly reachable image URL for Serper's
Lens endpoint. This ports the OSS path from the SFT image_search tool. Keys are
read from env ONLY (no in-code defaults). If oss2 is missing or upload fails,
callers must degrade gracefully.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
import time
from typing import Optional, Tuple

# All four are read from the environment, with no in-code defaults: the bucket and
# endpoint differ per deployment, and hardcoding one yields 403s instead of the
# real reason. `get(K) or ""` rather than `get(K, "")` so an env var set to the
# empty string is treated as unset.
OSS_ACCESS_KEY_ID = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID") or ""
OSS_ACCESS_KEY_SECRET = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET") or ""
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT") or ""
OSS_BUCKET_NAME = os.environ.get("OSS_BUCKET_NAME") or ""


def upload(local_path: str) -> Tuple[str, str]:
    """Upload and return (public_url, oss_key). Raises on failure."""
    import oss2

    missing = [k for k, v in (
        ("ALIBABA_CLOUD_ACCESS_KEY_ID", OSS_ACCESS_KEY_ID),
        ("ALIBABA_CLOUD_ACCESS_KEY_SECRET", OSS_ACCESS_KEY_SECRET),
        ("OSS_ENDPOINT", OSS_ENDPOINT),
        ("OSS_BUCKET_NAME", OSS_BUCKET_NAME),
    ) if not v]
    if missing:
        raise RuntimeError("OSS not configured, missing env: " + ", ".join(missing))
    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)

    with open(local_path, "rb") as f:
        img_bytes = f.read()
    if not img_bytes:
        raise IOError(f"image file is empty: {local_path}")

    seed = f"{local_path}:{time.time()}".encode("utf-8")
    suffix = hashlib.md5(seed).hexdigest()[:10]
    oss_key = f"tmp/image2image/{suffix}-{os.path.basename(local_path)}"
    mime, _ = mimetypes.guess_type(local_path)
    headers = {"Content-Type": mime} if mime else {}
    bucket.put_object(oss_key, img_bytes, headers=headers)
    url = f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT.replace('https://', '')}/{oss_key}"
    return url, oss_key


def delete(oss_key: Optional[str]) -> None:
    if not oss_key:
        return
    try:
        import oss2

        auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
        bucket.delete_object(oss_key)
    except Exception:  # noqa: BLE001
        pass
