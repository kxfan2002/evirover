#!/usr/bin/env python3
"""Fetch the benchmark images into benchmark/images/ and verify them.

The 674 images (918 MB) live in a HuggingFace dataset repo rather than in git.
Everything else (questions, GT masks) is in the checkout already.

    python3 scripts/download_images.py            # download, then verify
    python3 scripts/download_images.py --check     # verify what is on disk, no download

Every file is checked against scripts/images_manifest.json (size + sha256). A
missing or corrupt image is reported and exits non-zero -- it is never skipped,
because a silently absent image turns into a "model got it wrong" data point
instead of an infrastructure error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "scripts", "images_manifest.json")
IMAGES_DIR = os.path.join(REPO_ROOT, "benchmark", "images")
DEFAULT_REPO_ID = os.environ.get("BENCH_IMAGES_REPO", "bunny127/EviLens")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(files: dict, quick: bool) -> list:
    """Return a list of (relpath, reason) for every file that is not intact."""
    bad = []
    for i, (rel, meta) in enumerate(sorted(files.items()), 1):
        path = os.path.join(REPO_ROOT, "benchmark", rel)
        if not os.path.exists(path):
            bad.append((rel, "missing"))
            continue
        size = os.path.getsize(path)
        if size != meta["size"]:
            bad.append((rel, f"size {size} != expected {meta['size']}"))
            continue
        if not quick:
            got = sha256_of(path)
            if got != meta["sha256"]:
                bad.append((rel, f"sha256 {got[:12]}... != expected {meta['sha256'][:12]}..."))
        if i % 100 == 0:
            print(f"  verified {i}/{len(files)}", flush=True)
    return bad


def download(repo_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")
    print(f"[images] snapshot_download {repo_id} -> {IMAGES_DIR}", flush=True)
    os.makedirs(os.path.dirname(IMAGES_DIR), exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=os.path.join(REPO_ROOT, "benchmark"),
        allow_patterns=["images/**"],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help="HuggingFace dataset repo holding benchmark/images/")
    p.add_argument("--check", action="store_true", help="verify only, do not download")
    p.add_argument("--quick", action="store_true",
                   help="verify file size only, skip sha256 (much faster)")
    args = p.parse_args()

    with open(MANIFEST) as fh:
        man = json.load(fh)
    files = man["files"]
    print(f"[images] manifest: {man['n_files']} files, "
          f"{man['total_bytes'] / 1e6:.1f} MB", flush=True)

    if not args.check:
        download(args.repo_id)

    print("[images] verifying" + (" (size only)" if args.quick else " (size + sha256)"), flush=True)
    bad = verify(files, quick=args.quick)
    if bad:
        print(f"\n[images] {len(bad)}/{len(files)} file(s) NOT intact:", file=sys.stderr)
        for rel, why in bad[:20]:
            print(f"  {rel}: {why}", file=sys.stderr)
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more", file=sys.stderr)
        sys.exit(f"\nRefusing to report success. Re-run the download, or fetch "
                 f"{args.repo_id} manually into benchmark/images/.")
    print(f"[images] OK -- all {len(files)} images present and intact.")


if __name__ == "__main__":
    main()
