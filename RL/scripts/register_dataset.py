#!/usr/bin/env python3
"""Register the released RL pool with rLLM's dataset registry.

The trainer loads its prompts through `DatasetRegistry`, which keeps a parquet
copy per dataset name. This converts `rl_pool.jsonl` from the data release into
such an entry.

    python3 scripts/register_dataset.py --jsonl /path/to/rl_pool.jsonl

    # register only part of the pool
    python3 scripts/register_dataset.py --jsonl rl_pool.jsonl --task spot_diff counting
    python3 scripts/register_dataset.py --jsonl rl_pool.jsonl --name EviRover-RL-grounding \
        --task grounding_bbox

The name printed at the end is what goes into RLLM_DATASET.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="rl_pool.jsonl from the data release")
    p.add_argument("--name", default="EviRover-RL", help="dataset name to register as")
    p.add_argument("--split", default="train")
    p.add_argument("--task", nargs="*", default=None,
                   help="keep only these task values (default: keep everything)")
    args = p.parse_args()

    rows = []
    with open(args.jsonl, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{args.jsonl}:{n}: not valid JSON ({e})")
            for key in ("id", "task", "question", "images", "answer"):
                if key not in r:
                    sys.exit(f"{args.jsonl}:{n}: row is missing required key {key!r}")
            if args.task and r["task"] not in args.task:
                continue
            rows.append(r)

    if not rows:
        sys.exit("no rows selected -- check --task")

    try:
        from rllm.data.dataset import DatasetRegistry
    except ImportError as e:
        sys.exit(f"cannot import rllm ({e}). Install it first: pip install -e rllm")

    DatasetRegistry.register_dataset(args.name, rows, split=args.split)

    by_task: dict[str, int] = {}
    for r in rows:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    print(f"registered {len(rows)} rows as {args.name!r} (split={args.split!r})")
    for k in sorted(by_task):
        print(f"  {by_task[k]:6d}  {k}")
    print(f"\nSet RLLM_DATASET={args.name} and point MM_DATA_ROOT at the directory")
    print("the data release's media/*.tar were unpacked into.")


if __name__ == "__main__":
    main()
