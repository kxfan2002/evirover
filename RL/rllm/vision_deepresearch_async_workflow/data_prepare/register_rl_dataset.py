import argparse
import json
import random
from pathlib import Path
from typing import Any

# Fields that vary in type across task families (flat list / None / list-of-list)
# and therefore cannot share one parquet column. They are JSON-encoded on
# register and JSON-decoded on load. Keep in sync with the decoder in
# vision_deepresearch_async_workflow/train_deepresearch_workflow.py.
JSON_ENCODED_FIELDS = ("bbox_1000_xyxy", "answer")



def parse_args():
    parser = argparse.ArgumentParser(description="Register Vision-DeepResearch RL JSONL data to rllm")

    parser.add_argument(
        "--jsonl_path",
        type=str,
        default=None,
        help="Single JSONL file to load. By default it is shuffled and split by --train_ratio.",
    )
    parser.add_argument(
        "--train_jsonl_path",
        type=str,
        default=None,
        help="Explicit train JSONL file. Use this for final/rl_vdr_unified/train.jsonl.",
    )
    parser.add_argument(
        "--test_jsonl_path",
        type=str,
        default=None,
        help="Optional explicit test JSONL file. Empty/missing files are skipped.",
    )
    parser.add_argument(
        "--register_name",
        type=str,
        default="Vision-DeepResearch-QA",
        help="Name for the registered dataset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train ratio when using --jsonl_path without --no_split.",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Register --jsonl_path entirely as train and do not create a test split.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for shuffling when splitting --jsonl_path.",
    )

    return parser.parse_args()


def load_jsonl(file_path: str | Path | None) -> list[dict[str, Any]]:
    if not file_path:
        return []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    data: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            data.append(item)
    return data


def normalize_sample(item: dict[str, Any], fallback_id: int) -> dict[str, Any]:
    """Keep all unified RL fields; only fill minimal compatibility defaults."""
    out = dict(item)
    out.setdefault("id", str(fallback_id))

    if "question" not in out:
        query = out.get("query")
        if query is None:
            raise ValueError(f"sample {out['id']} has neither question nor query")
        out["question"] = query

    if "images" not in out:
        image_path = out.get("image_path")
        out["images"] = [image_path] if image_path else []

    if "image_path" not in out and out.get("images"):
        first = out["images"][0]
        if isinstance(first, str):
            out["image_path"] = first

    # Multi-task RL rows are heterogeneous: `bbox_1000_xyxy` / `answer` are a
    # flat [x,y,x,y] for grounding/segmentation, None for counting, and a
    # list-of-boxes for spot_diff. A single parquet column cannot mix those
    # arrow types ("cannot mix list and non-list"), so serialize them to JSON
    # strings here. They are decoded back to Python objects at load time in
    # train_deepresearch_workflow (see _decode_json_fields). `raw` is dropped:
    # it is redundant (top-level fields duplicate it) and nothing reads it.
    out.pop("raw", None)
    for _k in JSON_ENCODED_FIELDS:
        if _k in out and not isinstance(out[_k], str):
            out[_k] = json.dumps(out[_k])

    return out


def process_data(data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_sample(item, i) for i, item in enumerate(data_list)]


def register_dataset(data: list[dict[str, Any]], register_name: str, split: str):
    if not data:
        print(f"Skip empty split '{split}' for dataset '{register_name}'")
        return None
    from rllm.data.dataset import DatasetRegistry

    registry_dataset = DatasetRegistry.register_dataset(register_name, data, split)
    print(f"Registered {len(data)} samples to DatasetRegistry as '{register_name}' ({split})")
    return registry_dataset


def main():
    args = parse_args()

    if args.train_jsonl_path:
        train_data = load_jsonl(args.train_jsonl_path)
        test_data = load_jsonl(args.test_jsonl_path)
    elif args.jsonl_path:
        all_data = load_jsonl(args.jsonl_path)
        if args.no_split:
            train_data, test_data = all_data, []
        else:
            random.seed(args.random_seed)
            random.shuffle(all_data)
            split_idx = int(len(all_data) * args.train_ratio)
            train_data, test_data = all_data[:split_idx], all_data[split_idx:]
    else:
        raise ValueError("Provide --train_jsonl_path or --jsonl_path")

    train_output = process_data(train_data)
    test_output = process_data(test_data)

    print(f"Register Name: {args.register_name}")
    print(f"Train samples: {len(train_output)}")
    print(f"Test samples: {len(test_output)}")

    register_dataset(train_output, args.register_name, "train")
    register_dataset(test_output, args.register_name, "test")

    print("Dataset registration completed.")


if __name__ == "__main__":
    main()
