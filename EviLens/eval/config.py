"""Central configuration: paths, file->family+tag mapping, API defaults."""
import os

# Repo root = parent of this package. Everything (benchmark, results) is resolved
# relative to it, so the checkout works wherever it is cloned.
EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.environ.get("EVAL_BENCH_DIR") or os.path.join(EVAL_DIR, "benchmark")
IMAGES_DIR = os.path.join(BENCH_DIR, "images")
MASKS_DIR = os.path.join(BENCH_DIR, "masks")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# SAM3 checkpoint (official facebook/sam3, torch/CUDA), only needed for the
# segmentation family — `--no-sam` skips it entirely. Set SAM3_CHECKPOINT to the
# downloaded sam3.pt; the default is a conventional location under the checkout.
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT") or os.path.join(
    EVAL_DIR, "models", "sam3", "sam3.pt")

# Benchmark is organized into three top-level categories. `grounding` has two
# equally-weighted subcategories that are scored separately then macro-averaged:
#   recognition  - knowledge + search identification (anime/real/nano)
#   localization - pure fine-grained visual localization (visual probe + spot_diff)
# Each entry: relative jsonl path under BENCH_DIR -> (family, subcategory).
# Per-record `task` (grounding_bbox | spot_diff | segmentation | counting) selects
# the scorer; `image_tag` (per record) resolves the image path.
BENCH_FILES = {
    "grounding/recognition": {"family": "grounding", "subcategory": "recognition"},
    "grounding/localization": {"family": "grounding", "subcategory": "localization"},
    "grounding/spot_diff_only": {"family": "grounding", "subcategory": "localization"},
    "segmentation": {"family": "segmentation", "subcategory": ""},
    "counting": {"family": "counting", "subcategory": ""},
}

# What `--files` defaults to: the 688-question benchmark, each item exactly once.
# `grounding/spot_diff_only` is EXCLUDED on purpose — it is a strict subset of
# `grounding/localization` (the same 15 spot_diff ids), kept as a separate file
# only so spot_diff can be run alone. Including it in the default would score
# those 15 items twice and silently inflate spot_diff's weight in the report.
DEFAULT_FILES = [
    "grounding/recognition",
    "grounding/localization",
    "segmentation",
    "counting",
]

# Ordered categories and the grounding subcategories that macro-average into it.
# spot_diff records load under grounding/localization but are reported as their
# own top-level category (multi-box F1 is a different metric from grounding IoU).
CATEGORIES = ["grounding", "spot_diff", "segmentation", "counting"]
GROUNDING_SUBCATEGORIES = ["recognition", "localization"]

# task -> scorer family in the registry (grounding_bbox and spot_diff both live
# under the grounding-family scorers but use different scoring functions).
TASK_SCORER = {
    "grounding_bbox": "grounding",
    "spot_diff": "spot_diff",
    "segmentation": "segmentation",
    "counting": "counting",
}

# IoU threshold used for grounding Acc@0.5, seg IoU@0.5, and spot_diff matching.
IOU_THRESHOLD = 0.5

# Per-model coordinate convention for bbox/point answers. The pipeline (prompts,
# GT, scorers) is canonical xyxy / [x,y]. A few models ignore the prompt and emit
# their own native order — gemini returns yx (box [y1,x1,y2,x2], points [y,x]).
# The scorers normalize any "yx" model back to xyxy after parsing. Keys are
# matched as case-insensitive substrings of the --model string. Default: "xy".
MODEL_COORD_ORDER = {
    "gemini": "yx",
}


def coord_order_for(model: str) -> str:
    """Return "yx" or "xy" for a model name (substring match; default "xy")."""
    m = (model or "").lower()
    for key, order in MODEL_COORD_ORDER.items():
        if key in m:
            return order
    return "xy"

# API defaults (override via CLI / env).
DEFAULT_BASE_URL = os.environ.get("EVAL_BASE_URL", "")
DEFAULT_API_KEY = os.environ.get("EVAL_API_KEY", "")
DEFAULT_MODEL = os.environ.get("EVAL_MODEL", "")
DEFAULT_MAX_TOKENS = 16384
# Sampling params are pinned EXPLICITLY (not inherited from each model's
# generation_config) so every model under test is compared at the same settings.
# These are the standard Qwen values (temp 0.7 / top_p 0.8 / top_k 20).
# Escape hatch: pass a NEGATIVE value on the CLI (e.g. --temperature -1) to omit
# that field entirely, for endpoints that reject a given value or field.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MAX_WORKERS = 8
# Reasoning models can spend minutes on a single item, so keep the per-request
# timeout generous rather than counting slow-but-valid calls as errors.
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
# Rate-limit (HTTP 429) is distinctly recoverable and gets its own, more patient
# retry budget with longer backoff, independent of the general retry count.
DEFAULT_MAX_RETRIES_429 = 5
# Separate retry budget for empty responses (zero tokens in both content and
# reasoning). A model occasionally samples EOS as its first token: finish_reason is
# "stop" and nothing errors, so neither the general nor the 429 budget catches it.
DEFAULT_MAX_RETRIES_EMPTY = 5


def image_path(rel_path: str, tag: str) -> str:
    """Resolve a benchmark image's absolute path."""
    return os.path.join(IMAGES_DIR, tag, rel_path)


def mask_path(mask_rel_or_name: str) -> str:
    """Resolve a GT mask path. Benchmark masks live flat under benchmark/masks/."""
    return os.path.join(BENCH_DIR, mask_rel_or_name)
