#!/usr/bin/env bash
# One-shot launcher for the full 688-question benchmark (agent mode, all 9 tools).
#
#   # A) evaluate any OpenAI-compatible endpoint
#   MODEL_NAME=gpt-5.6 BASE_URL=https://api.example.com/v1 API_KEY=sk-... \
#     ./run_eval.sh my_run
#
#   # B) evaluate a local checkpoint (script starts vLLM for you)
#   MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct SERVE_GPU=0 ./run_eval.sh my_run
#
# Credentials come from the environment; nothing is baked into this file.
# See README.md for what each one is for and which are optional.
set -uo pipefail

RUN_NAME="${1:-run}"
OUT_DIR="${OUT_DIR:-results/$RUN_NAME}"

# ---------------------------------------------------------------- credentials
# Required for the search tools. text_search uses Serper here (see README for
# the Perplexity backend). SUMMARY_* drives browse's page summarizer.
SERPER_API_KEY="${SERPER_API_KEY:-}"
SUMMARY_BASE_URL="${SUMMARY_BASE_URL:-}"
SUMMARY_MODEL="${SUMMARY_MODEL:-}"
SUMMARY_API_KEY="${SUMMARY_API_KEY:-}"
export SERPER_API_KEY SUMMARY_BASE_URL SUMMARY_MODEL SUMMARY_API_KEY
# image_search uploads each crop to object storage (Serper Lens needs a public
# URL): ALIBABA_CLOUD_ACCESS_KEY_ID / _SECRET / OSS_ENDPOINT / OSS_BUCKET_NAME.
# JINA_API_KEY is optional and only improves browse's fetch success rate.

# --------------------------------------------------------------- SAM3 (seg)
# Mask IoU for the segmentation family needs facebook/sam3. Without it, run with
# SKIP_SEG=1 (segmentation is then simply not evaluated) -- do NOT pass --no-sam
# and report a segmentation number, it would only measure answer parsing.
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$PWD/models/sam3/sam3.pt}"
export SAM3_CHECKPOINT
[ -n "${SAM3_REPO:-}" ] && export SAM3_REPO

# ------------------------------------------------------------------- protocol
# The published protocol: 35 tool calls per question (--max-turns is then
# automatically 39) and temperature 0.7, with the remaining sampling fields
# pinned so every model is compared at the same settings. Endpoints that reject a
# field want a NEGATIVE value, which omits it entirely (e.g. TEMPERATURE=-1 for
# models that only accept temperature=1).
#
# spot_diff is repeated 8 times per question and averaged before the metric: with
# only 15 images it is the one category whose sample size cannot be grown, and a
# single rollout is too noisy to compare models on. `n` stays 15 either way.
COMMON=(--mode agent
        --max-tool-calls "${MAX_TOOL_CALLS:-35}"
        --repeat "${REPEAT:-8}" --repeat-tasks "${REPEAT_TASKS:-spot_diff}"
        --text-search-backend "${TEXT_SEARCH_BACKEND:-serper}"
        --serper-per-sample "${SERPER_PER_SAMPLE:-15}"
        --temperature "${TEMPERATURE:-0.7}"
        --top-p "${TOP_P:-0.8}"
        --top-k "${TOP_K:-20}"
        --max-workers "${MAX_WORKERS:-8}"
        --timeout "${TIMEOUT:-1800}")

# ---------------------------------------------------------- model under test
if [ -n "${MODEL_PATH:-}" ]; then
  COMMON+=(--model-path "$MODEL_PATH" --serve-gpu "${SERVE_GPU:-0}"
           --serve-port "${SERVE_PORT:-8000}")
  # SAM3 (~5.4 GB) shares the card with vLLM, so leave it room.
  export EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.8}"
  export EVAL_SAM_GPU="${EVAL_SAM_GPU:-${SERVE_GPU:-0}}"
elif [ -n "${BASE_URL:-}" ]; then
  COMMON+=(--base-url "$BASE_URL" --model "${MODEL_NAME:?set MODEL_NAME with BASE_URL}"
           --api-key "${API_KEY:-}")
else
  echo "Set MODEL_PATH (serve locally) or BASE_URL + MODEL_NAME (remote endpoint)." >&2
  exit 2
fi

# The local vLLM server must not be sent through an HTTP proxy.
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"

# ------------------------------------------------------------------- run
# One pass over all four families -> a single results dir and one combined
# report. Per-file jsonl is written incrementally, so re-running the same --out
# resumes and skips ids that already finished.
#
# SPLIT_SEG=1 instead runs it as two passes (the other 493 items, then the 195
# segmentation ones). Use it when vLLM plus a GPU-resident SAM3 will not fit on
# one card; the trade-off is two summary.json files to read side by side.
FILES_MAIN=(grounding/recognition grounding/localization counting)
[ "${SKIP_SEG:-0}" = "1" ] || FILES_SEG=(segmentation)

set -x
if [ "${SPLIT_SEG:-0}" = "1" ]; then
  python3 run_eval.py "${COMMON[@]}" --files "${FILES_MAIN[@]}" --out "$OUT_DIR/main"
  rc=$?
  if [ "${SKIP_SEG:-0}" != "1" ]; then
    python3 run_eval.py "${COMMON[@]}" --files "${FILES_SEG[@]}" --out "$OUT_DIR/seg"
    rc=$(( rc | $? ))
  fi
  set +x
  echo "Summaries: $OUT_DIR/main/summary.json  $OUT_DIR/seg/summary.json"
else
  python3 run_eval.py "${COMMON[@]}" \
    --files "${FILES_MAIN[@]}" ${FILES_SEG+"${FILES_SEG[@]}"} --out "$OUT_DIR"
  rc=$?
  set +x
  echo "Summary: $OUT_DIR/summary.json"
fi

exit "$rc"
