# EviRover: Reinforcing Agentic Perception Beyond a Glance

[![Dataset on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-bunny127%2FEviLens-yellow)](https://huggingface.co/datasets/bunny127/EviLens)
[![Code license](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)
[![Data license](https://img.shields.io/badge/data-CC%20BY--NC%204.0-lightgrey)](LICENSE-DATA)

Code and data for *EviRover: Reinforcing Agentic Perception Beyond a Glance*.

```
EviLens/    the benchmark: 688 instances, the scorers, and the agent eval harness
```

Training code will be added here.

---

# EviLens

A human-verified benchmark of **688 instances** for evaluating *perception under
insufficient evidence* — cases where a single glance at the image is not enough to
produce the required perceptual output, yet the model must still commit to a box,
a mask, or a count.

Conventional perception benchmarks assume the image plus the model's parametric
knowledge suffice to resolve the query. EviLens breaks that assumption in three
ways: the target may be too small to resolve without inspecting the image at a
finer scale (often well under 0.1% of the image area), it may only be findable by
comparing or scanning multiple regions, or identifying it may require knowledge
from outside the image.

EviLens is therefore meant to be evaluated **agentically**: the model under test
runs in a **10-tool loop** — cropping and inspecting the image, verifying,
searching the web, browsing pages, executing code — and gathers the missing
evidence before answering. Everything is driven through an OpenAI-compatible chat
endpoint, so any served model can be evaluated. A direct single-pass baseline is a
meaningful but much weaker reference point.

## Contents

Five categories; localization, recognition, and spot-the-difference are the three
grounding categories.

| Category | n | Missing evidence | Metrics |
|---|---|---|---|
| `grounding` / localization | 140 | Target is explicitly named but tiny or hidden among clutter (high-resolution scenes, I-spy puzzles) | IoU, R@0.5 |
| `grounding` / recognition | 182 | Target is visually salient but referred to indirectly; needs external information to identify | IoU, R@0.5 |
| `spot_diff` | 15 | Target is defined relative to a second panel; needs cross-panel comparison | F1_micro, F1_macro (`f1`) |
| `segmentation` | 195 | Recognition setting, but the output is a mask (box → mask via SAM3) | gIoU, cIoU |
| `counting` | 156 | Recognition setting, but the output is an integer | accuracy, MAE |

The 15 spot-the-difference instances contain **79 annotated differences** in
total; `f1_micro` pools counts across images and is the primary metric, while
`f1` averages per-image F1. Every instance is manually verified.

Images are high resolution on purpose — median 1988×1614, up to 9504×6102.
Downscaling them changes the task. Each category reports its own domain-standard
metrics and they are **not** folded into a single score. `parse_fail_rate` is
reported separately from being wrong, so a model that never emits a parseable
answer is visibly distinct from one that answers badly.

The 15 `spot_diff` questions are part of `grounding/localization.jsonl` and are
lifted into their own reported category. `grounding/spot_diff_only.jsonl` is the
same 15 items as a standalone file, for running them alone; it is deliberately
**not** in the default file set, since including both would score them twice.

## Install

```bash
cd EviLens
pip install -r requirements.txt
python3 scripts/download_images.py     # 674 images, 918 MB, sha256-verified
```

The 674 images live in a HuggingFace dataset,
[`bunny127/EviLens`](https://huggingface.co/datasets/bunny127/EviLens), rather
than in git; every file is verified by size and sha256 against
`scripts/images_manifest.json`. Override the source with
`BENCH_IMAGES_REPO=<org>/<dataset>` or `--repo-id`. The questions and the 195
ground-truth masks are already in this checkout.

For the `segmentation` family, also install
[facebook/sam3](https://github.com/facebookresearch/sam3) and point
`SAM3_CHECKPOINT` at its `sam3.pt` (`SAM3_REPO` too, if you use a plain clone
rather than a pip install). Without SAM3, run with `SKIP_SEG=1`.

## Run

```bash
cd EviLens

# A) any OpenAI-compatible endpoint
MODEL_NAME=my-model BASE_URL=https://api.example.com/v1 API_KEY=sk-... \
  ./run_eval.sh my_run

# B) a local checkpoint — the script starts vLLM for you
MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct SERVE_GPU=0 ./run_eval.sh my_run
```

Results go to `results/my_run/`: one `.jsonl` per benchmark file with per-sample
`score`, `parse_ok`, `components`, `pred`, `gt` and the raw output, plus
`summary.json` and a printed table. Re-running the same output directory resumes
and skips ids that already finished.

Smoke test without touching any API:

```bash
python3 run_eval.py --dry-run --no-sam --limit 3
```

`run_eval.sh` is a thin wrapper — `python3 run_eval.py --help` exposes the rest
(`--files`, `--ids`, `--repeat`, `--limit`, `--coord-order`, `--coord-space`, …).

### Tool credentials

The agent's 10 tools are `crop`, `verify`, `verify_part`, `verify_mask`,
`compare_lr`, `python`, and `text_search`, `browse`,
`text_search_image`, `image_search`.

| Env | Needed for | Required? |
|---|---|---|
| `SERPER_API_KEY` | `text_search`, `text_search_image`, `image_search` | yes |
| `SUMMARY_BASE_URL` / `SUMMARY_MODEL` / `SUMMARY_API_KEY` | `browse` summarizes each fetched page with an LLM | yes |
| `ALIBABA_CLOUD_ACCESS_KEY_ID` / `_SECRET` / `OSS_ENDPOINT` / `OSS_BUCKET_NAME` | `image_search` uploads the crop, since Serper Lens only accepts a public URL | yes for `image_search` |
| `PPLX_API_KEY` | only with `--text-search-backend perplexity` | no |
| `JINA_API_KEY` | improves `browse` fetch success | no |

**`run_eval.py` refuses to start when a tool dependency is missing or
unreachable**, rather than producing a complete-looking `summary.json` in which
that tool silently failed all 688 times. It preflights the search keys, actually
fetches a page through `browse`, calls the summarizer, and checks the object
storage config. Each check has its own escape hatch
(`--allow-missing-search-key`, `--allow-broken-browse`, `--allow-broken-summary`,
`--allow-broken-image-search`) which prints a warning that the resulting numbers
are not comparable to a fully-configured run.

### Comparability

Numbers are only comparable across runs that share a protocol. What matters:

- **Sampling.** Pinned explicitly (`temperature` 0.7 / `top_p` 0.8 / `top_k` 20)
  rather than inherited from each model's `generation_config`. Pass a *negative*
  value to omit a field for endpoints that reject it.
- **Tool budget.** `--max-tool-calls 35`, which sets `--max-turns` to 39.
- **Rollouts.** `spot_diff` is repeated 8× per question and averaged before the
  metric (`--repeat 8 --repeat-tasks spot_diff`); `n` stays 15. With only 15
  images it is the one category whose sample size cannot be grown, and a single
  rollout is too noisy to rank models on. The other four categories are single-
  rollout.
- **Tool availability.** A run with a degraded tool cannot be tabulated against
  one without — hence the preflight above.
- **Coordinate convention.** The pipeline is canonically `xyxy` normalized to
  0–1000. Models that ignore the prompt and emit their own order are normalized
  after parsing (`--coord-order`, and `MODEL_COORD_ORDER` in `eval/config.py`;
  Gemini, for instance, returns `yx`).

## Layout

Everything below lives under `EviLens/`.

```
run_eval.py                 entry point: load -> agent rollout -> score -> report
run_eval.sh                 launcher for the published protocol
unified_system_prompt.txt   the agent system prompt, used verbatim
eval/config.py              paths, file->family map, default files, API defaults
eval/prompts.py             system prompts and per-family answer-format hints
eval/dataset.py             jsonl loading, image path resolution, GT loading
eval/client.py              OpenAI-compatible client (base64 images, retries)
eval/serve.py               optional vLLM auto-launch for --model-path
eval/parsing.py             <answer> extraction; tolerant bbox / int / seg parsing
eval/scorers/               per-family scoring
eval/sam_backend.py         SAM3 singleton: box -> mask
eval/report.py              per-file / per-family / overall aggregation
eval/agent/runner.py        the tool-calling loop and turn budget
eval/agent/tools/           the 10 tools
benchmark/                  questions (jsonl) and GT masks; images fetched separately
scripts/download_images.py  fetches and verifies benchmark/images/
results_baseline/           reference numbers from the paper
```

`eval/agent/tools/messages.py` holds every failure string the model can see. It
is kept byte-identical to the training-side copy: evaluation and training must
show the model the same tool feedback, or the same checkpoint is not comparable
between them.

## License

Code is Apache-2.0 (`LICENSE`). The benchmark data — questions, masks, and the
separately-hosted images — is CC BY-NC 4.0 (`LICENSE-DATA`); note that many
images came from public web sources and remain under their owners' copyright.


## Citation

```bibtex
```
