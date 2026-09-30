# Reference numbers

Results on EviLens as reported in the paper.

## Protocol

As recorded for these runs:

```
--mode agent  --max-tool-calls 35  (--max-turns 39, automatic)
--text-search-backend serper  --serper-per-sample 10-15
--temperature 0.7  --top-p 0.8  --top-k 20
spot_diff: --repeat 8 --repeat-tasks spot_diff  (averaged per question; n stays 15)
browse summarizer: an OpenAI-compatible LLM; page retrieval via Jina
segmentation: predicted box -> mask with SAM3
served with vLLM 0.24.0+cu129 / torch 2.11
```

## Results

| Model | Loc IoU | Loc R@.5 | Rec IoU | Rec R@.5 | SD F1_mi | SD F1_ma | Seg gIoU | Seg cIoU | Cnt Acc (%) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | 0.276 | 0.251 | 0.265 | 0.231 | 0.005 | 0.008 | 0.203 | 0.279 | 19.2 |
| OneThinker-8B | 0.110 | 0.086 | 0.353 | 0.357 | 0.065 | 0.106 | 0.409 | 0.387 | 25.6 |
| InternVL-3.5-8B | 0.020 | 0.007 | 0.093 | 0.033 | 0.000 | 0.000 | 0.081 | 0.081 | 20.5 |
| **EviRover (4B)** | **0.444** | **0.500** | **0.648** | **0.687** | **0.171** | **0.266** | **0.686** | **0.708** | **50.0** |

Loc / Rec are the localization and recognition grounding categories; SD is
spot-the-difference, where micro-F1 is the primary metric. Counting accuracy is
shown as a percentage here; `summary.json` reports `accuracy` as a fraction.


## Files

One `summary.json` per model, in the format `run_eval.py` writes: per-category
`n`, `parse_ok`, `parse_fail_rate`, `mean_score`, `finish_reasons`, and that
category's metrics, over all 688 instances. Only these aggregates are published:
no per-sample jsonl, no model outputs, no trajectories.

```
results_baseline/
├── internvl35_8b/summary.json
├── onethinker8b/summary.json
└── evirover/summary.json
```
