# RL stage

The reinforcement-learning stage of EviRover. It starts from the SFT checkpoint
and optimises the policy against verifiable task rewards, with the real tool
loop in the rollout: the model's own crops, searches and page fetches are part
of the trajectory, so the gradient reaches the decision to look again as well as
the final answer.

This is a vendored, trimmed copy of
[Vision-DeepResearch](https://github.com/Osilly/Vision-DeepResearch),
[rLLM](https://github.com/agentica-project/rllm) and
[verl](https://github.com/volcengine/verl) with local modifications to all
three. See `NOTICE` for what differs from upstream and under which licenses.

```
launch_scripts/train_rl.sh            launcher for a single 8-GPU node
scripts/register_dataset.py           register rl_pool.jsonl with the trainer
requirements.txt                      the versions the released run used
rllm/verl/                            verl: PPO/GRPO trainer, FSDP workers, vLLM rollout
rllm/rllm/                            rLLM: agent/workflow trainer, dataset registry, rewards
rllm/vision_deepresearch_async_workflow/
    deepresearch_agent.py             the multi-turn rollout loop
    deepresearch_workflow.py          episode assembly, reward dispatch, masking
    mm_agent/                         the ten tools, the system prompt, the tool spec
    run/                              the base trainer invocation
```

## Install

Python 3.10, CUDA 12.9, one node with 8 GPUs.

```bash
cd RL
pip install -e rllm/verl       # must be this copy, not PyPI verl
pip install -e rllm            # must be this copy, not PyPI rllm
pip install -r requirements.txt
```

Both packages are installed from this tree on purpose: the released run depends
on local changes to each, so a PyPI `verl`/`rllm` will not reproduce it.

For the segmentation reward and the `verify_mask` tool, also install
[facebook/sam3](https://github.com/facebookresearch/sam3) and point
`SAM3_CHECKPOINT` at its `sam3.pt`. To train without segmentation, set
`VDR_SAM_PREINIT=0` and drop the segmentation rows from the dataset.

## Data

Prompts come from
[`bunny127/EviRover-RL-12K`](https://huggingface.co/datasets/bunny127/EviRover-RL-12K).
Fetch it, unpack the media, then register the pool:

```bash
python3 scripts/register_dataset.py --jsonl /path/to/rl_pool.jsonl
```

That writes a parquet copy under the registry and prints the name to put in
`RLLM_DATASET` (default `EviRover-RL`). `--task` registers a subset, e.g.
`--task spot_diff counting --name EviRover-RL-sd-cnt`.

Image paths in the pool are relative, so `MM_DATA_ROOT` must point at the
directory the data release's `media/*.tar` were unpacked into. The registry
ships empty — the paths in it were machine-local.

## Run

```bash
MODEL_PATH=/path/to/sft_ckpt \
MM_DATA_ROOT=/path/to/rl-data/media \
SAM3_CHECKPOINT=/path/to/sam3.pt \
SERPER_API_KEY=... JINA_API_KEY=... \
SUMMARY_BASE_URL=... SUMMARY_MODEL=... SUMMARY_API_KEY=... \
  ./launch_scripts/train_rl.sh
```

`EXP`, `STEPS`, `SAVE_FREQ`, `LR` and `CUDA_VISIBLE_DEVICES` can be overridden;
anything else is forwarded to the trainer, so `./launch_scripts/train_rl.sh
data.train_batch_size=8` works.

`STEPS` is **the last `global_step` to run, not a number of steps** — with
`resume_mode=auto` a restart continues the same experiment, and `global_step`
keeps counting across restarts. Checkpoints only land on multiples of
`SAVE_FREQ`, so set `STEPS` to a multiple of it or the final step is not saved.

No credential has a default. The search, browse and image-search tools fail
loudly without one rather than returning nothing, because a run with a silently
dead tool is not comparable to one where it worked.

| Env | Needed for |
|---|---|
| `SERPER_API_KEY` | `text_search`, `text_search_image`, `image_search` |
| `SUMMARY_BASE_URL` / `SUMMARY_MODEL` / `SUMMARY_API_KEY` | `browse` summarises each fetched page with an LLM |
| `JINA_API_KEY` | improves `browse` fetch success |
| `ALIBABA_CLOUD_ACCESS_KEY_ID` / `_SECRET` / `OSS_ENDPOINT` / `OSS_BUCKET_NAME` | `image_search` uploads the crop, since reverse image search needs a public URL |
| `SAM3_CHECKPOINT` / `SAM3_REPO` | the segmentation reward and `verify_mask` |

Rather than exporting these one by one, put them in a file:

```bash
cp .env.example rllm/.env
$EDITOR rllm/.env
```

The run script sources `rllm/.env` if it exists, with xtrace disabled while it
does so that keys are not echoed into the training log. Point `ENV_FILE`
somewhere else to use a different file. `.env.example` lists every variable the
released configuration reads, which credential each tool needs, and the
non-secret knobs with the values the released run used.

## Method

The objective is GRPO with an **EMA-normalized advantage**
(`algorithm.adv_estimator=ema_grpo`, implemented in
`rllm/verl/verl/trainer/ppo/core_algos.py`). Instead of dividing the centred
group advantage by that group's own reward std, the scale comes from a per-task
exponential moving average, with an optional length channel. Training a mixture
whose tasks have very different reward variance otherwise lets one task's
variance dominate the gradient; normalising per task against a slow-moving
estimate keeps the share each task contributes stable.

The estimator reads:

| flag | released value |
|---|---|
| `ema_grpo_std_source` | `centered` |
| `ema_grpo_use_length` | `True` |
| `ema_grpo_decay` | `0.99` |
| `ema_grpo_prior_decay` | `0.99` |
| `ema_grpo_anchor` | `False` |
| `ema_grpo_clip` | `10.0` |

`VDR_ADV_TRACKER_PATH` persists the EMA state; without it a resumed run
re-warms from zero and the first ~1/(1-decay) steps are not reproducible. It
does **not** read `algorithm.norm_adv_by_std_in_grpo` and warns if it is set.

Other settings of the released run:

| | |
|---|---|
| Hardware | 1 node, 8 GPUs, FSDP actor + vLLM rollout, Ulysses sequence parallel 2 |
| Learning rate | 5e-7, forced past the value restored from the checkpoint |
| Clipping | `clip_ratio` 0.2 / 0.28, no KL penalty, `loss_agg_mode=token-mean` |
| Rollout sampling | `temperature` 1.0 / `top_p` 0.97 |
| Vision tower | frozen |
| Prompt / response budget | 8192 / 16384 tokens per turn |

A trajectory that exhausts its generation budget on degenerate text is scored 0
even if it lands on the right answer, so "ramble, then guess" is not reinforced.

## License

The three vendored projects keep their own licenses: MIT for
Vision-DeepResearch (`LICENSE-Vision-DeepResearch`), Apache-2.0 for rLLM
(`rllm/LICENSE`) and verl (`rllm/verl/LICENSE`). `NOTICE` records the
attribution and the modifications.
