# SFT stage

The supervised fine-tuning stage of EviRover. It takes Qwen3-VL-4B-Instruct and
trains it on multi-turn agent trajectories — the model's own reasoning, tool
calls and tool responses — so that it will drive the tool loop at all. The
checkpoint this produces is the starting point for the RL stage; it is not the
released model.

This is a trimmed copy of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
with one source patch and our training configuration. See `NOTICE` for exactly
what differs from upstream.

```
examples/train_full/sft_qwen3vl_4b.yaml   the training configuration
examples/deepspeed/                       upstream DeepSpeed configs (ZeRO-3 is used)
scripts/train_sft.sh                      launcher for a single 8-GPU node
data/dataset_info.json                    where the dataset gets registered
requirements.txt                          the versions the checkpoint was trained with
src/llamafactory/                         LLaMA-Factory source
```

## Install

Python 3.11, one node with 8 GPUs.

```bash
cd SFT-LLaMA-Factory
pip install -e .
pip install -r requirements.txt
```

`pip install -e .` resolves the dependency ranges in `pyproject.toml`;
`requirements.txt` then pins them to the versions this checkpoint was actually
trained with, and adds `deepspeed`, `liger-kernel` and `flash-attn`, which the
training config needs but upstream does not list as core dependencies. Install
the `torch` build that matches your driver — see the note at the top of
`requirements.txt`.

Two environment variables are easy to get wrong and are both handled by
`scripts/train_sft.sh`:

- The environment's `bin/` must come first on `PATH`. `torchrun` launches its
  workers with the first `python` it finds, so if that is a different
  interpreter they all die with `ModuleNotFoundError: No module named
  'llamafactory'`.
- `DS_SKIP_CUDA_CHECK=1` is needed when the system CUDA toolkit does not match
  the one `torch` was built against. Without it DeepSpeed aborts while
  JIT-compiling `cpu_adam`.

## Data

The training data is not in this repository; it will be released separately.
When you have it, put it anywhere you like and tell the config about it in two
places.

Expected format — one JSON object per line, ShareGPT style:

```json
{
  "messages": [
    {"role": "system",    "content": "<the agent system prompt>"},
    {"role": "user",      "content": "<image>\n<the question>"},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>...</tool_call>"},
    {"role": "user",      "content": "<tool_response>...</tool_response>"},
    {"role": "assistant", "content": "<think>...</think>\n<answer>...</answer>"}
  ],
  "images": ["relative/path/to/image.jpg"]
}
```

Notes on the format: `<image>` marks where each image is spliced into the text,
and the number of `<image>` placeholders across the conversation must equal the
length of `images`. Paths in `images` are resolved relative to `media_dir`, not
to the jsonl file. Tool results come back as `user` turns. Assistant turns keep
their `<think>` blocks — `preserve_thinking` and `enable_thinking` in the config
are what makes earlier turns' reasoning part of the training signal rather than
being stripped.

**1. Register it** in `data/dataset_info.json`. Set `file_name` to the jsonl
(an absolute path is fine, and is resolved independently of `media_dir`); the
key is what the config's `dataset:` field refers to.

```json
{
  "evirover_sft": {
    "file_name": "/path/to/sft_data/evirover_sft.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
      "role_tag": "role", "content_tag": "content",
      "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"
    }
  }
}
```

**2. Point the config at the images.** In
`examples/train_full/sft_qwen3vl_4b.yaml`, set `media_dir` to the directory the
`images` paths are relative to, and set `tokenized_path` to a path that is used
by this dataset and nothing else:

```yaml
dataset: evirover_sft
media_dir: /path/to/sft_data
tokenized_path: ./cache/tokenized/evirover_sft
```

`tokenized_path` matters more than it looks. The preprocessed cache is keyed by
that path alone, not by the contents of the dataset, so pointing two different
datasets at the same `tokenized_path` with `overwrite_cache: false` will silently
train on whichever one got preprocessed first. One path per dataset; change the
data, change the path (or set `overwrite_cache: true`).

## Train

```bash
ENVBIN=/path/to/your/env/bin ./scripts/train_sft.sh
```

`GPUS` and `YAML` can be overridden; defaults are all 8 GPUs and
`examples/train_full/sft_qwen3vl_4b.yaml`.

The configuration is a full fine-tune with the vision tower frozen — only the
language model and the multimodal projector are updated — under DeepSpeed
ZeRO-3, bf16, gradient checkpointing, FlashAttention-2 and fused linear
cross-entropy. `cutoff_len` is 24576 tokens, because a trajectory with several
images and tool responses is long; `image_max_pixels` is 2,160,000 so that most
training images are not downscaled, which matters when the targets are small.
Effective batch size is 64 (1 per device x 8 accumulation steps x 8 GPUs) at
`learning_rate` 5e-6, cosine schedule, 4 epochs.

Checkpoints land in `output_dir` every `save_steps` (30). Expect a full run to
be slow: high-resolution images plus ZeRO-3 dominate the step time.

Results and logs:

```
output/evirover_sft_qwen3vl_4b/
├── checkpoint-*/            one per save_steps
├── trainer_log.jsonl        per-logging_steps loss
├── trainer_state.json
└── training_loss.png
```

## What the next stage expects

The RL stage loads an SFT checkpoint directory as its initial policy. If the
environment you run RL in has a different `transformers` version than the one
used here, the checkpoint's `config.json`, `tokenizer_config.json` and
processor files may need to be converted to that version's schema before it
will load — the weights themselves are unaffected.

## License

Apache-2.0. This directory is a derivative of LLaMA-Factory, which is also
Apache-2.0; `LICENSE` is upstream's and `NOTICE` records what was changed.
