#!/usr/bin/env bash
# EviRover RL stage: agentic perception RL on a single 8-GPU node.
#
#   MODEL_PATH=/path/to/sft_ckpt \
#   MM_DATA_ROOT=/path/to/rl-data/media \
#   SERPER_API_KEY=... JINA_API_KEY=... \
#   SUMMARY_BASE_URL=... SUMMARY_MODEL=... SUMMARY_API_KEY=... \
#   ./launch_scripts/train_rl.sh
#
# Register the prompts first:
#   python3 scripts/register_dataset.py --jsonl /path/to/rl_pool.jsonl
#
# No credential has a default here. The search, browse and image-search tools
# fail loudly without one, which is the intended behaviour: a run where a tool
# silently returned nothing is not comparable to one where it worked.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLLM="$HERE/rllm"

: "${MODEL_PATH:?set MODEL_PATH to the SFT checkpoint the policy starts from}"
: "${MM_DATA_ROOT:?set MM_DATA_ROOT to the directory rl-data/media/*.tar were unpacked into}"

EXP="${EXP:-evirover_rl}"
STEPS="${STEPS:-220}"              # the last global_step to run, not a step count
SAVE_FREQ="${SAVE_FREQ:-10}"
LR="${LR:-5e-7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# --- data and tools -------------------------------------------------------
export RLLM_DATASET="${RLLM_DATASET:-EviRover-RL}"
export RLLM_TOOLSET="${RLLM_TOOLSET:-all}"          # all ten tools
export MM_DATA_ROOT
export VDR_TEXT_SEARCH_BACKEND="${VDR_TEXT_SEARCH_BACKEND:-serper}"
export SERPER_BUDGET_LIMIT="${SERPER_BUDGET_LIMIT:-25}"

# --- SAM3, used by the segmentation reward and the verify_mask tool --------
export VDR_SAM_PREINIT="${VDR_SAM_PREINIT:-1}"
export VDR_SAM_GPU="${VDR_SAM_GPU:-7}"
: "${SAM3_CHECKPOINT:?set SAM3_CHECKPOINT to sam3.pt, or set VDR_SAM_PREINIT=0 to train without segmentation}"
export SAM3_CHECKPOINT
export SAM3_REPO="${SAM3_REPO:-}"
export MM_REWARD_SEG_REQUIRE_POINTS="${MM_REWARD_SEG_REQUIRE_POINTS:-1}"

# --- resume ---------------------------------------------------------------
# resume_mode=auto picks up this experiment's own latest checkpoint, so a run
# that died mid-way continues instead of restarting from step 0.
CKPT_DIR="checkpoints/${EXP}"
if [ -f "$RLLM/$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
  RESUME_ARGS=(trainer.resume_mode=auto)
else
  RESUME_ARGS=(trainer.resume_mode=disable)
fi

# The learning rate is restored from the checkpoint's lr_scheduler on resume,
# which silently overrides whatever is passed on the command line. This makes
# the value above win.
export VDR_FORCE_LR="$LR"

cd "$RLLM"
bash vision_deepresearch_async_workflow/run/vision_deepresearch_qwen3vl_4b_grpo_vllm_fsdp_1node.sh \
    algorithm.adv_estimator=ema_grpo \
    +algorithm.ema_grpo_std_source=centered \
    +algorithm.ema_grpo_use_length=True \
    +algorithm.ema_grpo_prior_decay=0.99 \
    +algorithm.ema_grpo_anchor=False \
    +algorithm.ema_grpo_decay=0.99 \
    +algorithm.ema_grpo_clip=10.0 \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.97 \
    trainer.total_training_steps=$((STEPS + 1)) \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$CKPT_DIR" \
    'trainer.logger=["console"]' \
    trainer.log_episodes=True \
    trainer.episode_log_dir="${EPISODE_LOG_DIR:-${TMPDIR:-/tmp}/episodes_${EXP}}" \
    "${RESUME_ARGS[@]}" \
    "$@"
rc=$?
echo "SCRIPT_EXIT_CODE=${rc}"
exit ${rc}
