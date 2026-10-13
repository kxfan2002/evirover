#!/usr/bin/env bash
set -xeuo pipefail

# Train a Qwen3-VL SFT checkpoint with the agentic perception workflow.
# This is intentionally FSDP/HF based, not the original 30B-A3B Megatron/MoE script.

# ========= env / secrets =========
# Default to the repo's rllm/.env. Override with: ENV_FILE=/path/to/.env bash this_script.sh
# Read it with xtrace disabled so API keys are not printed by `set -x`.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RLLM_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${RLLM_DIR}/.env}"
if [ -f "${ENV_FILE}" ]; then
  _RESTORE_XTRACE=0
  case "$-" in
    *x*) _RESTORE_XTRACE=1; set +x ;;
  esac
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  if [ "${_RESTORE_XTRACE}" = "1" ]; then
    set -x
  fi
  unset _RESTORE_XTRACE
else
  echo "[WARN] ENV_FILE not found: ${ENV_FILE}"
fi
export MM_AGENT_IMAGE_CACHE="${MM_AGENT_IMAGE_CACHE:-${TMPDIR:-/tmp}/vdr_mm_tools}"
export SERPER_BUDGET_LIMIT="${SERPER_BUDGET_LIMIT:-25}"

# ========= rollout =========
rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
ulimit -n 65535

dtype="bfloat16"
adv_estimator="grpo"

# ========= DAPO / KL / PPO clip =========
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_high=0.28

# ========= batch =========
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-32}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-4}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-32}"
n_parallel_tasks="${N_PARALLEL_TASKS:-64}"
n_parallel_tools="${N_PARALLEL_TOOLS:-512}"

# ========= length =========
max_prompt_length="${MAX_PROMPT_LENGTH:-8192}"
max_response_length="${MAX_RESPONSE_LENGTH:-16384}"

use_dynamic_bsz=True
actor_ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
infer_ppo_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"

# ========= parallel / offload =========
gen_tp="${GEN_TP:-1}"
actor_param_offload="${ACTOR_PARAM_OFFLOAD:-False}"
actor_optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
ref_param_offload="${REF_PARAM_OFFLOAD:-True}"

# ========= sampling =========
temperature="${TEMPERATURE:-0.7}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"
val_top_p="${VAL_TOP_P:-0.95}"
loss_agg_mode="seq-mean-token-sum"

# ========= cluster =========
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
project_name='vision-deepresearch'
exp_name='qwen3vl_4b_projtrain_agent_rl'

# ========= paths =========
# The SFT checkpoint the policy starts from. Must be loadable by the
# transformers version in this environment.
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the SFT checkpoint directory}"
CKPTS_DIR="${CKPTS_DIR:-checkpoints/${project_name}/${exp_name}}"

python3 -m vision_deepresearch_async_workflow.train_deepresearch_workflow \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=128 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=${return_raw_chat} \
    data.seed=3407 \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.dtype=${dtype} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=${n_parallel_tasks} \
    rllm.workflow.n_parallel_tools=${n_parallel_tools} \
    rllm.compact_filtering.enable=True \
    rllm.compact_filtering.mask_unknown=True \
    rllm.compact_filtering.mask_error=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    rllm.stepwise_advantage.enable=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=${SAVE_FREQ:-1} \
    trainer.test_freq=${TEST_FREQ:-0} \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    "$@"
