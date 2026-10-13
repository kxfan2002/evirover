import os

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "VLLM_USE_V1": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
        # Cap glibc malloc arenas so freed host memory (large per-step multimodal
        # pixel tensors) is returned to the OS instead of hoarded per-thread-arena
        # -> curbs the RSS climb that OOM-killed the 64-rollout run (2026-08-01).
        "MALLOC_ARENA_MAX": "2",
        # ARENA_MAX caps the NUMBER of arenas, not whether a freed block goes back
        # to the OS -- that is the mmap threshold, and glibc RAISES it dynamically
        # (up to 32MB) after the first large free. Past that point the per-step
        # multimodal host buffers come from the arena and free() never shrinks RSS;
        # only malloc_trim does, and we cannot call malloc_trim inside vLLM's
        # EngineCore because that loop is not ours. Pinning the threshold keeps
        # those buffers on the mmap path so free() == munmap().
        # Measured (smoke21b, 2026-08-19): driver and FSDP workers -- which DO get a
        # per-step malloc_trim -- were flat, while the vLLM engine processes climbed
        # 10.3 -> 62.0 GB over 21 steps, growth confined to generate and correlated
        # with image BYTES (r=+0.62, p=0.003), not image count.
        "MALLOC_MMAP_THRESHOLD_": "131072",
    },
    "worker_process_setup_hook": "rllm.patches.verl_patch_hook.setup",
}

FORWARD_PREFIXES = [
    "VLLM_",
    "SGL_",
    "SGLANG_",
    "HF_",
    "TOKENIZERS_",
    "DATASETS_",
    "TORCH_",
    "PYTORCH_",
    "DEEPSPEED_",
    "MEGATRON_",
    "NCCL_",
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "NV_",
    "NVIDIA_",
    # Reward-config vars (MM_REWARD_*, MM_DATA_ROOT, SAM3_*). The reward runs in
    # the driver today, but forward these so shaping/threshold overrides still
    # apply if reward computation ever moves to a Ray worker.
    "MM_",
    "SAM3_",
]


def _get_forwarded_env_vars():
    """
    Get the forwarded environment variables. The `RLLM_EXCLUDE` environment variable can be used to
    exclude specific environment variables or all variables with a specific prefix.

    Example:
    ```
    RLLM_EXCLUDE=VLLM*,CUDA*,NCCL_IB_DISABLE
    ```
    will exclude all variables with prefix `VLLM_`, `CUDA_`, and `NCCL_IB_DISABLE`.

    By default, all environment variables with prefix in `FORWARD_PREFIXES` are forwarded.
    """
    if os.environ.get("RLLM_EXCLUDE", None) is not None:
        rllm_exclude = str(os.environ.get("RLLM_EXCLUDE")).split(",")
    else:
        rllm_exclude = []

    forward_prefix = FORWARD_PREFIXES.copy()

    exclude_vars = set()
    for name in rllm_exclude:
        if "*" in name:  # denote a prefix match, e.g. "VLLM*"
            forward_prefix.remove(name.replace("*", "_"))
        else:
            exclude_vars.add(name)

    forwarded = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in forward_prefix) and k not in exclude_vars}
    return forwarded


def get_ppo_ray_runtime_env():
    env = PPO_RAY_RUNTIME_ENV["env_vars"].copy()
    env.update(_get_forwarded_env_vars())
    return {
        "env_vars": env,
        # "worker_process_setup_hook": PPO_RAY_RUNTIME_ENV["worker_process_setup_hook"],
    }
