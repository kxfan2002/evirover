#!/usr/bin/env bash
# Launch the EviRover SFT stage on a single 8-GPU node.
#
#   ENVBIN=/path/to/your/env/bin ./scripts/train_sft.sh
#
# Optional overrides: YAML, GPUS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YAML="${YAML:-examples/train_full/sft_qwen3vl_4b.yaml}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# The bin/ of the environment llamafactory is installed into. torchrun spawns its
# workers with whatever `python` is first on PATH, so this has to go in front --
# otherwise the workers pick up a different interpreter and die with
# "ModuleNotFoundError: No module named 'llamafactory'".
if [ -z "${ENVBIN:-}" ]; then
  echo "ENVBIN is not set. Point it at the bin/ directory of the environment" >&2
  echo "that has llamafactory installed, e.g. ENVBIN=/opt/conda/envs/lf/bin" >&2
  exit 1
fi
export PATH="$ENVBIN:$PATH"

export CUDA_VISIBLE_DEVICES="$GPUS"
export FORCE_TORCHRUN=1

# DeepSpeed JIT-compiles its cpu_adam op and asserts that the system CUDA
# toolkit version matches the one torch was built against. When they differ it
# aborts at optimizer init with a CUDAMismatchException, followed by a
# misleading "'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'". Set
# this when your /usr/local/cuda does not match torch's CUDA version.
export DS_SKIP_CUDA_CHECK=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HERE"
exec "$ENVBIN/llamafactory-cli" train "$YAML"
