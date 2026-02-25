#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

CUDA12_VENV_LIB="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
if [[ -d "$CUDA12_VENV_LIB" ]]; then
  export LD_LIBRARY_PATH="$CUDA12_VENV_LIB:${LD_LIBRARY_PATH:-}"
fi

if [[ -x "/usr/local/cuda/bin/ptxas" ]]; then
  export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "[run_phi4_reasoning_endpoints] vllm not found in PATH" >&2
  echo "Install first in venv: pip install 'vllm>=0.8.0'" >&2
  exit 1
fi

MODEL_ID="${MODEL_ID:-microsoft/Phi-4-reasoning-plus}"
HOST="${HOST:-127.0.0.1}"
PORT_FP8="${PORT_FP8:-8301}"
PORT_BF16="${PORT_BF16:-8302}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export HF_HOME="$HF_CACHE_DIR"

echo "[run_phi4_reasoning_endpoints] starting FP8 endpoint on :$PORT_FP8"
vllm serve "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT_FP8" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --quantization fp8 \
  $EXTRA_ARGS &
PID_FP8=$!

echo "[run_phi4_reasoning_endpoints] starting BF16 endpoint on :$PORT_BF16"
vllm serve "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT_BF16" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype bfloat16 \
  $EXTRA_ARGS &
PID_BF16=$!

cleanup() {
  kill "$PID_FP8" "$PID_BF16" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait