#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-microsoft/Phi-4-reasoning-plus}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
HOST_PORT_FP8="${HOST_PORT_FP8:-8301}"
HOST_PORT_BF16="${HOST_PORT_BF16:-8302}"
START_BF16="${START_BF16:-0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DUAL_MAX_MODEL_LEN="${DUAL_MAX_MODEL_LEN:-8192}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
FP8_DISABLED_KERNELS="${FP8_DISABLED_KERNELS:-FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel}"
REQUIRE_FREE_MB_FP8="${REQUIRE_FREE_MB_FP8:-20000}"
REQUIRE_FREE_MB_BF16="${REQUIRE_FREE_MB_BF16:-35000}"
WAIT_FOR_HEALTH="${WAIT_FOR_HEALTH:-1}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-900}"
HEALTH_POLL_SEC="${HEALTH_POLL_SEC:-5}"

EAGER_FLAG=""
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  EAGER_FLAG="--enforce-eager"
fi

MODEL_LEN_FP8="$MAX_MODEL_LEN"
MODEL_LEN_BF16="$MAX_MODEL_LEN"
if [[ "$START_BF16" == "1" ]]; then
  MODEL_LEN_FP8="$DUAL_MAX_MODEL_LEN"
  MODEL_LEN_BF16="$DUAL_MAX_MODEL_LEN"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[run_phi4_reasoning_endpoints_docker] docker not found" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[run_phi4_reasoning_endpoints_docker] cannot access docker daemon" >&2
  echo "Enable access (example): sudo usermod -aG docker $USER && newgrp docker" >&2
  exit 1
fi

mkdir -p "$HF_CACHE_DIR"

docker rm -f phi4-reasoning-fp8 phi4-reasoning-bf16 >/dev/null 2>&1 || true

compute_lines="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^\s*$/d' || true)"
if [[ -n "$compute_lines" ]]; then
  echo "[run_phi4_reasoning_endpoints_docker] active GPU compute processes detected:" >&2
  echo "$compute_lines" >&2
  echo "[run_phi4_reasoning_endpoints_docker] stop non-essential workloads before launch, then retry" >&2
  exit 1
fi

mem_line="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
if [[ -n "$mem_line" ]] && [[ "$mem_line" =~ ^[0-9]+$ ]]; then
  required_mb="$REQUIRE_FREE_MB_FP8"
  if [[ "$START_BF16" == "1" ]]; then
    required_mb="$((REQUIRE_FREE_MB_FP8 + REQUIRE_FREE_MB_BF16))"
  fi
  if (( mem_line < required_mb )); then
    echo "[run_phi4_reasoning_endpoints_docker] insufficient free GPU memory: ${mem_line} MiB < required ${required_mb} MiB" >&2
    echo "[run_phi4_reasoning_endpoints_docker] stop non-essential workloads or lower REQUIRE_FREE_MB_* thresholds" >&2
    exit 1
  fi
fi

wait_for_health() {
  local name="$1"
  local url="$2"
  local started_at now elapsed
  started_at="$(date +%s)"

  while true; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
      echo "[run_phi4_reasoning_endpoints_docker] ${name} exited before becoming healthy" >&2
      docker logs --tail 120 "$name" >&2 || true
      return 1
    fi

    if curl -fsS "$url/health" >/dev/null 2>&1; then
      echo "[run_phi4_reasoning_endpoints_docker] ${name} is healthy at ${url}/health"
      return 0
    fi

    now="$(date +%s)"
    elapsed=$((now - started_at))
    if (( elapsed >= HEALTH_TIMEOUT_SEC )); then
      echo "[run_phi4_reasoning_endpoints_docker] timeout waiting for ${name} health (${HEALTH_TIMEOUT_SEC}s)" >&2
      docker logs --tail 120 "$name" >&2 || true
      return 1
    fi

    sleep "$HEALTH_POLL_SEC"
  done
}

echo "[run_phi4_reasoning_endpoints_docker] starting FP8 container on :$HOST_PORT_FP8"
docker run -d \
  --name phi4-reasoning-fp8 \
  --gpus all \
  --ipc host \
  --network host \
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -e VLLM_DISABLED_KERNELS="$FP8_DISABLED_KERNELS" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  "$IMAGE" \
  "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$HOST_PORT_FP8" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MODEL_LEN_FP8" \
    --quantization fp8 \
    $EAGER_FLAG \
    $EXTRA_ARGS >/dev/null

if [[ "$START_BF16" == "1" ]]; then
  if [[ "$WAIT_FOR_HEALTH" == "1" ]]; then
    echo "[run_phi4_reasoning_endpoints_docker] waiting for FP8 before starting BF16"
    wait_for_health "phi4-reasoning-fp8" "http://127.0.0.1:$HOST_PORT_FP8"
  fi

  echo "[run_phi4_reasoning_endpoints_docker] starting BF16 container on :$HOST_PORT_BF16"
  docker run -d \
    --name phi4-reasoning-bf16 \
    --gpus all \
    --ipc host \
    --network host \
    -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    "$IMAGE" \
    "$MODEL_ID" \
      --host 0.0.0.0 \
      --port "$HOST_PORT_BF16" \
      --tensor-parallel-size "$TP_SIZE" \
      --max-model-len "$MODEL_LEN_BF16" \
      --dtype bfloat16 \
      $EAGER_FLAG \
      $EXTRA_ARGS >/dev/null
fi

echo "[run_phi4_reasoning_endpoints_docker] started"
echo "  fp8  -> http://127.0.0.1:$HOST_PORT_FP8"
echo "  fp8 max_model_len=$MODEL_LEN_FP8"
if [[ "$START_BF16" == "1" ]]; then
  echo "  bf16 -> http://127.0.0.1:$HOST_PORT_BF16"
  echo "  bf16 max_model_len=$MODEL_LEN_BF16"
else
  echo "  bf16 -> disabled (set START_BF16=1 to enable)"
fi
echo "Use: docker logs -f phi4-reasoning-fp8"
echo "Use: docker logs -f phi4-reasoning-bf16"

if [[ "$WAIT_FOR_HEALTH" == "1" ]]; then
  echo "[run_phi4_reasoning_endpoints_docker] waiting for endpoint connectivity"
  if [[ "$START_BF16" != "1" ]]; then
    wait_for_health "phi4-reasoning-fp8" "http://127.0.0.1:$HOST_PORT_FP8"
  fi
  if [[ "$START_BF16" == "1" ]]; then
    wait_for_health "phi4-reasoning-bf16" "http://127.0.0.1:$HOST_PORT_BF16"
  fi
fi
