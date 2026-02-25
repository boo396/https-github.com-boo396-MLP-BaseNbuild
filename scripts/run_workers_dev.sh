#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

TRTLLM_MODEL_NAME="${TRTLLM_MODEL_NAME:-microsoft/Phi-4-reasoning-plus}"
TRTLLM_FP8_URL="${TRTLLM_FP8_URL:-http://127.0.0.1:8301}"
TRTLLM_BF16_URL="${TRTLLM_BF16_URL:-http://127.0.0.1:8302}"
TRTLLM_TIMEOUT_S="${TRTLLM_TIMEOUT_S:-60}"
TRTLLM_MAX_NEW_TOKENS="${TRTLLM_MAX_NEW_TOKENS:-512}"
TRTLLM_TEMPERATURE="${TRTLLM_TEMPERATURE:-0.0}"

python -m mlp_basenbuild.expert_worker --name small_text --runtime onnxruntime_int4 --precision int4 --port 8101 &
PID_SMALL=$!

if python - <<'PY' >/dev/null 2>&1
import onnxruntime_genai
PY
then
  python -m mlp_basenbuild.expert_worker --name vision_path --runtime onnxruntime_int4 --precision int4 --model-id microsoft/Phi-4-multimodal-instruct-onnx --model-subdir gpu/gpu-int4-rtn-block-32 --cache-dir ~/.cache/huggingface --max-new-tokens 256 --temperature 0.0 --port 8102 &
else
  echo "[run_workers_dev] onnxruntime_genai missing; starting stub vision worker"
  python -m mlp_basenbuild.expert_worker --name vision_path --runtime stub --precision int4 --port 8102 &
fi

PID_VISION=$!
python -m mlp_basenbuild.expert_worker --name frontier_reasoning --runtime trtllm --precision fp8 --trtllm-model-name "$TRTLLM_MODEL_NAME" --trtllm-fp8-url "$TRTLLM_FP8_URL" --trtllm-bf16-url "$TRTLLM_BF16_URL" --trtllm-timeout-s "$TRTLLM_TIMEOUT_S" --max-new-tokens "$TRTLLM_MAX_NEW_TOKENS" --temperature "$TRTLLM_TEMPERATURE" --port 8103 &
PID_REASON=$!

cleanup() {
  kill "$PID_SMALL" "$PID_VISION" "$PID_REASON" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait
