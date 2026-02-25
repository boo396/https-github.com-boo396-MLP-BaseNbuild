#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python -m mlp_basenbuild.mock_trtllm_server --precision fp8 --port 8301 --fail-mode reasoning-only &
PID_FP8=$!
python -m mlp_basenbuild.mock_trtllm_server --precision bf16 --port 8302 --fail-mode never &
PID_BF16=$!

cleanup() {
  kill "$PID_FP8" "$PID_BF16" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait
