#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

./scripts/check_native_stack.sh

python -m mlp_basenbuild.server --config configs/config.arm.yaml &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 2
curl -sS -m 10 http://127.0.0.1:8090/health | jq .

curl -sS -m 10 -X POST http://127.0.0.1:8090/route \
  -H 'Content-Type: application/json' \
  -d '{"text":"prove this theorem step by step","has_image":false}' | jq .

echo "Smoke test passed"
