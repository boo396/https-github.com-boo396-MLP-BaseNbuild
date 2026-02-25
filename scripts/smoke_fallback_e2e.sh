#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

cleanup_existing() {
  pkill -f "mlp_basenbuild.server --config configs/config.arm.yaml" 2>/dev/null || true
  pkill -f "mlp_basenbuild.expert_worker" 2>/dev/null || true
  pkill -f "mlp_basenbuild.mock_trtllm_server" 2>/dev/null || true
  pkill -f "scripts/run_workers_dev.sh" 2>/dev/null || true
  pkill -f "scripts/run_mock_trtllm.sh" 2>/dev/null || true
}

wait_for_health() {
  local url="$1"
  local retries="${2:-60}"
  local delay_s="${3:-0.5}"

  for _ in $(seq 1 "$retries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay_s"
  done

  echo "[smoke_fallback_e2e] timeout waiting for $url" >&2
  return 1
}

cleanup() {
  set +e
  if [[ -n "${PID_ROUTER:-}" ]]; then kill "$PID_ROUTER" 2>/dev/null || true; fi
  if [[ -n "${PID_WORKERS:-}" ]]; then kill "$PID_WORKERS" 2>/dev/null || true; fi
  if [[ -n "${PID_MOCK:-}" ]]; then kill "$PID_MOCK" 2>/dev/null || true; fi
  cleanup_existing
}
trap cleanup EXIT INT TERM

cleanup_existing

./scripts/run_mock_trtllm.sh >/tmp/mock_trtllm.log 2>&1 &
PID_MOCK=$!

./scripts/run_workers_dev.sh >/tmp/workers_dev.log 2>&1 &
PID_WORKERS=$!

python -m mlp_basenbuild.server --config configs/config.arm.yaml >/tmp/router_dev.log 2>&1 &
PID_ROUTER=$!

wait_for_health "http://127.0.0.1:8301/health"
wait_for_health "http://127.0.0.1:8302/health"
wait_for_health "http://127.0.0.1:8103/health"
wait_for_health "http://127.0.0.1:8090/health"

python - <<'PY'
from __future__ import annotations

import json
import urllib.request


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


route_payload = {
    "text": "prove this theorem step by step",
    "has_image": False,
}
route_response = post_json("http://127.0.0.1:8090/route", route_payload)

route_result = (((route_response.get("worker_response") or {}).get("details") or {}).get("result") or {})
if route_result.get("used_precision") != "bf16" or route_result.get("fallback_used") is not True:
    raise SystemExit(f"FAIL: expected BF16 fallback via /route, got: {route_result}")

infer_payload = {
    "text": "summarize this paragraph",
    "has_image": False,
    "route": {"dispatch_target": "frontier_reasoning", "dispatch_backend": "trtllm"},
}
infer_response = post_json("http://127.0.0.1:8103/infer", infer_payload)

infer_result = (((infer_response.get("details") or {}).get("result") or {}))
if infer_result.get("used_precision") != "fp8" or infer_result.get("fallback_used") is not False:
    raise SystemExit(f"FAIL: expected FP8 primary via /infer, got: {infer_result}")

print("PASS: route fallback=bf16 and direct infer=fp8")
PY

echo "[smoke_fallback_e2e] success"
