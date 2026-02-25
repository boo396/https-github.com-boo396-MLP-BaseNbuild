#!/usr/bin/env bash
set -euo pipefail

FP8_URL="${FP8_URL:-http://127.0.0.1:8301}"
BF16_URL="${BF16_URL:-http://127.0.0.1:8302}"
MODEL_NAME="${MODEL_NAME:-microsoft/Phi-4-reasoning-plus}"
CHECK_BF16="${CHECK_BF16:-0}"

payload='{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"prove this theorem step by step"}],"max_tokens":128,"temperature":0.0}'

echo "[check_reasoning_endpoints] FP8 health"
curl -fsS "$FP8_URL/health" | head -c 300; echo

if [[ "$CHECK_BF16" == "1" ]]; then
	echo "[check_reasoning_endpoints] BF16 health"
	curl -fsS "$BF16_URL/health" | head -c 300; echo
fi

echo "[check_reasoning_endpoints] FP8 inference"
curl -fsS -X POST "$FP8_URL/v1/chat/completions" -H 'content-type: application/json' -d "$payload" | head -c 500; echo

if [[ "$CHECK_BF16" == "1" ]]; then
	echo "[check_reasoning_endpoints] BF16 inference"
	curl -fsS -X POST "$BF16_URL/v1/chat/completions" -H 'content-type: application/json' -d "$payload" | head -c 500; echo
fi

echo "[check_reasoning_endpoints] done"
