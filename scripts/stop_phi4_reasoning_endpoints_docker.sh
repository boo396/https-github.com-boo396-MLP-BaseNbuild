#!/usr/bin/env bash
set -euo pipefail

docker rm -f phi4-reasoning-fp8 phi4-reasoning-bf16 >/dev/null 2>&1 || true
echo "[stop_phi4_reasoning_endpoints_docker] stopped"
