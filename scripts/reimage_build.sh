#!/usr/bin/env bash
set -euo pipefail

# One-command post-reimage build path for GB10.
# Usage:
#   ./scripts/reimage_build.sh
#   ./scripts/reimage_build.sh --strict

STRICT_FLAG=""
if [[ "${1:-}" == "--strict" ]]; then
  STRICT_FLAG="--strict"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[reimage-build] Step 1/3: Provisioning host + env"
sudo ./scripts/provision_gb10_native.sh ${STRICT_FLAG}

echo "[reimage-build] Step 2/3: Exporting ONNX/TensorRT engine"
./scripts/export_engine.sh || {
  echo "[reimage-build] Engine export failed. If TensorRT is unavailable, install native stack and retry."
  exit 10
}

echo "[reimage-build] Step 3/3: Running smoke test"
./scripts/post_reimage_smoke.sh

echo "[reimage-build] Completed successfully"
