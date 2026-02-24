#!/usr/bin/env bash
set -euo pipefail

# GB10 ARM post-reimage provisioning for native TensorRT/ONNX path.
# This script is intentionally host-native (minimal abstraction).

STRICT_MODE=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT_MODE=1
fi
if [[ "${STRICT_TRT:-0}" == "1" ]]; then
  STRICT_MODE=1
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo ./scripts/provision_gb10_native.sh"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

log() { echo "[provision] $*"; }

log "Updating apt metadata"
apt-get update

log "Installing baseline build/tooling packages"
apt-get install -y --no-install-recommends \
  software-properties-common \
  apt-transport-https \
  ca-certificates \
  gnupg \
  curl \
  wget \
  git \
  build-essential \
  pkg-config \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  jq \
  htop \
  numactl

# NVIDIA package install strategy:
# 1) If TensorRT packages are already available from configured repos, install directly.
# 2) If not available, fail with a clear message so operator can enable JetPack/L4T repos.

log "Checking availability of TensorRT packages from apt repos"
if apt-cache policy tensorrt | grep -q Candidate; then
  log "Installing TensorRT runtime/tooling from apt"
  apt-get install -y --no-install-recommends \
    tensorrt \
    libnvinfer-bin \
    libnvinfer-dev \
    libnvonnxparsers-dev \
    libnvinfer-plugin-dev
else
  log "TensorRT apt package candidate not found."
  log "Ensure NVIDIA JetPack/L4T apt repos are configured, then re-run this script."
  if [[ "$STRICT_MODE" == "1" ]]; then
    log "Strict mode enabled: failing because TensorRT packages are unavailable."
    exit 2
  fi
fi

log "Ensuring Python virtualenv exists"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .

# Best-effort Python bindings for native stack
# These may not be available on all ARM images; keep non-fatal.
pip install --no-cache-dir cuda-python || true
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt || true

log "Provisioning complete. Running native stack check"
./scripts/check_native_stack.sh || true

if [[ "$STRICT_MODE" == "1" ]]; then
  if ! command -v trtexec >/dev/null 2>&1; then
    log "Strict mode enabled: trtexec not found after provisioning."
    exit 3
  fi
  if ! python3 - <<'PY'
import importlib.util
ok = bool(importlib.util.find_spec('tensorrt')) and bool(importlib.util.find_spec('cuda'))
raise SystemExit(0 if ok else 1)
PY
  then
    log "Strict mode enabled: Python TensorRT/cuda modules are unavailable after provisioning."
    exit 4
  fi
fi

echo
echo "Next steps:"
echo "  1) ./scripts/export_engine.sh"
echo "  2) set mlp.backend: tensorrt in configs/config.arm.yaml"
echo "  3) ./scripts/run_dev.sh"
