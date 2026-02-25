#!/usr/bin/env bash
set -euo pipefail

# GB10 ARM post-reimage provisioning for native TensorRT/ONNX path.
# This script is intentionally host-native (minimal abstraction).
# TensorRT install mode:
#   TRT_INSTALL_METHOD=apt (default): install from configured apt repos
#   TRT_INSTALL_METHOD=pip: install Python TensorRT wheel in venv (no apt TensorRT required)

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

TRT_INSTALL_METHOD="${TRT_INSTALL_METHOD:-apt}"
if [[ "$TRT_INSTALL_METHOD" != "apt" && "$TRT_INSTALL_METHOD" != "pip" ]]; then
  log "Unsupported TRT_INSTALL_METHOD='${TRT_INSTALL_METHOD}'. Use 'apt' or 'pip'."
  exit 2
fi

if [[ "$TRT_INSTALL_METHOD" == "apt" ]]; then
  log "TensorRT install method: apt"
  log "Checking availability of TensorRT packages from apt repos"
  if apt-cache policy tensorrt | awk '/Candidate:/{print $2}' | grep -qv '(none)'; then
    log "Installing TensorRT runtime/tooling from apt"
    apt-get install -y --no-install-recommends \
      tensorrt \
      libnvinfer-bin \
      libnvinfer-dev \
      libnvonnxparsers-dev \
      libnvinfer-plugin-dev \
      python3-libnvinfer
  else
    log "TensorRT apt package candidate not found."
    log "Ensure NVIDIA JetPack/L4T apt repos are configured, then re-run this script."
    if [[ "$STRICT_MODE" == "1" ]]; then
      log "Strict mode enabled: failing because TensorRT packages are unavailable."
      exit 2
    fi
  fi
else
  log "TensorRT install method: pip"
  log "Skipping apt TensorRT package install"
fi

log "Ensuring Python virtualenv exists"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# GB10-safe PyTorch install path.
# Override defaults if your host uses a different CUDA/PyTorch build.
TORCH_VERSION="${TORCH_VERSION:-2.9.1+cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_GB10_TORCH="${INSTALL_GB10_TORCH:-1}"

if [[ "$INSTALL_GB10_TORCH" == "1" ]]; then
  log "Installing PyTorch (${TORCH_VERSION}) from ${TORCH_INDEX_URL}"
  if ! pip install --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}"; then
    log "GB10 torch install failed; falling back to default resolver (torch>=2.3)"
    pip install "torch>=2.3"
  fi
fi

pip install -e .

# Best-effort CUDA Python bindings for native stack
# May not be available on all ARM images; keep non-fatal.
pip install --no-cache-dir cuda-python || true

if [[ "$TRT_INSTALL_METHOD" == "pip" ]]; then
  # TensorRT Python bindings from pip path only (avoid mixing apt + pip TensorRT installs).
  pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt
fi

log "Provisioning complete. Running native stack check"
./scripts/check_native_stack.sh || true

if [[ "$STRICT_MODE" == "1" ]]; then
  if [[ "$TRT_INSTALL_METHOD" == "apt" ]]; then
    if ! command -v trtexec >/dev/null 2>&1; then
      log "Strict mode enabled: trtexec not found after apt provisioning."
      exit 3
    fi
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
