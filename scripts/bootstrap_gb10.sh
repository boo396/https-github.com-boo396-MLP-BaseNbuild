#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential curl git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# GB10-safe PyTorch install path.
# Override defaults if your host uses a different CUDA/PyTorch build.
TORCH_VERSION="${TORCH_VERSION:-2.9.1+cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_GB10_TORCH="${INSTALL_GB10_TORCH:-1}"

if [[ "$INSTALL_GB10_TORCH" == "1" ]]; then
	echo "Installing PyTorch (${TORCH_VERSION}) from ${TORCH_INDEX_URL}"
	if ! pip install --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}"; then
		echo "GB10 torch install failed; falling back to default resolver (torch>=2.3)"
		pip install "torch>=2.3"
	fi
fi

pip install -e .

echo "Bootstrap complete. Run: source .venv/bin/activate && python -m mlp_basenbuild.server --config configs/config.arm.yaml"
