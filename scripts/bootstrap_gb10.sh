#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential curl git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

echo "Bootstrap complete. Run: source .venv/bin/activate && python -m mlp_basenbuild.server --config configs/config.arm.yaml"
