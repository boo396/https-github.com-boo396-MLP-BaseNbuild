#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python -m mlp_basenbuild.server --config configs/config.arm.yaml
