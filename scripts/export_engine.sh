#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python -m mlp_basenbuild.export_engine --config configs/config.arm.yaml --build-engine
