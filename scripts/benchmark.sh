#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python -m mlp_basenbuild.benchmark --url http://127.0.0.1:8090/route --requests 200 --concurrency 8
