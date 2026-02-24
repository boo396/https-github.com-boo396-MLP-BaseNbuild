#!/usr/bin/env bash
set -euo pipefail

echo "arch: $(uname -m)"
echo "trtexec: $(command -v trtexec || echo missing)"

dpkg -l | grep -E 'nvinfer|tensorrt|libnvonnxparsers' || true

python3 - <<'PY'
import importlib.util
print('python_tensorrt', bool(importlib.util.find_spec('tensorrt')))
print('python_cuda', bool(importlib.util.find_spec('cuda')))
PY
