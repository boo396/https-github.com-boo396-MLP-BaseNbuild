#!/usr/bin/env bash
set -euo pipefail

# Optional cleanup script for aggressive dependency grind iterations.
# Use only on dedicated/dev boxes.

echo "This will remove apt caches, pip caches, and local build artifacts."
read -r -p "Continue? [y/N] " ans
if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
  echo "Aborted"
  exit 0
fi

sudo apt-get clean
rm -rf ~/.cache/pip ~/.cache/pypoetry ~/.cache/uv || true
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f \( -name '*.onnx' -o -name '*.engine' -o -name '*.pth' \) -delete

echo "Cleanup complete"
