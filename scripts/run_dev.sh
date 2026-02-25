#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

CONFIG_PATH="${CONFIG_PATH:-configs/config.arm.yaml}"
AUTO_CPU_FALLBACK="${AUTO_CPU_FALLBACK:-1}"
CPU_FALLBACK_CONFIG="${CPU_FALLBACK_CONFIG:-/tmp/config.router.cpu.yaml}"

launch_server() {
	local cfg_path="$1"
	python -m mlp_basenbuild.server --config "$cfg_path"
}

print_banner() {
	local mode="$1"
	local cfg_path="$2"
	echo "[run_dev] router_start mode=${mode} config=${cfg_path} auto_cpu_fallback=${AUTO_CPU_FALLBACK}"
}

generate_cpu_config() {
	local src_cfg="$1"
	local out_cfg="$2"
	python3 - "$src_cfg" "$out_cfg" <<'PY'
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
out = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text())
cfg.setdefault("mlp", {})
cfg["mlp"]["device"] = "cpu"
cfg["mlp"]["compile"] = False
cfg["mlp"]["backend"] = "pytorch"
cfg["mlp"]["use_fp16_cuda"] = False
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY
}

if [[ "$AUTO_CPU_FALLBACK" != "1" ]]; then
	print_banner "CONFIG_ONLY" "$CONFIG_PATH"
	launch_server "$CONFIG_PATH"
	exit $?
fi

ERR_LOG="$(mktemp /tmp/run_dev_router_err.XXXXXX.log)"
trap 'rm -f "$ERR_LOG"' EXIT

print_banner "PRIMARY" "$CONFIG_PATH"
if launch_server "$CONFIG_PATH" 2> >(tee "$ERR_LOG" >&2); then
	exit 0
fi

if grep -qiE 'cuda error: out of memory|torch\.AcceleratorError: CUDA error: out of memory' "$ERR_LOG"; then
	echo "[run_dev] detected CUDA OOM during router startup; falling back to CPU config"
	generate_cpu_config "$CONFIG_PATH" "$CPU_FALLBACK_CONFIG" >/dev/null
	print_banner "CPU_FALLBACK" "$CPU_FALLBACK_CONFIG"
	launch_server "$CPU_FALLBACK_CONFIG"
	exit $?
fi

echo "[run_dev] startup failed (non-OOM); not applying CPU fallback" >&2
exit 1
