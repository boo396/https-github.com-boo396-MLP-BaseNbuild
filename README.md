# MLP BaseNbuild

GB10-first (ARM SoC) fast routing baseline focused on **speed over parameter count** and **minimal abstraction**.

## What this repo is
- In-process router service (no external embedding microservice)
- Lightweight MLP scorer with CPU shortcut gate
- ARM-native setup path for rebuildable systems
- Benchmark harness for p50/p95 and throughput

## Quick start
1. Install deps:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -e .`
2. Run server:
   - `python -m mlp_basenbuild.server --config configs/config.arm.yaml`
3. Run benchmark:
   - `python -m mlp_basenbuild.benchmark --url http://127.0.0.1:8090/route --requests 200 --concurrency 8`

## Phase 2: Native TensorRT path (GB10 ARM)
### Reimage + Provision (recommended)
1. Post-reimage provisioning:
   - `sudo ./scripts/provision_gb10_native.sh`
   - strict mode (required TensorRT): `sudo ./scripts/provision_gb10_native.sh --strict`
   - env variant: `STRICT_TRT=1 sudo ./scripts/provision_gb10_native.sh`
2. Validate host/native stack:
   - `./scripts/check_native_stack.sh`
3. Build ONNX + TensorRT engine:
   - `./scripts/export_engine.sh`
4. Enable TensorRT backend:
   - set `mlp.backend: tensorrt` in `configs/config.arm.yaml`
5. Run smoke test:
   - `./scripts/post_reimage_smoke.sh`

### Manual fallback flow
1. Check native stack:
   - `./scripts/check_native_stack.sh`
2. Export ONNX + build engine (uses host `trtexec` if present, else TensorRT Python API):
   - `./scripts/export_engine.sh`
3. Switch backend in `configs/config.arm.yaml`:
   - `mlp.backend: tensorrt`
4. Restart server and verify:
   - `curl http://127.0.0.1:8090/health`
   - Health returns `backend: tensorrt` when active, otherwise fallback is `pytorch`.

## GB10 performance intent
- Keep routing in one process
- Preload model once at startup
- Use CPU regex/keyword fast-path before GPU inference
- Optional `torch.compile` and fp16 autocast when CUDA available

## Next upgrades
- Replace hash embedding with fused tokenizer/encoder tuned for GB10
- Export MLP to ONNX/TensorRT using host-native NVIDIA ARM stack
- Pin NUMA/CPU affinity and tune micro-batch windows

## Cleanup for rebuild cycles
- Optional aggressive cleanup script:
   - `./scripts/reformat_cleanup.sh`
