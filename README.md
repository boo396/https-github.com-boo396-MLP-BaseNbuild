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
   - multimodal worker deps: `pip install -e .[multimodal]`
   - GB10 PyTorch override knobs (used by bootstrap/provision scripts):
     - `TORCH_VERSION` (default: `2.9.1+cu128`)
     - `TORCH_INDEX_URL` (default: `https://download.pytorch.org/whl/cu128`)
     - `INSTALL_GB10_TORCH=0` to skip script-managed torch install
2. Run server:
   - `python -m mlp_basenbuild.server --config configs/config.arm.yaml`
   - optional local workers (new dispatch path): `./scripts/run_workers_dev.sh`
3. Run benchmark:
   - `python -m mlp_basenbuild.benchmark --url http://127.0.0.1:8090/route --requests 200 --concurrency 8`

## Routing + worker dispatch (in progress)
- `routing.top_k` controls top-k candidate tracking from router logits.
- `routing.enable_worker_dispatch` toggles dispatch to local worker endpoints.
- `experts.worker_endpoints` maps standard expert models (INT4 path target).
- `deep_thinking.worker_endpoint` maps reasoning/deep-thinking expert (FP8/BF16 path target).
- `vision_path` worker now loads `microsoft/Phi-4-multimodal-instruct-onnx` through `onnxruntime-genai` from local HF cache.
- `frontier_reasoning` worker uses TensorRT-LLM OpenAI-compatible endpoints: FP8 first (`trtllm_fp8_url`), BF16 fallback (`trtllm_bf16_url`).

### Multimodal request example
- Route with image URL:
   - `curl -s -X POST http://127.0.0.1:8090/route -H 'content-type: application/json' -d '{"text":"Describe this image in detail","image_url":"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"}'`
- Route with local image path:
   - `curl -s -X POST http://127.0.0.1:8090/route -H 'content-type: application/json' -d '{"text":"What objects are present?","image_path":"/tmp/sample.png"}'`

### Deep-thinking backend expectation
- Run TensorRT-LLM serving endpoints separately (example):
   - FP8 endpoint at `http://127.0.0.1:8301/v1/chat/completions`
   - BF16 fallback endpoint at `http://127.0.0.1:8302/v1/chat/completions`
- The `frontier_reasoning` worker at `:8103` proxies requests and automatically falls back to BF16 on FP8 failure.

### Real Phi-4 reasoning endpoints (FP8 + BF16)
- Start real endpoints (OpenAI-compatible) for `microsoft/Phi-4-reasoning-plus`:
   - `./scripts/run_phi4_reasoning_endpoints.sh`
- Verify health + inference on both endpoints:
   - `./scripts/check_reasoning_endpoints.sh`
- Then start workers (uses `http://127.0.0.1:8301` and `http://127.0.0.1:8302` by default):
   - `./scripts/run_workers_dev.sh`
- Optional env overrides for worker endpoint targets:
   - `TRTLLM_FP8_URL=http://127.0.0.1:8301 TRTLLM_BF16_URL=http://127.0.0.1:8302 ./scripts/run_workers_dev.sh`

### Router startup fallback
- `./scripts/run_dev.sh` now auto-falls back to CPU mode if router startup fails with CUDA OOM.
- CPU fallback writes a temporary config at `/tmp/config.router.cpu.yaml` and relaunches automatically.
- Startup now prints a mode banner (`PRIMARY`, `CPU_FALLBACK`, or `CONFIG_ONLY`) with the active config path.
- Controls:
   - `AUTO_CPU_FALLBACK=0 ./scripts/run_dev.sh` to disable fallback
   - `CPU_FALLBACK_CONFIG=/tmp/custom.router.cpu.yaml ./scripts/run_dev.sh` to override fallback path

### Feasibility note: native Option B on GB10
- GPU-enabled torch can be installed natively (`torch==2.9.1+cu128`), but native vLLM startup on GB10 currently fails in this environment with Triton/PTX codegen:
   - `ptxas fatal : Value 'sm_121a' is not defined for option 'gpu-name'`
- This indicates a toolchain compatibility gap for this host-native stack.
- Workaround from Triton issue #9181: set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` so Triton uses system `ptxas` (newer Blackwell support).

### Fallback Option A: Dockerized real endpoints
- Start Docker-based endpoint(s) with FP8 as default:
   - `./scripts/run_phi4_reasoning_endpoints_docker.sh`
- Launcher now waits for `/health` connectivity before returning success (default behavior).
- Connectivity wait controls:
   - `WAIT_FOR_HEALTH=0` to skip waiting
   - `HEALTH_TIMEOUT_SEC=...` to change max wait (default `900`)
   - `HEALTH_POLL_SEC=...` to change polling interval (default `5`)
- Default behavior is FP8-only to preserve GPU memory headroom; enable BF16 companion with:
   - `START_BF16=1 ./scripts/run_phi4_reasoning_endpoints_docker.sh`
- In dual mode the launcher uses a safer default context length (`DUAL_MAX_MODEL_LEN=8192`) and starts BF16 only after FP8 health is up, to reduce shared-memory startup contention.
- Launcher now performs preflight checks and exits early if non-essential GPU compute workloads are active.
- Docker launcher defaults to `--enforce-eager` to avoid Triton autotune PTX issues on GB10.
- Docker launcher defaults to `vllm/vllm-openai:nightly` and sets `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel` for the FP8 container.
- This forces a stable FP8 fallback kernel path on GB10 (`ChannelWiseTorchFP8ScaledMMLinearKernel`) and avoids `cutlass_scaled_mm` internal runtime errors.
- To disable eager mode explicitly: `ENFORCE_EAGER=0 ./scripts/run_phi4_reasoning_endpoints_docker.sh`
- To override FP8 kernel disable list: `FP8_DISABLED_KERNELS=... ./scripts/run_phi4_reasoning_endpoints_docker.sh`
- To override image tag: `IMAGE=vllm/vllm-openai:<tag> ./scripts/run_phi4_reasoning_endpoints_docker.sh`
- To adjust memory guards: `REQUIRE_FREE_MB_FP8=... REQUIRE_FREE_MB_BF16=... ./scripts/run_phi4_reasoning_endpoints_docker.sh`
- Stop Docker endpoints:
   - `./scripts/stop_phi4_reasoning_endpoints_docker.sh`
- Validate FP8 endpoint:
   - `./scripts/check_reasoning_endpoints.sh`
- Validate both endpoints when BF16 is enabled:
   - `CHECK_BF16=1 ./scripts/check_reasoning_endpoints.sh`
- Then run workers against default URLs (`:8301` and `:8302`):
   - `./scripts/run_workers_dev.sh`

### Local mock TRT-LLM (fallback testing)
- Start mock FP8/BF16 backends:
   - `./scripts/run_mock_trtllm.sh`
- Mock defaults:
   - FP8 (`:8301`) uses `reasoning-only` fail mode (simulates FP8 failure on hard prompts)
   - BF16 (`:8302`) always succeeds
- With workers + router running, test fallback path:
   - `curl -s -X POST http://127.0.0.1:8090/route -H 'content-type: application/json' -d '{"text":"prove this theorem step by step","has_image":false}'`
   - Expect `worker_response.details.result.used_precision` to be `bf16` and `fallback_used` to be `true`.

### One-command e2e fallback smoke test
- Starts mocks + workers + router, then validates:
   - `/route` reasoning request falls back from FP8 to BF16
   - direct `/infer` non-reasoning request stays on FP8
- Run:
   - `./scripts/smoke_fallback_e2e.sh`

## Phase 2: Native TensorRT path (GB10 ARM)
### Reimage + Provision (recommended)
0. One-command path:
   - `./scripts/reimage_build.sh`
   - strict: `./scripts/reimage_build.sh --strict`
1. Post-reimage provisioning:
   - `sudo ./scripts/provision_gb10_native.sh`
   - strict mode (required TensorRT): `sudo ./scripts/provision_gb10_native.sh --strict`
   - env variant: `STRICT_TRT=1 sudo ./scripts/provision_gb10_native.sh`
   - optional custom PyTorch wheel source:
     - `TORCH_VERSION=2.9.1+cu128 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 sudo ./scripts/provision_gb10_native.sh`
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
