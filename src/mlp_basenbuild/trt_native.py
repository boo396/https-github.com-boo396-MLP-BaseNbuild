from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def _normalize_precision(precision: str) -> str:
    normalized = precision.strip().lower()
    if normalized in {"fp32", "float", "float32"}:
        return "fp32"
    if normalized in {"fp16", "half", "float16"}:
        return "fp16"
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    if normalized in {"fp8", "float8"}:
        return "fp8"
    raise ValueError(f"Unsupported TensorRT precision: {precision}")


def detect_tensorrt_environment() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "trtexec": shutil.which("trtexec"),
        "python_tensorrt": False,
        "python_cuda": False,
    }
    try:
        import tensorrt as _trt  # noqa: F401
        status["python_tensorrt"] = True
    except Exception:
        pass

    try:
        from cuda import cudart as _cudart  # noqa: F401
        status["python_cuda"] = True
    except Exception:
        try:
            from cuda.bindings import runtime as _cudart  # noqa: F401
            status["python_cuda"] = True
        except Exception:
            pass

    status["ready"] = bool(status["trtexec"] or (status["python_tensorrt"] and status["python_cuda"]))
    return status


def export_router_to_onnx(model: torch.nn.Module, input_dim: int, onnx_path: str, opset: int = 18) -> str:
    target = Path(onnx_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(target),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        external_data=False,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return str(target)


def _build_engine_python(onnx_path: str, engine_path: str, precision: str, workspace_mb: int) -> str:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed: " + " | ".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * 1024 * 1024)

    precision = _normalize_precision(precision)
    if precision == "fp16":
        if not getattr(builder, "platform_has_fast_fp16", False):
            raise RuntimeError("Requested FP16, but platform_has_fast_fp16 is false")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        if not hasattr(trt.BuilderFlag, "BF16"):
            raise RuntimeError("Requested BF16, but TensorRT build does not expose BuilderFlag.BF16")
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision == "fp8":
        if not hasattr(trt.BuilderFlag, "FP8"):
            raise RuntimeError("Requested FP8, but TensorRT build does not expose BuilderFlag.FP8")
        has_fast_fp8 = getattr(builder, "platform_has_fast_fp8", True)
        if not has_fast_fp8:
            raise RuntimeError("Requested FP8, but platform_has_fast_fp8 is false")
        config.set_flag(trt.BuilderFlag.FP8)

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = tuple(input_tensor.shape)
    input_dim = int(input_shape[-1] if len(input_shape) > 1 and input_shape[-1] > 0 else 1024)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, input_dim), (8, input_dim), (32, input_dim))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT serialized engine build failed")

    target = Path(engine_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(serialized)
    return str(target)


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    workspace_mb: int = 2048,
    trtexec_bin: str = "trtexec",
) -> str:
    precision = _normalize_precision(precision)
    trtexec_path = shutil.which(trtexec_bin)
    if trtexec_path:
        cmd = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--minShapes=input:1x1024",
            "--optShapes=input:8x1024",
            "--maxShapes=input:32x1024",
            f"--workspace={int(workspace_mb)}",
        ]
        if precision == "fp16":
            cmd.append("--fp16")
        elif precision == "bf16":
            cmd.append("--bf16")
        elif precision == "fp8":
            cmd.append("--fp8")
        subprocess.run(cmd, check=True)
        return str(engine_path)

    return _build_engine_python(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
        workspace_mb=workspace_mb,
    )


def _cuda_ok(result: Tuple[int, ...], op_name: str) -> Tuple[Any, ...]:
    code = int(result[0])
    if code != 0:
        raise RuntimeError(f"CUDA failure for {op_name}: code={code}")
    return result[1:]


class TensorRTRuntime:
    def __init__(self, engine_path: str):
        import tensorrt as trt
        try:
            from cuda import cudart  # type: ignore
        except Exception:
            from cuda.bindings import runtime as cudart  # type: ignore

        self._cudart = cudart
        self._logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self._logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")

        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self._is_legacy_bindings = hasattr(engine, "num_bindings") and hasattr(engine, "get_binding_name")
        if self._is_legacy_bindings:
            names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
            self.input_idx = engine.get_binding_index("input") if "input" in names else 0
            self.output_idx = engine.get_binding_index("output") if "output" in names else 1
            self.input_name = names[self.input_idx]
            self.output_name = names[self.output_idx]
        else:
            input_names = []
            output_names = []
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    input_names.append(name)
                elif mode == trt.TensorIOMode.OUTPUT:
                    output_names.append(name)
            if not input_names or not output_names:
                raise RuntimeError("Failed to resolve TensorRT input/output tensors")
            self.input_name = "input" if "input" in input_names else input_names[0]
            self.output_name = "output" if "output" in output_names else output_names[0]

    def infer(self, batch_embeddings: np.ndarray) -> np.ndarray:
        if batch_embeddings.ndim != 2:
            raise ValueError(f"Expected [batch, dim], got {batch_embeddings.shape}")

        host_in = np.ascontiguousarray(batch_embeddings.astype(np.float32))
        batch, dim = int(host_in.shape[0]), int(host_in.shape[1])

        if self._is_legacy_bindings:
            self.context.set_binding_shape(self.input_idx, (batch, dim))
            out_shape = tuple(self.context.get_binding_shape(self.output_idx))
        else:
            self.context.set_input_shape(self.input_name, (batch, dim))
            out_shape = tuple(self.context.get_tensor_shape(self.output_name))

        host_out = np.empty(out_shape, dtype=np.float32)

        d_in = _cuda_ok(self._cudart.cudaMalloc(host_in.nbytes), "cudaMalloc(input)")[0]
        d_out = _cuda_ok(self._cudart.cudaMalloc(host_out.nbytes), "cudaMalloc(output)")[0]
        stream = _cuda_ok(self._cudart.cudaStreamCreate(), "cudaStreamCreate")[0]

        try:
            _cuda_ok(
                self._cudart.cudaMemcpyAsync(
                    d_in,
                    host_in.ctypes.data,
                    host_in.nbytes,
                    self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
                "cudaMemcpyAsync(H2D)",
            )

            if self._is_legacy_bindings:
                bindings = [0] * self.engine.num_bindings
                bindings[self.input_idx] = int(d_in)
                bindings[self.output_idx] = int(d_out)
                self.context.execute_async_v2(bindings=bindings, stream_handle=stream)
            else:
                self.context.set_tensor_address(self.input_name, int(d_in))
                self.context.set_tensor_address(self.output_name, int(d_out))
                self.context.execute_async_v3(stream)

            _cuda_ok(
                self._cudart.cudaMemcpyAsync(
                    host_out.ctypes.data,
                    d_out,
                    host_out.nbytes,
                    self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                ),
                "cudaMemcpyAsync(D2H)",
            )
            _cuda_ok(self._cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
        finally:
            self._cudart.cudaFree(d_in)
            self._cudart.cudaFree(d_out)
            self._cudart.cudaStreamDestroy(stream)

        return host_out
