from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, model_validator


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8090


class RoutingConfig(BaseModel):
    model_names: List[str]
    keyword_shortcuts: Dict[str, List[str]]
    shortcut_confidence: float = 0.97
    top_k: int = 2
    enable_worker_dispatch: bool = True


class MLPConfig(BaseModel):
    input_dim: int = 1024
    hidden_dims: List[int] = [256, 128]
    dropout: float = 0.0
    compile: bool = True
    use_fp16_cuda: bool = True
    device: str = "auto"
    backend: str = "pytorch"
    checkpoint_path: str = "artifacts/router_mlp.pth"
    onnx_path: str = "artifacts/router_mlp.onnx"
    engine_path: str = "artifacts/router_mlp.engine"
    trt_precision: str = "fp8"
    trt_workspace_mb: int = 2048
    allow_fallback: bool = True


class ExpertsConfig(BaseModel):
    runtime: str = "onnxruntime_int4"
    hidden_size: int = 4096
    ffn_dim: int = 16384
    divisible_by: int = 128
    preferred_models: List[str] = ["small_text", "vision_path"]
    worker_endpoints: Dict[str, str] = {}
    timeout_s: float = 20.0
    vision_model_id: str = "microsoft/Phi-4-multimodal-instruct-onnx"
    vision_model_subdir: str = "gpu/gpu-int4-rtn-block-32"
    hf_cache_dir: str = "~/.cache/huggingface"
    max_new_tokens: int = 256
    temperature: float = 0.0

    @model_validator(mode="after")
    def validate_dims(self) -> "ExpertsConfig":
        if not (1536 <= self.hidden_size <= 4096):
            raise ValueError("experts.hidden_size must be between 1536 and 4096")
        if not (8192 <= self.ffn_dim <= 16384):
            raise ValueError("experts.ffn_dim must be between 8192 and 16384")
        if self.hidden_size % self.divisible_by != 0:
            raise ValueError("experts.hidden_size must be divisible by experts.divisible_by")
        if self.ffn_dim % self.divisible_by != 0:
            raise ValueError("experts.ffn_dim must be divisible by experts.divisible_by")
        return self


class DeepThinkingConfig(BaseModel):
    model_name: str = "frontier_reasoning"
    runtime: str = "trtllm"
    precision: str = "fp8"
    fallback_precision: str = "bf16"
    worker_endpoint: str = ""
    trtllm_model_name: str = "microsoft/Phi-4-reasoning-plus"
    trtllm_fp8_url: str = "http://127.0.0.1:8301"
    trtllm_bf16_url: str = ""
    trtllm_timeout_s: float = 60.0
    max_new_tokens: int = 512
    temperature: float = 0.0
    timeout_s: float = 45.0
    escalation_confidence: float = 0.55


class DispatchConfig(BaseModel):
    return_worker_response: bool = True


class AppConfig(BaseModel):
    server: ServerConfig
    routing: RoutingConfig
    mlp: MLPConfig
    experts: ExpertsConfig = ExpertsConfig()
    deep_thinking: DeepThinkingConfig = DeepThinkingConfig()
    dispatch: DispatchConfig = DispatchConfig()


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
