from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8090


class RoutingConfig(BaseModel):
    model_names: List[str]
    keyword_shortcuts: Dict[str, List[str]]
    shortcut_confidence: float = 0.97


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
    trt_precision: str = "fp16"
    trt_workspace_mb: int = 2048
    allow_fallback: bool = True


class AppConfig(BaseModel):
    server: ServerConfig
    routing: RoutingConfig
    mlp: MLPConfig


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
