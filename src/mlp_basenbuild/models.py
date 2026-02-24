from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import MLPConfig


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


@dataclass
class InferenceBundle:
    model: RouterMLP
    device: torch.device
    use_fp16_cuda: bool
    backend: str
    trt_runtime: Optional[Any]


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_inference_bundle(config: MLPConfig, output_dim: int) -> InferenceBundle:
    device = resolve_device(config.device)
    model = RouterMLP(
        input_dim=config.input_dim,
        output_dim=output_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    checkpoint_path = Path(config.checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

    model.eval()

    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    backend = "pytorch"
    trt_runtime: Optional[Any] = None

    if config.backend.lower() == "tensorrt":
        try:
            from .trt_native import TensorRTRuntime

            engine_path = Path(config.engine_path)
            if not engine_path.exists():
                raise FileNotFoundError(f"TensorRT engine not found at {engine_path}")
            trt_runtime = TensorRTRuntime(str(engine_path))
            backend = "tensorrt"
        except Exception:
            if not config.allow_fallback:
                raise

    return InferenceBundle(
        model=model,
        device=device,
        use_fp16_cuda=config.use_fp16_cuda,
        backend=backend,
        trt_runtime=trt_runtime,
    )


def fast_hash_embedding(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    for token in text.lower().split():
        idx = (hash(token) % dim)
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def predict_probs(bundle: InferenceBundle, embedding: np.ndarray) -> np.ndarray:
    if bundle.backend == "tensorrt" and bundle.trt_runtime is not None:
        probs = bundle.trt_runtime.infer(embedding.reshape(1, -1))[0]
        return probs

    x = torch.from_numpy(embedding).unsqueeze(0).to(bundle.device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if bundle.device.type == "cuda" and bundle.use_fp16_cuda else nullcontext()
    with torch.inference_mode(), amp_ctx:
        out = bundle.model(x)
    return out.detach().float().cpu().numpy()[0]
