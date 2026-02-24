from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config
from .models import RouterMLP, resolve_device
from .trt_native import build_engine_from_onnx, export_router_to_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.arm.yaml")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dim = len(cfg.routing.model_names)
    device = resolve_device(cfg.mlp.device)

    model = RouterMLP(
        input_dim=cfg.mlp.input_dim,
        output_dim=output_dim,
        hidden_dims=cfg.mlp.hidden_dims,
        dropout=cfg.mlp.dropout,
    ).to(device)

    ckpt_path = Path(cfg.mlp.checkpoint_path)
    if ckpt_path.exists():
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

    onnx_path = export_router_to_onnx(
        model=model,
        input_dim=cfg.mlp.input_dim,
        onnx_path=cfg.mlp.onnx_path,
        opset=args.opset,
    )
    print(f"Exported ONNX: {onnx_path}")

    if args.build_engine:
        engine_path = build_engine_from_onnx(
            onnx_path=cfg.mlp.onnx_path,
            engine_path=cfg.mlp.engine_path,
            precision=cfg.mlp.trt_precision,
            workspace_mb=cfg.mlp.trt_workspace_mb,
        )
        print(f"Built TensorRT engine: {engine_path}")


if __name__ == "__main__":
    main()
