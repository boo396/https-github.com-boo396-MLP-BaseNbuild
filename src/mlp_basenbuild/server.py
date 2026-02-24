from __future__ import annotations

import argparse
import time

import uvicorn
from fastapi import FastAPI

from .config import AppConfig, load_config
from .models import InferenceBundle, build_inference_bundle
from .router import RouteRequest, RouteResponse, route_request


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="MLP BaseNbuild", version="0.1.0")
    bundle: InferenceBundle = build_inference_bundle(config.mlp, len(config.routing.model_names))

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "ts": int(time.time()), "backend": bundle.backend, "device": str(bundle.device)}

    @app.post("/route", response_model=RouteResponse)
    async def route(req: RouteRequest) -> RouteResponse:
        return route_request(req, config, bundle)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.arm.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
