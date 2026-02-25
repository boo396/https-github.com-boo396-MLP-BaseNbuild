from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .experts import Phi4MultimodalOnnxRunner, TRTLLMDeepThinkingRunner


class InferRequest(BaseModel):
    text: str
    has_image: bool = False
    image_url: str | None = None
    image_path: str | None = None
    route: Dict[str, Any]


class InferResponse(BaseModel):
    worker: str
    runtime: str
    accepted: bool
    ts: int
    details: Dict[str, Any]


def create_worker_app(
    name: str,
    runtime: str,
    precision: str,
    model_id: str,
    model_subdir: str,
    cache_dir: str,
    max_new_tokens: int,
    temperature: float,
    trtllm_model_name: str,
    trtllm_fp8_url: str,
    trtllm_bf16_url: str,
    trtllm_api_key: str,
    trtllm_timeout_s: float,
) -> FastAPI:
    runner: Phi4MultimodalOnnxRunner | None = None
    deepthink_runner: TRTLLMDeepThinkingRunner | None = None

    if name == "vision_path" and runtime.startswith("onnxruntime"):
        runner = Phi4MultimodalOnnxRunner(
            model_id=model_id,
            model_subdir=model_subdir,
            cache_dir=cache_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    if name == "frontier_reasoning" and runtime == "trtllm":
        deepthink_runner = TRTLLMDeepThinkingRunner(
            model_name=trtllm_model_name,
            fp8_base_url=trtllm_fp8_url,
            bf16_base_url=trtllm_bf16_url,
            api_key=trtllm_api_key,
            timeout_s=trtllm_timeout_s,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    app = FastAPI(title=f"MLP Expert Worker ({name})", version="0.1.0")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "worker": name,
            "runtime": runtime,
            "precision": precision,
            "model_id": model_id,
            "model_subdir": model_subdir,
            "trtllm_model": trtllm_model_name,
            "trtllm_fp8_url": trtllm_fp8_url,
            "trtllm_bf16_url": trtllm_bf16_url,
            "ts": int(time.time()),
        }

    @app.post("/infer", response_model=InferResponse)
    async def infer(req: InferRequest) -> InferResponse:
        dispatch_target = str(req.route.get("dispatch_target", ""))
        accepted = dispatch_target == name
        details = {
            "dispatch_target": dispatch_target,
            "has_image": req.has_image,
            "text_chars": len(req.text),
            "runtime": runtime,
        }

        if accepted and runner is not None:
            result = runner.generate(
                text=req.text,
                has_image=req.has_image,
                image_url=req.image_url,
                image_path=req.image_path,
            )
            details["result"] = result

        if accepted and deepthink_runner is not None:
            result = deepthink_runner.generate(prompt=req.text)
            details["result"] = result

        return InferResponse(
            worker=name,
            runtime=runtime,
            accepted=accepted,
            ts=int(time.time()),
            details=details,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Worker model name, e.g. small_text")
    parser.add_argument("--runtime", default="onnxruntime_int4", help="Worker runtime label")
    parser.add_argument("--precision", default="int4", help="Runtime precision label")
    parser.add_argument("--model-id", default="microsoft/Phi-4-multimodal-instruct-onnx")
    parser.add_argument("--model-subdir", default="gpu/gpu-int4-rtn-block-32")
    parser.add_argument("--cache-dir", default="~/.cache/huggingface")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trtllm-model-name", default="microsoft/Phi-4-reasoning-plus")
    parser.add_argument("--trtllm-fp8-url", default="http://127.0.0.1:8301")
    parser.add_argument("--trtllm-bf16-url", default="")
    parser.add_argument("--trtllm-api-key", default="")
    parser.add_argument("--trtllm-timeout-s", type=float, default=60.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    trtllm_api_key = args.trtllm_api_key or os.getenv("TRTLLM_API_KEY", "")

    app = create_worker_app(
        name=args.name,
        runtime=args.runtime,
        precision=args.precision,
        model_id=args.model_id,
        model_subdir=args.model_subdir,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        trtllm_model_name=args.trtllm_model_name,
        trtllm_fp8_url=args.trtllm_fp8_url,
        trtllm_bf16_url=args.trtllm_bf16_url,
        trtllm_api_key=trtllm_api_key,
        trtllm_timeout_s=args.trtllm_timeout_s,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
