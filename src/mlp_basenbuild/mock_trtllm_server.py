from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0


class ChatChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]


def create_mock_app(precision: str, fail_mode: str) -> FastAPI:
    app = FastAPI(title=f"Mock TRT-LLM ({precision})", version="0.1.0")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "precision": precision,
            "fail_mode": fail_mode,
            "ts": int(time.time()),
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
        if fail_mode == "always":
            raise HTTPException(status_code=503, detail=f"mock {precision} configured to fail")

        if fail_mode == "reasoning-only":
            prompt_text = "\n".join(message.content for message in req.messages)
            keywords = ["prove", "theorem", "derive", "rigorous", "step by step"]
            if any(keyword in prompt_text.lower() for keyword in keywords):
                raise HTTPException(status_code=503, detail=f"mock {precision} reasoning-path failure")

        prompt_text = "\n".join(message.content for message in req.messages)
        response_text = f"[{precision}] mock response for model={req.model}: {prompt_text[:200]}"
        choice = ChatChoice(index=0, message={"role": "assistant", "content": response_text})
        return ChatCompletionResponse(
            id=f"mock-{precision}-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=req.model,
            choices=[choice],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp8", "bf16"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--fail-mode", choices=["never", "always", "reasoning-only"], default="never")
    args = parser.parse_args()

    fail_mode = os.getenv("MOCK_TRTLLM_FAIL_MODE", args.fail_mode)
    app = create_mock_app(precision=args.precision, fail_mode=fail_mode)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
