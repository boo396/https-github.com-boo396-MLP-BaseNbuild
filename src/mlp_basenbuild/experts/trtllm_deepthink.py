from __future__ import annotations

from typing import Any

import httpx


class TRTLLMDeepThinkingRunner:
    def __init__(
        self,
        model_name: str,
        fp8_base_url: str,
        bf16_base_url: str = "",
        api_key: str = "",
        timeout_s: float = 60.0,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.fp8_base_url = fp8_base_url.rstrip("/")
        self.bf16_base_url = bf16_base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {"content-type": "application/json"}
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _request(self, base_url: str, prompt: str) -> dict[str, Any]:
        if not base_url:
            raise RuntimeError("Missing TensorRT-LLM endpoint URL")

        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            text = self._extract_text(data)
            return {
                "text": text,
                "raw": data,
                "endpoint": base_url,
            }

    def generate(self, prompt: str) -> dict[str, Any]:
        try:
            fp8_result = self._request(self.fp8_base_url, prompt)
            fp8_result["used_precision"] = "fp8"
            fp8_result["fallback_used"] = False
            return fp8_result
        except Exception as fp8_exc:
            if not self.bf16_base_url:
                raise RuntimeError(f"FP8 inference failed and no BF16 fallback URL configured: {fp8_exc}") from fp8_exc

            bf16_result = self._request(self.bf16_base_url, prompt)
            bf16_result["used_precision"] = "bf16"
            bf16_result["fallback_used"] = True
            bf16_result["fp8_error"] = str(fp8_exc)
            return bf16_result