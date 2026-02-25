from __future__ import annotations

from typing import Any, Dict

import httpx

from .config import AppConfig
from .router import RouteRequest, RouteResponse


class WorkerDispatcher:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def _resolve_endpoint_and_timeout(self, route: RouteResponse) -> tuple[str | None, float]:
        dispatch_target = route.dispatch_target
        if not dispatch_target:
            return None, 0.0

        if dispatch_target == self.cfg.deep_thinking.model_name:
            endpoint = self.cfg.deep_thinking.worker_endpoint or self.cfg.experts.worker_endpoints.get(dispatch_target, "")
            return (endpoint or None), self.cfg.deep_thinking.timeout_s

        endpoint = self.cfg.experts.worker_endpoints.get(dispatch_target, "")
        if endpoint:
            return endpoint, self.cfg.experts.timeout_s

        return None, 0.0

    async def dispatch(self, req: RouteRequest, route: RouteResponse) -> RouteResponse:
        endpoint, timeout_s = self._resolve_endpoint_and_timeout(route)
        if not endpoint:
            route.worker_status = "skipped_no_endpoint"
            return route

        has_image = req.has_image or bool(req.image_url) or bool(req.image_path)

        payload: Dict[str, Any] = {
            "text": req.text,
            "has_image": has_image,
            "image_url": req.image_url,
            "image_path": req.image_path,
            "route": {
                "model": route.model,
                "confidence": route.confidence,
                "top_k_models": route.top_k_models,
                "dispatch_target": route.dispatch_target,
                "dispatch_backend": route.dispatch_backend,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(endpoint, json=payload)
            route.worker_invoked = True
            route.worker_status = f"http_{response.status_code}"
            if response.is_success and self.cfg.dispatch.return_worker_response:
                route.worker_response = response.json()
            return route
        except Exception as exc:
            route.worker_invoked = True
            route.worker_status = f"error:{type(exc).__name__}"
            route.worker_response = {"error": str(exc)}
            return route
