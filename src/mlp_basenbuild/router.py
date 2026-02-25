from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel

from .config import AppConfig
from .models import InferenceBundle, fast_hash_embedding, predict_probs


class RouteRequest(BaseModel):
    text: str
    has_image: bool = False
    image_url: str | None = None
    image_path: str | None = None


class RouteResponse(BaseModel):
    model: str
    confidence: float
    source: str
    probabilities: Dict[str, float]
    top_k_models: List[str]
    dispatch_target: str | None = None
    dispatch_backend: str | None = None
    worker_invoked: bool = False
    worker_status: str | None = None
    worker_response: Dict[str, object] | None = None


def shortcut_route(text: str, has_image: bool, model_names: List[str], keyword_shortcuts: Dict[str, List[str]]) -> Tuple[str | None, float]:
    text_lower = text.lower()

    if has_image and "vision_path" in model_names:
        return "vision_path", 0.99

    for target_model, keywords in keyword_shortcuts.items():
        if target_model not in model_names:
            continue
        if any(keyword in text_lower for keyword in keywords):
            return target_model, 0.98

    return None, 0.0


def route_request(req: RouteRequest, cfg: AppConfig, bundle: InferenceBundle) -> RouteResponse:
    top_k = max(1, min(cfg.routing.top_k, len(cfg.routing.model_names)))
    has_image = req.has_image or bool(req.image_url) or bool(req.image_path)

    shortcut_model, shortcut_conf = shortcut_route(
        text=req.text,
        has_image=has_image,
        model_names=cfg.routing.model_names,
        keyword_shortcuts=cfg.routing.keyword_shortcuts,
    )

    if shortcut_model is not None and shortcut_conf >= cfg.routing.shortcut_confidence:
        probs = {name: (1.0 if name == shortcut_model else 0.0) for name in cfg.routing.model_names}
        top_k_models = [shortcut_model]
        dispatch_target = cfg.deep_thinking.model_name if shortcut_model == cfg.deep_thinking.model_name else shortcut_model
        dispatch_backend = cfg.deep_thinking.runtime if dispatch_target == cfg.deep_thinking.model_name else cfg.experts.runtime
        return RouteResponse(
            model=shortcut_model,
            confidence=shortcut_conf,
            source="shortcut",
            probabilities=probs,
            top_k_models=top_k_models,
            dispatch_target=dispatch_target,
            dispatch_backend=dispatch_backend,
        )

    embedding = fast_hash_embedding(req.text, cfg.mlp.input_dim)
    probs_array = predict_probs(bundle, embedding)

    probs = {name: float(probs_array[idx]) for idx, name in enumerate(cfg.routing.model_names)}
    best_model = max(probs, key=probs.get)
    ranked_models = [name for name, _ in sorted(probs.items(), key=lambda item: item[1], reverse=True)]
    top_k_models = ranked_models[:top_k]

    dispatch_target = cfg.deep_thinking.model_name if best_model == cfg.deep_thinking.model_name and probs[best_model] >= cfg.deep_thinking.escalation_confidence else best_model
    dispatch_backend = cfg.deep_thinking.runtime if dispatch_target == cfg.deep_thinking.model_name else cfg.experts.runtime

    return RouteResponse(
        model=best_model,
        confidence=probs[best_model],
        source="mlp",
        probabilities=probs,
        top_k_models=top_k_models,
        dispatch_target=dispatch_target,
        dispatch_backend=dispatch_backend,
    )
