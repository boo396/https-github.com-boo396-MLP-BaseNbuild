from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel

from .config import AppConfig
from .models import InferenceBundle, fast_hash_embedding, predict_probs


class RouteRequest(BaseModel):
    text: str
    has_image: bool = False


class RouteResponse(BaseModel):
    model: str
    confidence: float
    source: str
    probabilities: Dict[str, float]


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
    shortcut_model, shortcut_conf = shortcut_route(
        text=req.text,
        has_image=req.has_image,
        model_names=cfg.routing.model_names,
        keyword_shortcuts=cfg.routing.keyword_shortcuts,
    )

    if shortcut_model is not None and shortcut_conf >= cfg.routing.shortcut_confidence:
        probs = {name: (1.0 if name == shortcut_model else 0.0) for name in cfg.routing.model_names}
        return RouteResponse(
            model=shortcut_model,
            confidence=shortcut_conf,
            source="shortcut",
            probabilities=probs,
        )

    embedding = fast_hash_embedding(req.text, cfg.mlp.input_dim)
    probs_array = predict_probs(bundle, embedding)

    probs = {name: float(probs_array[idx]) for idx, name in enumerate(cfg.routing.model_names)}
    best_model = max(probs, key=probs.get)

    return RouteResponse(
        model=best_model,
        confidence=probs[best_model],
        source="mlp",
        probabilities=probs,
    )
