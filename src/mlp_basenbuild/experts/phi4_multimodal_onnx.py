from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import httpx
from huggingface_hub import snapshot_download


class Phi4MultimodalOnnxRunner:
    def __init__(
        self,
        model_id: str,
        model_subdir: str,
        cache_dir: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        try:
            import onnxruntime_genai as og
        except Exception as exc:
            raise RuntimeError("onnxruntime-genai is required for Phi-4 multimodal ONNX worker") from exc

        self._og = og
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        cache_path = str(Path(cache_dir).expanduser())
        local_snapshot = snapshot_download(repo_id=model_id, cache_dir=cache_path)
        model_root = Path(local_snapshot)
        if model_subdir:
            model_root = model_root / model_subdir
        if not model_root.exists():
            raise FileNotFoundError(f"Model path not found: {model_root}")

        self.model_path = str(model_root)
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)

        processor_factory = getattr(self.model, "create_multimodal_processor", None)
        if callable(processor_factory):
            self.processor = processor_factory()
        else:
            processor_cls = getattr(og, "MultiModalProcessor", None)
            if processor_cls is None:
                raise RuntimeError("No multimodal processor available in onnxruntime_genai")
            self.processor = processor_cls(self.model)

    def _resolve_image(self, image_url: str | None, image_path: str | None) -> tuple[Any | None, str | None]:
        if image_path:
            path = Path(image_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"image_path not found: {path}")
            return self._og.Images.open(str(path)), None

        if image_url:
            response = httpx.get(image_url, timeout=30.0)
            response.raise_for_status()
            suffix = Path(image_url).suffix or ".img"
            with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            return self._og.Images.open(temp_path), temp_path

        return None, None

    def _build_prompt(self, text: str, has_image: bool) -> str:
        if has_image:
            return f"<|user|>\n<|image_1|>\n{text}\n<|end|>\n<|assistant|>"
        return f"<|user|>\n{text}\n<|end|>\n<|assistant|>"

    def generate(self, text: str, has_image: bool, image_url: str | None = None, image_path: str | None = None) -> dict[str, Any]:
        prompt = self._build_prompt(text=text, has_image=has_image)
        images, temp_image_path = self._resolve_image(image_url=image_url, image_path=image_path)

        try:
            if images is not None:
                processed_inputs = self.processor(prompt, images=images)
            else:
                processed_inputs = self.processor(prompt)

            params = self._og.GeneratorParams(self.model)
            params.set_inputs(processed_inputs)
            params.set_search_options(max_length=int(self.max_new_tokens), temperature=float(self.temperature))

            generator = self._og.Generator(self.model, params)
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

            sequence = generator.get_sequence(0)
            text_out = self.tokenizer.decode(sequence)
            return {
                "text": text_out,
                "model_path": self.model_path,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
        finally:
            if temp_image_path:
                try:
                    Path(temp_image_path).unlink(missing_ok=True)
                except Exception:
                    pass
