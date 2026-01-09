"""Embedding encoder for retrieval models (CLIP/SigLIP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str
    device: str = "cpu"
    dtype: str = "float32"
    normalize: bool = True


class RetrievalEncoder:
    """Loads a vision-text encoder and exposes embedding helpers."""

    def __init__(self, config: EncoderConfig) -> None:
        self._config = config
        self._processor = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._config.model_name)
        self._model = AutoModel.from_pretrained(self._config.model_name)
        self._device = torch.device(self._config.device)
        self._model.to(self._device)
        self._model.eval()

    def encode_images(self, images: List[object]) -> List[List[float]]:
        self._ensure_loaded()
        import torch

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(self._model, "get_image_features"):
                embeddings = self._model.get_image_features(**inputs)
            else:
                outputs = self._model(**inputs)
                embeddings = getattr(outputs, "pooler_output", outputs[0][:, 0])
        embeddings = embeddings.float()
        if self._config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().tolist()

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        import torch

        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(self._model, "get_text_features"):
                embeddings = self._model.get_text_features(**inputs)
            else:
                outputs = self._model(**inputs)
                embeddings = getattr(outputs, "pooler_output", outputs[0][:, 0])
        embeddings = embeddings.float()
        if self._config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().tolist()
