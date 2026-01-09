"""Helpers to compute embeddings for evolution logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .encoder import EncoderConfig, RetrievalEncoder


@dataclass
class EmbeddingService:
    """High-level embedding service for text and images."""

    encoder: RetrievalEncoder
    pooling: str = "mean"

    def embed_text(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        vectors = self.encoder.encode_texts([text])
        return vectors[0] if vectors else None

    def embed_images(self, images: List[object]) -> Optional[List[float]]:
        if not images:
            return None
        vectors = self.encoder.encode_images(images)
        if not vectors:
            return None
        return _pool_vectors(vectors, mode=self.pooling)

    def embed_image_paths(self, paths: List[str]) -> Optional[List[float]]:
        images = _load_images(paths)
        if not images:
            return None
        return self.embed_images(images)


def build_embedding_service_from_env() -> EmbeddingService:
    import os

    model_name = os.environ.get(
        "EVOLUTION_EMBED_MODEL", "google/siglip-base-patch16-224"
    )
    device = os.environ.get("EVOLUTION_EMBED_DEVICE", "cpu")
    dtype = os.environ.get("EVOLUTION_EMBED_DTYPE", "float32")
    normalize = os.environ.get("EVOLUTION_EMBED_NORMALIZE", "1") == "1"
    pooling = os.environ.get("EVOLUTION_EMBED_POOLING", "mean")

    encoder = RetrievalEncoder(
        EncoderConfig(
            model_name=model_name,
            device=device,
            dtype=dtype,
            normalize=normalize,
        )
    )
    return EmbeddingService(encoder=encoder, pooling=pooling)


def _pool_vectors(vectors: List[List[float]], *, mode: str) -> List[float]:
    if not vectors:
        return []
    if mode not in {"mean", "first"}:
        raise ValueError("pooling must be 'mean' or 'first'")
    if mode == "first":
        return vectors[0]
    length = len(vectors[0])
    summed = [0.0 for _ in range(length)]
    for vector in vectors:
        for idx, value in enumerate(vector):
            summed[idx] += value
    return [value / len(vectors) for value in summed]


def _load_images(paths: List[str]) -> List[object]:
    from PIL import Image

    images = []
    for path in paths:
        try:
            images.append(Image.open(path).convert("RGB"))
        except OSError:
            continue
    return images
