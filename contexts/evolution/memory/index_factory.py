"""Factory for retrieval index backends."""

from __future__ import annotations

from typing import Optional

from .vector_index import VectorIndex


def create_index(*, backend: str, metric: str) -> object:
    """Create an index backend, falling back to in-memory if needed."""

    normalized = (backend or "").lower()
    if normalized in {"faiss", "faiss-cpu", "faiss-gpu"}:
        try:
            from .faiss_index import FaissIndex

            return FaissIndex(metric=metric)
        except Exception:
            return VectorIndex(metric=metric)
    return VectorIndex(metric=metric)
