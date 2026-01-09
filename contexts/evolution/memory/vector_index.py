"""In-memory vector index placeholder for retrieval backends."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..interfaces.retriever import RetrievalResult


@dataclass
class _VectorEntry:
    vector: List[float]
    metadata: Dict[str, Any]


class VectorIndex:
    """Simple in-memory vector index used as a FAISS placeholder."""

    def __init__(self, *, metric: str = "cosine") -> None:
        self._metric = metric
        self._entries: Dict[str, _VectorEntry] = {}
        self._lock = threading.RLock()

    def add(
        self,
        *,
        ids: Sequence[str],
        embeddings: Any,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        vectors = _coerce_vectors(embeddings)
        if len(ids) != len(vectors):
            raise ValueError("ids and embeddings must be the same length")
        if metadata is None:
            metadata = [{} for _ in ids]
        if len(metadata) != len(ids):
            raise ValueError("metadata must be the same length as ids")
        with self._lock:
            for idx, vector, meta in zip(ids, vectors, metadata):
                self._entries[idx] = _VectorEntry(vector=vector, metadata=dict(meta))

    def search(self, *, query_embedding: Any, top_k: int) -> List[RetrievalResult]:
        with self._lock:
            if not self._entries:
                return []
            query_vector = _coerce_vector(query_embedding)
            scored: List[Tuple[str, float]] = []
            for idx, entry in self._entries.items():
                score = _similarity(query_vector, entry.vector, metric=self._metric)
                scored.append((idx, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            results = []
            for idx, score in scored[:top_k]:
                results.append(
                    RetrievalResult(id=idx, score=score, metadata=self._entries[idx].metadata)
                )
            return results

    def delete(self, *, ids: Sequence[str]) -> None:
        with self._lock:
            for idx in ids:
                self._entries.pop(idx, None)

    def items(self) -> Iterable[str]:
        with self._lock:
            return list(self._entries.keys())


def _coerce_vectors(embeddings: Any) -> List[List[float]]:
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    if not isinstance(embeddings, list):
        raise TypeError("embeddings must be a list or array-like")
    if not embeddings:
        return []
    if isinstance(embeddings[0], list):
        return [
            [float(value) for value in vector]
            for vector in embeddings
        ]
    return [[float(value) for value in embeddings]]


def _coerce_vector(embedding: Any) -> List[float]:
    vectors = _coerce_vectors(embedding)
    if not vectors:
        raise ValueError("query embedding must not be empty")
    return vectors[0]


def _similarity(left: List[float], right: List[float], *, metric: str) -> float:
    if metric not in {"cosine", "dot"}:
        raise ValueError("metric must be 'cosine' or 'dot'")
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    dot = sum(a * b for a, b in zip(left, right))
    if metric == "dot":
        return dot
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
