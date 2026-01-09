"""FAISS-backed vector index implementation."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..interfaces.retriever import RetrievalResult


@dataclass
class _VectorEntry:
    metadata: Dict[str, Any]


class FaissIndex:
    """FAISS-backed vector index with optional cosine normalization."""

    def __init__(self, *, metric: str = "cosine") -> None:
        self._metric = metric
        self._index = None
        self._ids: List[str] = []
        self._metadata: Dict[str, _VectorEntry] = {}
        self._deleted: set[str] = set()
        self._lock = threading.RLock()

    def _ensure_index(self, dimension: int) -> None:
        if self._index is not None:
            return
        import faiss  # type: ignore

        if self._metric == "dot":
            self._index = faiss.IndexFlatIP(dimension)
        elif self._metric == "cosine":
            self._index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError("metric must be 'cosine' or 'dot'")

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
        if not vectors:
            return
        with self._lock:
            self._ensure_index(len(vectors[0]))
            if self._metric == "cosine":
                vectors = _normalize(vectors)
            import numpy as np

            self._index.add(np.array(vectors, dtype="float32"))
            for idx, meta in zip(ids, metadata):
                self._ids.append(idx)
                self._metadata[idx] = _VectorEntry(metadata=dict(meta))
                if idx in self._deleted:
                    self._deleted.remove(idx)

    def search(self, *, query_embedding: Any, top_k: int) -> List[RetrievalResult]:
        with self._lock:
            if self._index is None or not self._ids:
                return []
            query_vector = _coerce_vectors(query_embedding)
            if not query_vector:
                return []
            if self._metric == "cosine":
                query_vector = _normalize(query_vector)
            import numpy as np

            search_k = min(len(self._ids), max(top_k, top_k + len(self._deleted)))
            scores, indices = self._index.search(
                np.array(query_vector, dtype="float32"), search_k
            )
            results: List[RetrievalResult] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0 or idx >= len(self._ids):
                    continue
                doc_id = self._ids[idx]
                if doc_id in self._deleted:
                    continue
                entry = self._metadata.get(doc_id)
                results.append(
                    RetrievalResult(
                        id=doc_id,
                        score=float(score),
                        metadata=entry.metadata if entry else {},
                    )
                )
                if len(results) >= top_k:
                    break
            return results

    def delete(self, *, ids: Sequence[str]) -> None:
        with self._lock:
            for idx in ids:
                if idx in self._metadata:
                    self._deleted.add(idx)
                    self._metadata.pop(idx, None)

    def items(self) -> Iterable[str]:
        with self._lock:
            return [idx for idx in self._ids if idx not in self._deleted]


def _coerce_vectors(embeddings: Any) -> List[List[float]]:
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    if not isinstance(embeddings, list):
        raise TypeError("embeddings must be a list or array-like")
    if not embeddings:
        return []
    if isinstance(embeddings[0], list):
        return [[float(value) for value in vector] for vector in embeddings]
    return [[float(value) for value in embeddings]]


def _normalize(vectors: List[List[float]]) -> List[List[float]]:
    normalized = []
    for vector in vectors:
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0.0:
            normalized.append([0.0 for _ in vector])
        else:
            normalized.append([value / norm for value in vector])
    return normalized
