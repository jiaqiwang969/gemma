"""Interfaces for retrieval backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class RetrievalResult:
    """Single retrieval hit."""

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Index and query vector embeddings."""

    def add(
        self,
        *,
        ids: Sequence[str],
        embeddings: Any,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        ...

    def search(self, *, query_embedding: Any, top_k: int) -> List[RetrievalResult]:
        ...

    def delete(self, *, ids: Sequence[str]) -> None:
        ...
