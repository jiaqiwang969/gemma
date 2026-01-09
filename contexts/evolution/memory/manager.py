"""Memory manager that syncs storage and vector index."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

from ..interfaces.memory_store import MemoryStore
from ..interfaces.retriever import RetrievalResult, Retriever
from ..schema.memory_fragment import MemoryFragment


class MemoryManager:
    """Coordinates memory storage with retrieval index updates."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        index: Retriever,
        embedding_selector: Optional[Callable[[MemoryFragment], Any]] = None,
    ) -> None:
        self._store = store
        self._index = index
        self._embedding_selector = embedding_selector or (
            lambda fragment: fragment.multimodal_embedding
        )

    def add(self, *, fragments: Sequence[MemoryFragment]) -> None:
        self._store.add(fragments=fragments)
        ids = []
        embeddings = []
        metadata = []
        for fragment in fragments:
            embedding = self._embedding_selector(fragment)
            if embedding is None:
                continue
            ids.append(fragment.id)
            embeddings.append(embedding)
            metadata.append(
                {
                    "session_id": fragment.session_id,
                    "timestamp": fragment.timestamp.isoformat(),
                    "source": fragment.source,
                }
            )
        if ids:
            self._index.add(ids=ids, embeddings=embeddings, metadata=metadata)

    def search(self, *, query_embedding: Any, top_k: int) -> Iterable[RetrievalResult]:
        return self._index.search(query_embedding=query_embedding, top_k=top_k)

    def get(self, *, fragment_id: str) -> Optional[MemoryFragment]:
        return self._store.get(fragment_id=fragment_id)

    def delete(self, *, fragment_ids: Sequence[str]) -> None:
        self._index.delete(ids=fragment_ids)

    def close(self) -> None:
        self._store.close()
