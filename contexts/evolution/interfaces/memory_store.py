"""Interfaces for memory storage backends."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Protocol, Sequence

from ..schema.memory_fragment import MemoryFragment


class MemoryStore(Protocol):
    """Persists and queries memory fragments."""

    def add(self, *, fragments: Sequence[MemoryFragment]) -> None:
        ...

    def get(self, *, fragment_id: str) -> Optional[MemoryFragment]:
        ...

    def query_time_range(
        self, *, session_id: Optional[str], start: datetime, end: datetime
    ) -> Iterable[MemoryFragment]:
        ...

    def close(self) -> None:
        ...
