"""Interfaces for interaction log storage."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Protocol, Sequence

from ..schema.interaction_log import InteractionLog


class LogStore(Protocol):
    """Persists and queries interaction logs."""

    def append(self, *, logs: Sequence[InteractionLog]) -> None:
        ...

    def get(self, *, log_id: str) -> Optional[InteractionLog]:
        ...

    def query_time_range(
        self, *, session_id: Optional[str], start: datetime, end: datetime
    ) -> Iterable[InteractionLog]:
        ...

    def close(self) -> None:
        ...
