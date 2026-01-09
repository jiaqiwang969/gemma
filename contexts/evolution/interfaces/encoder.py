"""Interfaces for multi-modal encoders."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence


class Encoder(Protocol):
    """Encodes multimodal inputs into embeddings."""

    def encode(
        self,
        *,
        images: Optional[Sequence[Any]] = None,
        audio: Optional[Any] = None,
        text: Optional[Sequence[str]] = None,
    ) -> Any:
        ...

    def encode_for_retrieval(
        self,
        *,
        images: Optional[Sequence[Any]] = None,
        audio: Optional[Any] = None,
        text: Optional[Sequence[str]] = None,
    ) -> Any:
        ...

    def encode_for_generation(
        self,
        *,
        images: Optional[Sequence[Any]] = None,
        audio: Optional[Any] = None,
        text: Optional[Sequence[str]] = None,
    ) -> Any:
        ...
