"""Configuration for evolution components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class EvolutionConfig:
    """Runtime configuration for storage, retrieval, and training."""

    storage_root: Path = Path("artifacts/evolution")
    sqlite_path: Path = field(
        default_factory=lambda: Path("artifacts/evolution/logs.db")
    )
    artifact_root: Path = field(
        default_factory=lambda: Path("artifacts/evolution/artifacts")
    )
    vector_index_path: Path = field(
        default_factory=lambda: Path("artifacts/evolution/index")
    )
    retention_days: int = 30
    min_feedback_score: float = 0.6
    short_term_window_minutes: int = 5
    short_term_window_frames: int = 5
    history_retrieval_model: str = "siglip"
    history_retrieval_backend: str = "faiss"
    max_model_versions: Optional[int] = None

    def ensure_paths(self) -> None:
        """Create storage directories if they do not exist."""

        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.vector_index_path.mkdir(parents=True, exist_ok=True)
