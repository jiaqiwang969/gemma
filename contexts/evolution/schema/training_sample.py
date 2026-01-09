"""Training sample schemas for retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalSample:
    """Sample for contrastive retrieval training."""

    query: str
    positive_ids: List[str]
    negative_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "positive_ids": list(self.positive_ids),
            "negative_ids": list(self.negative_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalSample":
        return cls(
            query=data.get("query", ""),
            positive_ids=list(data.get("positive_ids", [])),
            negative_ids=list(data.get("negative_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class GenerationSample:
    """Sample for supervised generation training."""

    input_text: str
    context: Optional[str]
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "context": self.context,
            "response": self.response,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationSample":
        return cls(
            input_text=data.get("input_text", ""),
            context=data.get("context"),
            response=data.get("response", ""),
            metadata=dict(data.get("metadata", {})),
        )
