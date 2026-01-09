"""Interaction log schema for conversational data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Feedback:
    """Explicit or implicit feedback for a response."""

    rating: Optional[str] = None
    score: Optional[float] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rating": self.rating,
            "score": self.score,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feedback":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            rating=data.get("rating"),
            score=data.get("score"),
            timestamp=timestamp,
            source=data.get("source"),
        )


@dataclass
class InteractionLog:
    """Structured record of a single interaction turn."""

    id: str
    session_id: str
    turn_index: int
    timestamp: datetime

    user_input: str
    input_modalities: List[str] = field(default_factory=lambda: ["text"])
    attachments: List[str] = field(default_factory=list)

    assistant_response: Optional[str] = None
    thought_signature: Optional[str] = None
    response_time_ms: Optional[int] = None

    model_version: Optional[str] = None
    lora_version: Optional[str] = None

    retrieved_doc_ids: List[str] = field(default_factory=list)
    feedback: Optional[Feedback] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "input_modalities": list(self.input_modalities),
            "attachments": list(self.attachments),
            "assistant_response": self.assistant_response,
            "thought_signature": self.thought_signature,
            "response_time_ms": self.response_time_ms,
            "model_version": self.model_version,
            "lora_version": self.lora_version,
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionLog":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        feedback_data = data.get("feedback")
        feedback = Feedback.from_dict(feedback_data) if feedback_data else None

        return cls(
            id=data["id"],
            session_id=data["session_id"],
            turn_index=data["turn_index"],
            timestamp=timestamp,
            user_input=data.get("user_input", ""),
            input_modalities=list(data.get("input_modalities", ["text"])),
            attachments=list(data.get("attachments", [])),
            assistant_response=data.get("assistant_response"),
            thought_signature=data.get("thought_signature"),
            response_time_ms=data.get("response_time_ms"),
            model_version=data.get("model_version"),
            lora_version=data.get("lora_version"),
            retrieved_doc_ids=list(data.get("retrieved_doc_ids", [])),
            feedback=feedback,
            metadata=dict(data.get("metadata", {})),
        )
