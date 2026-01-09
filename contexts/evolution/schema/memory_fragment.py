"""Memory fragment schema for multimodal interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MemoryFragment:
    """Atomic memory unit used for retrieval and training."""

    id: str
    timestamp: datetime
    session_id: str
    turn_index: Optional[int] = None
    source: str = "unknown"

    keyframe_path: Optional[str] = None
    audio_path: Optional[str] = None
    text_input: Optional[str] = None

    visual_embedding: Optional[Any] = None
    audio_embedding: Optional[Any] = None
    multimodal_embedding: Optional[Any] = None

    transcript: Optional[str] = None
    scene_description: Optional[str] = None
    detected_objects: List[str] = field(default_factory=list)
    user_intent: Optional[str] = None

    ai_response: Optional[str] = None

    gaze_heatmap_path: Optional[str] = None
    focus_duration_s: Optional[float] = None

    explicit_feedback: Optional[str] = None
    implicit_feedback_score: Optional[float] = None

    model_version: Optional[str] = None
    lora_version: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """Serialize to a dictionary for logging or storage."""

        payload: Dict[str, Any] = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "source": self.source,
            "keyframe_path": self.keyframe_path,
            "audio_path": self.audio_path,
            "text_input": self.text_input,
            "transcript": self.transcript,
            "scene_description": self.scene_description,
            "detected_objects": list(self.detected_objects),
            "user_intent": self.user_intent,
            "ai_response": self.ai_response,
            "gaze_heatmap_path": self.gaze_heatmap_path,
            "focus_duration_s": self.focus_duration_s,
            "explicit_feedback": self.explicit_feedback,
            "implicit_feedback_score": self.implicit_feedback_score,
            "model_version": self.model_version,
            "lora_version": self.lora_version,
            "metadata": dict(self.metadata),
        }

        if include_embeddings:
            payload.update(
                {
                    "visual_embedding": self.visual_embedding,
                    "audio_embedding": self.audio_embedding,
                    "multimodal_embedding": self.multimodal_embedding,
                }
            )

        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryFragment":
        """Deserialize from a stored dictionary."""

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            id=data["id"],
            timestamp=timestamp,
            session_id=data["session_id"],
            turn_index=data.get("turn_index"),
            source=data.get("source", "unknown"),
            keyframe_path=data.get("keyframe_path"),
            audio_path=data.get("audio_path"),
            text_input=data.get("text_input"),
            visual_embedding=data.get("visual_embedding"),
            audio_embedding=data.get("audio_embedding"),
            multimodal_embedding=data.get("multimodal_embedding"),
            transcript=data.get("transcript"),
            scene_description=data.get("scene_description"),
            detected_objects=list(data.get("detected_objects", [])),
            user_intent=data.get("user_intent"),
            ai_response=data.get("ai_response"),
            gaze_heatmap_path=data.get("gaze_heatmap_path"),
            focus_duration_s=data.get("focus_duration_s"),
            explicit_feedback=data.get("explicit_feedback"),
            implicit_feedback_score=data.get("implicit_feedback_score"),
            model_version=data.get("model_version"),
            lora_version=data.get("lora_version"),
            metadata=dict(data.get("metadata", {})),
        )
