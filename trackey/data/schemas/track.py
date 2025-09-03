from pydantic import BaseModel, Field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import List, Union
from .detection import Detection


class Track(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    private_id: Union[UUID, int] = Field(description="Track id generated for the track by the tracker library")
    detections: List[Detection]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the detection")
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        return (f"Track(id={self.id}, private_id={self.private_id}, "
                f"confidence={self.confidence:.2f}, "
                f"detections={len(self.detections)}, "
                f"last_seen={self.last_seen.isoformat()})")

    def __repr__(self) -> str:
        # Use same string for repr so printing lists of Tracks is clean
        return self.__str__()
