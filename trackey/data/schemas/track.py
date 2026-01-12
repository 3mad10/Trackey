from pydantic import BaseModel, Field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import List, Union, Tuple, Optional
from collections import deque
from trackey.data.schemas.detection import Detection


class Track(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    private_id: Union[UUID, int]
    detections: deque[Detection] = Field(
        default_factory=lambda: deque(maxlen=30)
    )
    confidence: float = Field(ge=0.0, le=1.0)
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    view_track: bool = True
    metadata: Optional[dict] = None

    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        points = []

        for det in self.detections:
            if det.bbox is not None:
                points.append((det.bbox.cx, det.bbox.cy))
            elif det.points is not None:
                points.append(det.points)

        return points
