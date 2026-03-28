from pydantic import BaseModel, Field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import List, Union, Tuple, Optional
from collections import deque
from typing import ClassVar
import itertools
from trackey.data.schemas.detection import BoundingBox


class Track(BaseModel):
    id: Union[UUID, int]
    history: deque[BoundingBox] = Field(
        default_factory=lambda: deque(maxlen=30)
    )
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    age: int = Field(ge=0, default=1)
    class_name: str
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    view_track: bool = True
    metadata: Optional[dict] = None
    features: Optional[List[float]] = None

    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        points = []

        for bbox in self.history:
            points.append((bbox.cx, bbox.cy))

        return points
