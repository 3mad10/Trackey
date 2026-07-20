from datetime import datetime, timezone
from uuid import UUID
from typing import List, Union, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
import numpy as np

from trackey.data.schemas.detection import BoundingBox
from trackey.data.schemas.identity import Identity


@dataclass(slots=True)
class Track:
    tracker_id: int
    bbox: BoundingBox
    class_name: str

    identity: Optional[Identity] = None
    confidence: float = 1.0
    age: int = 1

    history: deque[BoundingBox] = field(
        default_factory=lambda: deque(maxlen=30)
    )

    last_seen: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.age < 0:
            raise ValueError("age must be >= 0")

    @property
    def trajectory(self)-> List[Tuple[float, float]]:
        return [(b.cx, b.cy) for b in self.history]

    @property
    def global_id(self) -> Optional[UUID]:
        return self.identity.global_id if self.identity else None

    @property
    def label(self) -> Optional[str]:
        return self.identity.label if self.identity else None