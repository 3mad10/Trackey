from typing import Optional, Union
from dataclasses import dataclass
from uuid import UUID

from trackey.data.schemas.event     import BaseEvent
from trackey.core.register          import register_event
from trackey.core.context           import Frame


@register_event("count_exceeded_event")
@dataclass
class CountExceededEvent(BaseEvent):
    count: int
    zone_name: Optional[str] = None

@register_event("snapshot_event")
@dataclass
class SnapshotEvent(BaseEvent):
    frame:  Frame
    reason: str

@register_event("intrusion_event")
@dataclass
class IntrusionEvent(BaseEvent):
    zone_name: str
    track_id:  Union[UUID, int]
    frame:     Frame

@register_event("line_crossed_event")
@dataclass
class LineCrossedEvent(BaseEvent):
    line_name: str
    direction: str
    track_id:  Union[UUID, int]
