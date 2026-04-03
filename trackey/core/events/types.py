from typing import Optional
from trackey.data.schemas.event import Event


class CountExceededEvent(Event):
    zone_name: Optional[str] = None
    count: int
    threshold: int

