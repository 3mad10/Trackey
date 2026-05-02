from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Type, Dict

from trackey.core.utils.path import PathExtractor


@dataclass
class BaseEvent:
    frame_id:  int
    camera_id: str
    timestamp: datetime

@dataclass
class EventDefinition:
    event_type:     Type[BaseEvent]
    extract:        Dict[str, str]

    def build(self, ctx) -> BaseEvent:
        extracted = {
            field: PathExtractor(path).extract(ctx)
            for field, path in self.extract.items()
        }
        return self.event_type(
            frame_id=ctx.frame_id,
            camera_id=ctx.camera_id,
            timestamp=datetime.now(timezone.utc),
            **extracted
        )