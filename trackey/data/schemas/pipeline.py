import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.event import BaseEvent
from trackey.data.schemas.frame import Frame
from trackey.core.context import FrameContext

class NodeCfg(BaseModel):
    name: str
    type: str
    processor: Optional[str]
    zone: Optional[str]
    params: Optional[Dict]


class PipelineResult(BaseModel):
    frame_id:       int
    timestamp:      float
    camera_id:      str
    detections:     List[Detection]
    tracks:         List[Track]
    analytics:      Dict[str, Any]
    events:         List[BaseEvent]
    metadata:       Dict[str, Any]
    rendered_frame: Optional[np.ndarray] = None  # post-render image
    raw_frame:      Optional[Frame]      = None  # original Frame object

    model_config = {
        "arbitrary_types_allowed": True
    }

    @classmethod
    def from_context(cls, ctx: FrameContext,
                     rendered: Optional[np.ndarray] = None) -> "PipelineResult":
        return cls(
            frame_id=ctx.frame_id,
            timestamp=ctx.timestamp,
            camera_id=ctx.camera_id,
            detections=ctx.detections,
            tracks=ctx.tracks,
            analytics=ctx.analytics,
            events=ctx.events,
            metadata=ctx.metadata,
            rendered_frame=rendered,
            raw_frame=ctx.frame      # Frame object, not np.ndarray
        )
    