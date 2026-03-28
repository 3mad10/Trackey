from pydantic import BaseModel, Field
from typing import List, Dict, Any

from trackey.data.schemas.track import Track
from trackey.core.io.output.viewer.drawable import Drawable
from trackey.data.schemas.detection import Detection

class PipelineResult(BaseModel):
    frame_id: int
    timestamp: float
    detections: List[Detection]
    tracks: List[Track]
    analytics: Dict[str, Any]
    metadata: Dict[str, Any]
