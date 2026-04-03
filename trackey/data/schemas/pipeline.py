from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection


class NodeCfg(BaseModel):
    name: str
    type: str
    processor: Optional[str]
    zone: Optional[str]
    params: Optional[Dict]

class PipelineResult(BaseModel):
    frame_id: int
    timestamp: float
    detections: List[Detection]
    tracks: List[Track]
    analytics: Dict[str, Any]
    metadata: Dict[str, Any]
