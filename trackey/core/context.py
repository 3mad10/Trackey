from typing import List, Dict, Any
from dataclasses import dataclass, field
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.track import Track
from trackey.core.scene.mappings import ZoneMemberships


@dataclass
class FrameContext:
    frame: Frame
    frame_id: int = 0
    timestamp: float = 0.0
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    zone_memberships: ZoneMemberships = field(default_factory=ZoneMemberships)
    analytics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)