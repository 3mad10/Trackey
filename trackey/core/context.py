from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, replace

from trackey.data.schemas.frame import Frame
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.track import Track
from trackey.data.schemas.event import BaseEvent
from trackey.data.schemas.feature import Embedding
from trackey.core.scene.mappings import ZoneMemberships


@dataclass(frozen=True)
class FrameContext:
    # hardware input — raw frame from source
    frame: Frame

    # identity — set once by Engine, never changed
    frame_id: int = 0
    camera_id: int = 0
    timestamp: float = 0.0

    # pipeline data — enriched by nodes
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    zone_memberships: ZoneMemberships = field(default_factory=ZoneMemberships)
    analytics: Dict[str, Any] = field(default_factory=dict)
    events: List[BaseEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # DAG routing — set by SwitchNode, consumed by executor
    active_branch: Optional[str] = None

    def with_detections(self, detections: List[Detection]) -> "FrameContext":
        return replace(self, detections=detections)

    def with_tracks(self, tracks: List[Track]) -> "FrameContext":
        return replace(self, tracks=tracks)

    def with_analytics(self, key: str, value: Any) -> "FrameContext":
        return replace(self, analytics={**self.analytics, key: value})
    
    def with_embeddings(self, tracks: List[Track], embeddings: List[Embedding]) -> "FrameContext":
        for i in range(len(tracks)):
            tracks[i].embedding=embeddings[i]
        return replace(self, tracks=tracks)
    
    def with_memberships(self, memberships: ZoneMemberships) -> "FrameContext":
        return replace(self, zone_memberships=memberships)
    
    def with_branch(self, branch: str) -> "FrameContext":
        return replace(self, active_branch=branch)
        