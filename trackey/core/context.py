from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, replace
from uuid import UUID

from trackey.data.schemas.frame import Frame
from trackey.data.schemas.detection import Detection, DetectionSource
from trackey.data.schemas.track import Track
from trackey.data.schemas.event import BaseEvent
from trackey.core.scene.mappings import ZoneMemberships


@dataclass(frozen=True)
class FrameContext:

    # hardware input — raw frame from source
    frame:                      Frame

    # identity — set once by Engine, never changed
    frame_id:                   int   = 0
    camera_id:                  str   = ""
    timestamp:                  float = 0.0

    # pipeline data — enriched by nodes
    detections:                 Dict[DetectionSource, 
                                     List[Detection]]       = field(default_factory=dict)

    # key   = tracktracker_id (UUID)
    # value = Key Value dict of detection source - detection object
    # populated by FaceAssociationNode or similar
    detections_associations:    Dict[int, Dict[str, List[Detection]]] = field(default_factory=dict)

    tracks:                     List[Track]                 = field(default_factory=list)
    # ephemeral per-track computed values
    # dwell_time, speed, pose, embeddings
    per_track:                  Dict[str, Dict[int, Any]]  = field(default_factory=dict)
    # Scene level analytics e.g. counts, heatmaps
    analytics:                  Dict[str, Any]              = field(default_factory=dict)
    zone_memberships:           ZoneMemberships             = field(default_factory=ZoneMemberships)
    events:                     List[BaseEvent]             = field(default_factory=list)
    # Latency, memory, timings
    metadata:                   Dict[str, Any]              = field(default_factory=dict)
    
    # DAG routing — set by SwitchNode/ConditionNode, consumed by executor
    active_branch:              Optional[str]               = None

    # ------------------------------------------------------------------ #
    # Pure functional updates — no business logic, no mutation            #
    # ------------------------------------------------------------------ #

    def with_detections(self, source: str,
                         detections: List[Detection]) -> "FrameContext":
        return replace(
            self,
            detections={**self.detections, source: detections}
        )

    def get_detections(self, source: str) -> List[Detection]:
        return self.detections.get(source, [])

    def with_associations(self,
                           associations: Dict[Union[UUID, int], List[Detection]]
                           ) -> "FrameContext":
        return replace(self, detections_associations=associations)

    def get_associations(self, track_id: UUID) -> List[Detection]:
        return self.detections_associations.get(track_id, [])

    def with_tracks(self, tracks: List[Track]) -> "FrameContext":
        return replace(self, tracks=tracks)

    def with_analytics(self, key: str, value: Any) -> "FrameContext":
        return replace(self, analytics={**self.analytics, key: value})

    def with_memberships(self,
                          memberships: ZoneMemberships) -> "FrameContext":
        return replace(self, zone_memberships=memberships)

    def with_branch(self, branch: str) -> "FrameContext":
        return replace(self, active_branch=branch)

    def with_events(self, events: List[BaseEvent]) -> "FrameContext":
        return replace(self, events=events)

    def with_metadata(self, key: str, value: Any) -> "FrameContext":
        return replace(self, metadata={**self.metadata, key: value})