from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
from uuid import UUID

from trackey.data.schemas.track import Track
from trackey.core.scene.scene import Scene


@dataclass
class ZoneMemberships:
    by_track: Dict[UUID, List[str]] = field(default_factory=dict)
    by_zone: Dict[str, List[UUID]] = field(default_factory=dict)

    @classmethod
    def build(cls, tracks: List[Track], scene: Scene) -> "ZoneMemberships":
        by_track = defaultdict(list)
        by_zone = defaultdict(list)
        
        for track in tracks:
            position = track.bbox.center
            for zone_name, zone in scene.zones.items():
                if zone.contains(position):
                    by_track[track.tracker_id].append(zone_name)
                    by_zone[zone_name].append(track.tracker_id)
        
        return cls(by_track=dict(by_track), by_zone=dict(by_zone))