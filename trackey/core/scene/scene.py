from typing import List, Optional
from  trackey.data.schemas.geometry import Zone, Line

class Scene:
    def __init__(self,
                 zones: Optional[List[Zone]] = [],
                 lines: Optional[List[Line]] = []):
        self.zones = {
            zone.name: zone
            for zone in (zones or [])
        }
        self.lines = {
            line.name: line
            for line in (lines or [])
        }

    def get_zone(self,name:str):
        return self.zones.get(name)

    def get_line(self,name:str):
        return self.lines.get(name)