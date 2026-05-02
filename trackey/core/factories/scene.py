import yaml
import logging
from pathlib import Path
from trackey.core.scene.scene import Scene
from trackey.data.schemas.geometry import Zone, Line, Polygon
from trackey.core.factories.builder import Builder


logger = logging.getLogger(__name__)

class SceneBuilder(Builder):
    SCENE_CFG_FORMAT = f" \
    scene:\
        - zones: \
            - name: <zone-name> \
              polygon: \
                - [p1x,p1y] \
                - [p2x,p2y] \
                - [p3x,p3y] \
                - [p4x,p4y] \
              <You Can Optionally add the following> \
              color: [b, g, r] \
    "
    def __init__(self, cfg_path: str):
        self.cfg = self._load_yaml(cfg_path)

    def build(self):
        scene = self._build_scene()
        return scene
    
    def _build_scene(self):
        """Build all nodes from the loaded YAML in order."""
        zones=[]
        lines=[]
        scene = self.cfg.get("scene", [])
        if not isinstance(scene, dict):
            raise TypeError(f"[SceneBuilder] Scene must be in the following format \
             {self.SCENE_CFG_FORMAT}")
        
        if "zones" in scene:
            for zone_cfg in scene["zones"]:
                zones.append(self._build_zone(zone_cfg))
        if "lines" in scene:
            for line_cfg in scene["lines"]:
                lines.append(self._build_line(line_cfg))
        return Scene(zones=zones, lines=lines)
        
    def _build_zone(self, zone_cfg):
        if "name" not in zone_cfg:
            raise ValueError(f"[SceneBuilder] Zone must contain a unique name \
                             {self.SCENE_CFG_FORMAT}")
        if "polygon" not in zone_cfg:
            raise ValueError(f"[SceneBuilder] Zone must contain a polygon \
            {self.SCENE_CFG_FORMAT}")
        return Zone.model_validate(
            {
                'name': zone_cfg['name'],
                'polygon': Polygon(points=zone_cfg["polygon"]),
                'color': zone_cfg['color']
            }
        )
    
    def _build_line(self, line_cfg):
        if "name" not in line_cfg:
            raise ValueError("[SceneBuilder] Line must contain a unique name")
        if "start" not in line_cfg:
            raise ValueError("[SceneBuilder] Line must contain start tuple")
        if "end" not in line_cfg:
            raise ValueError("[SceneBuilder] Line must contain start tuple")
        return Line(**line_cfg)