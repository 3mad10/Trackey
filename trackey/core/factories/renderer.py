import logging
from typing import Optional
from trackey.core.factories.builder import Builder
from trackey.core.scene.scene import Scene
from trackey.core.interfaces.renderer import Renderer
from trackey.core.registries.render import RENDERER_REGISTRY
from trackey.core.rendering.styling import (
    RendererStyles,
    PolygonStyle,
    BBoxStyle,
    TextStyle,
    TrailStyle,
    TrailFadeMode,
    DetectionStyle,
    TrackStyle,
    LineStyle,
    ZoneStyle,
    ZonesStyles,
    AnalyticsStyle,
    SceneLinesStyle,
    SceneLineStyle,
    PanelStyle
)

logger = logging.getLogger(__name__)


class RendererBuilder(Builder):

    def __init__(self, cfg_path: str, scene: Optional[Scene] = None):
        self.cfg   = self._load_yaml(cfg_path)
        self.scene = scene

    def build(self) -> Optional[Renderer]:
        renderer_cfg = self.cfg.get("renderer")
        if not renderer_cfg:
            return None

        if not isinstance(renderer_cfg, dict):
            raise ValueError("[RendererBuilder] 'renderer' must be a dict")

        renderer_type = renderer_cfg.get("type")
        if not renderer_type:
            raise ValueError("[RendererBuilder] Missing 'type'")

        renderer_cls = RENDERER_REGISTRY.get(renderer_type)
        if not renderer_cls:
            raise ValueError(
                f"[RendererBuilder] Unknown renderer '{renderer_type}'. "
                f"Available: {list(RENDERER_REGISTRY.keys())}"
            )

        styles   = self._build_styles(renderer_cfg.get("styles", {}))
        renderer = renderer_cls(scene=self.scene, styles=styles)
        logger.info(f"[RendererBuilder] Built renderer: {renderer_type}")
        return renderer

    # ------------------------------------------------------------------ #
    # Styles                                                             #
    # ------------------------------------------------------------------ #

    def _build_styles(self, cfg: dict) -> RendererStyles:
        if not cfg:
            return RendererStyles()
        return RendererStyles(
            detections=self._build_detection_style(cfg.get("detections", {})),
            tracks=self._build_track_style(cfg.get("tracks", {})),
            zones=self._build_zones_style(cfg.get("zones", {})),
            lines=self._build_lines_style(cfg.get("lines", {})),
            analytics=self._build_analytics_style(cfg.get("analytics", {})),
        )

    def _build_detection_style(self, cfg: dict) -> DetectionStyle:
        if not cfg:
            return DetectionStyle()
        defaults = DetectionStyle()
        return DetectionStyle(
            show=cfg.get("show", defaults.show),
            show_bbox=cfg.get("show_bbox", defaults.show_bbox),
            show_label=cfg.get("show_label", defaults.show_label),
            bbox=BBoxStyle(**cfg["bbox"])    if "bbox"  in cfg else BBoxStyle(),
            label=TextStyle(**cfg["label"])  if "label" in cfg else TextStyle(),
        )

    def _build_track_style(self, cfg: dict) -> TrackStyle:
        if not cfg:
            return TrackStyle()
        defaults  = TrackStyle()
        trail_cfg = cfg.get("trail", {})
        if "fade_mode" in trail_cfg:
            trail_cfg = {
                **trail_cfg,
                "fade_mode": TrailFadeMode(trail_cfg["fade_mode"])
            }
        return TrackStyle(
            show=cfg.get("show", defaults.show),
            show_bbox=cfg.get("show_bbox", defaults.show_bbox),
            show_id=cfg.get("show_id", defaults.show_id),
            show_trail=cfg.get("show_trail", defaults.show_trail),
            show_label=cfg.get("show_label", defaults.show_label),
            bbox=BBoxStyle(**cfg["bbox"])           if "bbox"     in cfg else BBoxStyle(),
            id_label=TextStyle(**cfg["id_label"])   if "id_label" in cfg else TextStyle(),
            label=TextStyle(**cfg["label"])         if "label" in cfg else TextStyle(),
            trail=TrailStyle(**trail_cfg)           if trail_cfg  else TrailStyle(),
        )

    def _build_zones_style(self, cfg: dict) -> ZonesStyles:
        if not cfg:
            return ZonesStyles()
        defaults  = ZoneStyle()
        per_zone_style = {}
        for zone_name, zone_cfg in cfg.items():
            if not isinstance(zone_cfg, dict):
                continue
            per_zone_style[zone_name] = ZoneStyle(
                show=zone_cfg.get("show", defaults.show),
                show_label=zone_cfg.get("show_label", defaults.show_label),
                polygon=PolygonStyle(**zone_cfg["polygon"]) if "polygon" in zone_cfg else defaults.polygon,
                label=TextStyle(**zone_cfg["label"])        if "label" in zone_cfg else defaults.label,

            )
        return ZonesStyles(
            show=cfg.get("show", ZonesStyles().show),
            per_zone=per_zone_style
            )

    def _build_lines_style(self, cfg: dict) -> SceneLinesStyle:
        if not cfg:
            return SceneLinesStyle()
        defaults  = SceneLineStyle()
        per_line_style = {}
        for line_name, line_cfg in cfg.items():
            if not isinstance(line_cfg, dict):
                continue
            per_line_style[line_name] = SceneLineStyle(
                show=line_cfg.get("show", defaults.show),
                show_label=line_cfg.get("show_label", defaults.show_label),
                line=LineStyle(**line_cfg["line"]) if "line" in line_cfg else defaults.line,
                label=TextStyle(**line_cfg["label"]) if "label" in line_cfg else defaults.label,
            )
        return SceneLinesStyle(
            show=cfg.get("show", SceneLinesStyle().show),
            per_line=per_line_style
            )

    def _build_analytics_style(self, cfg: dict) -> AnalyticsStyle:
        if not cfg:
            return AnalyticsStyle()
        panels = {}
        for key, panel_style in cfg.items():
            if key == "show" or not isinstance(panel_style, dict):
                continue
            panels[key] = PanelStyle(
                show=panel_style.get("show", True),
                position=tuple(panel_style.get("position", [0.02, 0.05])),
                show_keys=tuple(panel_style.get("show_keys", [])),
                panel=PolygonStyle(**panel_style["panel"]) if "panel" in panel_style else PolygonStyle(),
                text=TextStyle(**panel_style["text"])      if "text"  in panel_style else TextStyle(),
            )
        return AnalyticsStyle(
            show=cfg.get("show", AnalyticsStyle().show),
            panels=panels
        )