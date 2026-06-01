from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict
from enum import Enum

# ------------------------------------------------------------------ #
# Types                                                              #
# ------------------------------------------------------------------ #
class TrailFadeMode(Enum):
    NONE      = "none"       # uniform color, no fade
    ALPHA     = "alpha"      # opacity decreases toward tail
    GRADIENT  = "gradient"   # color interpolates from head to tail color

# ------------------------------------------------------------------ #
# Base                                                               #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class Style:
    color: Tuple[int, int, int] = (255, 255, 255)
    alpha: float                = 1.0


# ------------------------------------------------------------------ #
# Primitive styles                                                   #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class PolygonStyle(Style):
    color:     Tuple[int, int, int] = (255, 0, 255)
    alpha:     float = 0.3
    thickness: int   = 2
    filled:    bool  = False


@dataclass(frozen=True)
class LineStyle(Style):
    color:     Tuple[int, int, int] = (255, 0, 255)
    alpha:     float = 1.0
    thickness: int   = 2


@dataclass(frozen=True)
class BBoxStyle(Style):
    color:     Tuple[int, int, int] = (0, 255, 0)
    alpha:     float = 1.0
    thickness: int   = 2
    filled:    bool  = False


@dataclass(frozen=True)
class PointStyle(Style):
    color:  Tuple[int, int, int] = (0, 255, 0)
    alpha:  float = 1.0
    radius: int   = 3


@dataclass(frozen=True)
class TextStyle(Style):
    color:      Tuple[int, int, int] = (255, 255, 255)
    alpha:      float = 1.0
    font_size:  float = 0.5
    thickness:  int   = 1
    # font_family reserved for future non-OpenCV renderers
    font_family: str  = "sans-serif"


@dataclass(frozen=True)
class TrailStyle(Style):
    color:      Tuple[int, int, int] = (255, 255, 255)
    alpha:      float                = 1.0
    thickness:  int                  = 2
    max_points: int                  = 30
    fade_mode:  TrailFadeMode        = TrailFadeMode.ALPHA
    tail_color: Tuple[int, int, int] = (0, 0, 0)   # for gradient mode — color at tail
    min_alpha:  float                = 0.1         # for alpha mode — opacity at tail


# ------------------------------------------------------------------ #
# Domain styles                                                      #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class DetectionStyle:
    show:       bool        = True
    bbox:       BBoxStyle   = field(default_factory=BBoxStyle)
    label:      TextStyle   = field(default_factory=TextStyle)
    show_bbox:  bool        = False
    show_label: bool        = True

@dataclass(frozen=True)
class TrackStyle:
    show:       bool        = True
    bbox:       BBoxStyle   = field(default_factory=BBoxStyle)
    id_label:   TextStyle   = field(default_factory=TextStyle)
    trail:      TrailStyle  = field(default_factory=TrailStyle)
    show_bbox:  bool        = True
    show_id:    bool        = True
    show_trail: bool        = True

@dataclass(frozen=True)
class ZoneStyle:
    show:           bool            = True
    show_label:     bool            = True
    polygon:        PolygonStyle    = field(default_factory=PolygonStyle)
    label:          TextStyle       = field(default_factory=TextStyle)

@dataclass(frozen=True)
class ZonesStyles:
    show:       bool                    = True
    default:    ZoneStyle               = field(default_factory=ZoneStyle)
    per_zone:   dict[str, ZoneStyle]    = field(default_factory=dict)

@dataclass(frozen=True)
class SceneLineStyle:
    show:       bool        = True
    show_label: bool        = True
    line:       LineStyle   = field(default_factory=LineStyle)
    label:      TextStyle   = field(default_factory=TextStyle)

@dataclass(frozen=True)
class SceneLinesStyle:
    show:       bool                        = True
    default:    SceneLineStyle              = field(default_factory=SceneLineStyle)
    per_line:   dict[str, SceneLineStyle]   = field(default_factory=dict)

@dataclass(frozen=True)
class PanelStyle:
    show:         bool                  = True
    panel:        PolygonStyle          = field(default_factory=PolygonStyle)
    text:         TextStyle             = field(default_factory=TextStyle)
    padding:      int                   = 10
    line_spacing: int                   = 20
    position:     Tuple[float, float]   = (0.02, 0.05)
    show_keys:    Tuple[str, ...]       = ()

@dataclass(frozen=True)
class AnalyticsStyle:
    show:       bool                = True
    panels:     list[PanelStyle]    = field(default_factory=list)

# ------------------------------------------------------------------ #
# Root renderer style                                                #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class RendererStyles:
    detections: DetectionStyle  = field(default_factory=DetectionStyle)
    tracks:     TrackStyle      = field(default_factory=TrackStyle)
    zones:      ZonesStyles     = field(default_factory=ZonesStyles)
    lines:      SceneLinesStyle = field(default_factory=SceneLinesStyle)
    analytics:  AnalyticsStyle  = field(default_factory=AnalyticsStyle)