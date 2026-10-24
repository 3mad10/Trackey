from dataclasses import dataclass
from typing import List, Tuple
from trackey.core.rendering.styling import (
    BBoxStyle,
    PointStyle,
    LineStyle,
    TextStyle,
    PanelStyle,
    PolygonStyle
)


@dataclass
class BBoxDrawable:
    bbox:           Tuple[float, float, float, float]
    style:          BBoxStyle

@dataclass
class PointDrawable:
    point:          Tuple[float, float]
    style:          PointStyle

@dataclass
class KeypointsDrawable:
    keypoints:      List[Tuple[float, float]]
    style:          List[PointStyle]

@dataclass
class PolygonDrawable:
    points:         List[Tuple[float, float]]
    style:          PolygonStyle

@dataclass
class LineDrawable:
    start:          Tuple[float, float]
    end:            Tuple[float, float]
    style:          LineStyle

@dataclass
class TextDrawable:
    text:           str
    position:       Tuple[float, float]
    style:          TextStyle

@dataclass
class PanelDrawable:
    position: Tuple[float, float]
    lines: list[str]
    style: PanelStyle
