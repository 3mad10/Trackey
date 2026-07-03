import cv2
import logging
import numpy as np
from typing import Optional, List, Any, Tuple

from trackey.core.pipeline.constants import BASE_HEIGHT
from trackey.core.context import FrameContext
from trackey.core.interfaces.renderer import Renderer
from trackey.core.scene.scene import Scene
from trackey.data.schemas.frame import Frame
from trackey.core.register import register_renderer
from trackey.core.rendering.styling import (
    RendererStyles,
    TrailFadeMode,
    ZoneStyle,
    SceneLineStyle,
    PanelStyle,
)
from trackey.core.rendering.drawables import (
    BBoxDrawable,
    TextDrawable,
    PolygonDrawable,
    LineDrawable,
    KeypointsDrawable,
)

logger = logging.getLogger(__name__)


@register_renderer("opencv")
class OpenCVRenderer(Renderer):
    """
    All coordinates are normalized (0.0-1.0).

    Drawables store normalized coords.
    Draw methods convert to pixel using self._w / self._h.

    Scene drawables are pre-built at initialize() from normalized
    zone/line geometry - no per-frame conversion needed.

    Dynamic drawables (detections, tracks) convert bbox pixel coords
    back to normalized before storing so the same draw path is used.
    """

    def __init__(self,
                 scene:  Optional[Scene] = None,
                 styles: RendererStyles  = None):
        self.scene  = scene
        self.styles = styles or RendererStyles()
        self._w:    int = 0
        self._h:    int = 0
        self._scene_drawables: List = []

    # ------------------------------------------------------------------ #
    # Renderer interface                                                   #
    # ------------------------------------------------------------------ #

    def initialize(self, frame: Frame) -> None:
        """
        Called once with the first frame.
        Stores resolution and pre-builds static scene drawables.
        Zone/line points are already normalized - stored as-is.
        """
        self._w = frame.width
        self._h = frame.height
        if self.scene:
            self._scene_drawables = self._build_scene_drawables()

    def render(self, ctx: FrameContext) -> np.ndarray:
        img = ctx.frame.frame.copy()
        self._draw_scene(img)
        self._draw_detections(img, ctx)
        self._draw_tracks(img, ctx)
        self._draw_analytics(img, ctx)
        return img

    # ------------------------------------------------------------------ #
    # Coordinate helpers                                                   #
    # ------------------------------------------------------------------ #

    def _to_px(self, x: float, y: float) -> Tuple[int, int]:
        """Normalized -> pixel."""
        return int(x * self._w), int(y * self._h)

    def _bbox_px_to_norm(self, bbox: Tuple[int, int, int, int]
                          ) -> Tuple[float, float, float, float]:
        """Pixel xyxy -> normalized xyxy."""
        x1, y1, x2, y2 = bbox
        return (x1 / self._w, y1 / self._h,
                x2 / self._w, y2 / self._h)

    def _norm_rect_to_points(self, x: float, y: float,
                               w: float, h: float) -> List[Tuple[float, float]]:
        """Normalized rect -> normalized polygon points."""
        return [
            (x,     y),
            (x + w, y),
            (x + w, y + h),
            (x,     y + h),
        ]

    # ------------------------------------------------------------------ #
    # Scene - pre-built normalized drawables                               #
    # ------------------------------------------------------------------ #

    def _draw_scene(self, img: np.ndarray) -> None:
        for drawable in self._scene_drawables:
            self._draw(drawable, img)

    def _build_scene_drawables(self) -> List:
        drawables = []
        drawables.extend(self._build_zone_drawables())
        drawables.extend(self._build_line_drawables())
        return drawables

    def _build_zone_drawables(self) -> List:
        """
        Zone polygon points are already normalized.
        Store as-is - no conversion at initialize or render time.
        """
        drawables = []
        if not self.styles.zones.show:
            return drawables

        for zone in self.scene.zones.values():
            zone_style: Optional[ZoneStyle] = self.styles.zones.per_zone.get(
                zone.name, self.styles.zones.default
            )
            if zone_style is None or not zone_style.show:
                continue

            drawables.append(PolygonDrawable(
                points=zone.polygon.points,   # already normalized
                style=zone_style.polygon,
            ))

            if zone_style.show_label:
                lx, ly = zone.polygon.points[0]
                drawables.append(TextDrawable(
                    text=zone.name,
                    position=(lx, ly - 0.02),
                    style=zone_style.label,
                ))

        return drawables

    def _build_line_drawables(self) -> List:
        """
        Line start/end points are already normalized.
        Store as-is.
        """
        drawables = []
        if not self.styles.lines.show:
            return drawables

        for line in self.scene.lines.values():
            scene_line_style: Optional[SceneLineStyle] = self.styles.lines.per_line.get(
                line.name, self.styles.lines.default
            )
            if scene_line_style is None or not scene_line_style.show:
                continue

            drawables.append(LineDrawable(
                start=line.start,    # already normalized
                end=line.end,        # already normalized
                style=scene_line_style.line,
            ))

            if scene_line_style.show_label:
                mx = (line.start[0] + line.end[0]) / 2
                my = (line.start[1] + line.end[1]) / 2
                drawables.append(TextDrawable(
                    text=line.name,
                    position=(mx, my - 0.02),
                    style=scene_line_style.label,
                ))

        return drawables

    # ------------------------------------------------------------------ #
    # Detections - convert pixel bbox to normalized                        #
    # ------------------------------------------------------------------ #

    def _draw_detections(self, img: np.ndarray,
                          ctx: FrameContext) -> None:
        style = self.styles.detections
        if not style.show:
            return

        for det_source in ctx.detections:
            for det in ctx.detections[det_source]:
                if det.bbox is None:
                    continue

                px_bbox   = det.bbox.to_pixel_xyxy(self._w, self._h)
                norm_bbox = self._bbox_px_to_norm(px_bbox)
                x1, y1, _, _ = norm_bbox
                
                if style.show_bbox:
                    self._draw(BBoxDrawable(
                        bbox=norm_bbox,
                        style=style.bbox,
                    ), img)

                if style.show_label and det.class_name:
                    self._draw(TextDrawable(
                        text=f"{det.class_name} {det.confidence:.2f}",
                        position=(x1, y1 - 0.015),
                        style=style.label,
                    ), img)

    # ------------------------------------------------------------------ #
    # Tracks - convert pixel bbox to normalized                            #
    # ------------------------------------------------------------------ #

    def _draw_tracks(self, img: np.ndarray,
                      ctx: FrameContext) -> None:
        style = self.styles.tracks
        if not style.show:
            return

        for track in ctx.tracks:
            if not track.history:
                continue

            det_bbox = track.history[-1]

            px_bbox      = det_bbox.to_pixel_xyxy(self._w, self._h)
            norm_bbox    = self._bbox_px_to_norm(px_bbox)

            if style.show_bbox:
                self._draw(BBoxDrawable(
                    bbox=norm_bbox,
                    style=style.bbox,
                ), img)

            if style.show_id:
                anchor_x, anchor_y = self._bbox_anchor_top_left(norm_bbox)
                self._draw(TextDrawable(
                    text=f"ID: {str(track.id)[:8]}",
                    position=(anchor_x, anchor_y - 0.015),
                    style=style.id_label,
                ), img)
            
            if style.show_label:
                anchor_x, anchor_y = self._bbox_anchor_top_left(norm_bbox)
                self._draw(TextDrawable(
                    text=f"class: {str(track.class_name)[:8]}",
                    position=(anchor_x, anchor_y - 0.045),
                    style=style.label,
                ), img)

            if style.show_trail:
                self._draw_trail(track, style.trail, img)

    def _draw_trail(self, track, style, img: np.ndarray) -> None:
        # bbox.cx / cy are already normalized center coords
        points = [
            (bbox.cx, bbox.cy)
            for bbox in track.history
        ]
        if len(points) < 2:
            return

        n = len(points)
        for i in range(1, n):
            t            = i / (n - 1)
            color, alpha = self._trail_segment_style(style, t)
            p0           = self._to_px(*points[i - 1])
            p1           = self._to_px(*points[i])

            if alpha < 1.0:
                overlay = img.copy()
                cv2.line(overlay, p0, p1, color, style.thickness)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            else:
                cv2.line(img, p0, p1, color, style.thickness)

    def _trail_segment_style(self, style,
                              t: float) -> Tuple[Tuple, float]:
        if style.fade_mode == TrailFadeMode.NONE:
            return style.color, style.alpha
        if style.fade_mode == TrailFadeMode.ALPHA:
            alpha = style.min_alpha + t * (style.alpha - style.min_alpha)
            return style.color, alpha
        # GRADIENT
        color = tuple(
            int(style.tail_color[c] + t * (style.color[c] - style.tail_color[c]))
            for c in range(3)
        )
        return color, style.alpha

    # ------------------------------------------------------------------ #
    # Analytics                                                            #
    # ------------------------------------------------------------------ #

    def _draw_analytics(self, img: np.ndarray,
                         ctx: FrameContext) -> None:
        analytics_style = self.styles.analytics
        if not analytics_style.show:
            return

        auto_y = 0.05  # normalized auto-stack starting y

        for source, data in ctx.analytics.items():
            panel_style: Optional[PanelStyle] = analytics_style.panels.get(source)
            if panel_style is None or not panel_style.show:
                continue
            if not isinstance(data, dict):
                continue

            position = panel_style.position or (0.02, auto_y)
            self._draw_panel(panel_style, data, img, position)

            line_count = len(panel_style.show_keys) or len(data)
            auto_y    += line_count * 0.04 + 0.02

    def _draw_panel(self, style: PanelStyle,
                    data: dict,
                    img: np.ndarray,
                    position: Tuple[float, float]) -> None:
        px, py = position   # normalized

        keys  = list(style.show_keys) if style.show_keys else list(data.keys())
        lines = [f"{k}: {data[k]}" for k in keys if k in data]
        if not lines:
            return

        line_h_norm = style.text.font_size * 0.04
        padding_n   = 0.01

        if style.panel.filled:
            panel_w = max(len(l) for l in lines) * style.text.font_size * 0.012
            panel_h = len(lines) * line_h_norm + padding_n * 2
            self._draw(PolygonDrawable(
                points=self._norm_rect_to_points(px, py, panel_w, panel_h),
                style=style.panel,
            ), img)

        y_offset = py + padding_n
        for line in lines:
            self._draw(TextDrawable(
                text=line,
                position=(px + padding_n, y_offset),
                style=style.text,
            ), img)
            y_offset += line_h_norm

    # ------------------------------------------------------------------ #
    # Draw dispatch                                                        #
    # ------------------------------------------------------------------ #

    def _draw(self, drawable: Any, img: np.ndarray) -> None:
        handlers = {
            BBoxDrawable:      self._draw_bbox,
            TextDrawable:      self._draw_text,
            PolygonDrawable:   self._draw_polygon,
            LineDrawable:      self._draw_line,
            KeypointsDrawable: self._draw_keypoints,
        }
        handler = handlers.get(type(drawable))
        if handler:
            handler(drawable, img)
        else:
            logger.warning(
                f"[OpenCVRenderer] No handler for {type(drawable).__name__}"
            )

    # ------------------------------------------------------------------ #
    # Primitive draw methods - all accept normalized, convert to pixel    #
    # ------------------------------------------------------------------ #

    def _draw_bbox(self, d: BBoxDrawable, img: np.ndarray) -> None:
        x1, y1, x2, y2 = d.bbox           # normalized
        p1 = self._to_px(x1, y1)
        p2 = self._to_px(x2, y2)
        cv2.rectangle(img, p1, p2,
                      d.style.color, d.style.thickness)

    def _draw_text(self, d: TextDrawable, img: np.ndarray) -> None:
        x, y       = self._to_px(*d.position)        # normalized -> pixel
        font_scale = d.style.font_size * self._h / BASE_HEIGHT
        cv2.putText(
            img, d.text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, d.style.color, d.style.thickness,
            cv2.LINE_AA,
        )

    def _draw_polygon(self, d: PolygonDrawable, img: np.ndarray) -> None:
        pixel_points = [self._to_px(x, y) for x, y in d.points]
        if not pixel_points:
            return

        pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))

        if d.style.filled:
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], d.style.color)
            cv2.addWeighted(
                overlay, d.style.alpha,
                img,     1 - d.style.alpha,
                0, img,
            )
        else:
            cv2.polylines(img, [pts], True,
                          d.style.color, d.style.thickness)

    def _draw_line(self, d: LineDrawable, img: np.ndarray) -> None:
        start = self._to_px(*d.start)    # normalized -> pixel
        end   = self._to_px(*d.end)      # normalized -> pixel
        cv2.line(img, start, end,
                 d.style.color, d.style.thickness)

    def _draw_keypoints(self, d: KeypointsDrawable,
                         img: np.ndarray) -> None:
        # keypoints come from to_pixel_xy() - already pixel coords
        for pt in d.keypoints:
            cv2.circle(img, pt, 3, d.color, -1)
    
    # ------------------------------------------------------------------ #
    # Anchors helpers                                                    #
    # ------------------------------------------------------------------ #
    def _bbox_anchor_top_left(self, bbox):
        x1, y1, _, _ = bbox
        return x1, y1