import cv2
import logging
import numpy as np
from typing import Optional, List, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

    def __init__(self,
                 scene:  Optional[Scene] = None,
                 styles: RendererStyles  = None):
        self.scene  = scene
        self.styles = styles or RendererStyles()
        self._w:    int  = 0
        self._h:    int  = 0
        self._scene_drawables: List = []

    # ------------------------------------------------------------------ #
    # Renderer interface                                                   #
    # ------------------------------------------------------------------ #

    def initialize(self, frame: Frame) -> None:
        self._w = frame.width
        self._h = frame.height
        if self.scene:
            self._scene_drawables = self._build_scene_drawables()

    def render(self, ctx: FrameContext) -> np.ndarray:
        img = ctx.frame.frame.copy()
        self._draw_scene(img)
        # TODO
        # self._draw_detections(img, ctx)
        # self._draw_tracks(img, ctx)
        # self._draw_analytics(img, ctx)
        return img

    # ------------------------------------------------------------------ #
    # Scene (static — pre-built at initialize)                            #
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
        drawables = []
        if not self.styles.zones.show:
            return drawables

        for zone in self.scene.zones.values():
            zone_style: Optional[ZoneStyle] = self.styles.zones.per_zone[zone.name]
            if zone_style is None or not zone_style.show:
                continue

            polygon_style = zone_style.polygon
            text_style = zone_style.label
            drawables.append(PolygonDrawable(
                points=zone.polygon.points,
                style=polygon_style
            ))
            if zone_style.show_label:
                drawables.append(TextDrawable(
                    text=zone.name,
                    position=zone.polygon.points[0],
                    style=text_style
                ))
        return drawables

    def _build_line_drawables(self) -> List:
        drawables = []
        if not self.styles.lines.show:
            return drawables

        for line in self.scene.lines.values():
            scene_line_style: Optional[SceneLineStyle] = self.styles.lines.per_line[line.name]
            if line_style is None or not scene_line_style.show:
                continue

            line_style = scene_line_style.line
            text_style = scene_line_style.label

            drawables.append(LineDrawable(
                start=line.start,
                end=line.end,
                style=line_style
            ))
            if scene_line_style.show_label:
                drawables.append(TextDrawable(
                    text=line.name,
                    position=line.start,
                    style=text_style
                ))
        return drawables

    # ------------------------------------------------------------------ #
    # Detections (dynamic)                                                 #
    # ------------------------------------------------------------------ #

    def _draw_detections(self, img: np.ndarray,
                          ctx: FrameContext) -> None:
        style = self.styles.detections
        if not style.show:
            return

        for det in ctx.detections:
            if det.bbox is None:
                continue

            bbox = det.bbox.to_pixel_xyxy(self._w, self._h)

            if style.show_bbox:
                self._draw(BBoxDrawable(
                    bbox=bbox,
                    color=style.bbox.color,
                    thickness=style.bbox.thickness,
                ), img)

            if style.show_label and det.class_name:
                label = f"{det.class_name} {det.confidence:.2f}"
                self._draw(TextDrawable(
                    text=label,
                    position=(bbox[0], bbox[1] - 10),
                    color=style.label.color,
                    font_scale=style.label.font_size,
                    thickness=style.label.thickness,
                ), img)

    # ------------------------------------------------------------------ #
    # Tracks (dynamic)                                                     #
    # ------------------------------------------------------------------ #

    def _draw_tracks(self, img: np.ndarray,
                      ctx: FrameContext) -> None:
        style = self.styles.tracks
        if not style.show:
            return

        for track in ctx.tracks:
            if not track.detections:
                continue

            det = track.detections[-1]
            if det.bbox is None:
                continue

            bbox = det.bbox.to_pixel_xyxy(self._w, self._h)

            if style.show_bbox:
                self._draw(BBoxDrawable(
                    bbox=bbox,
                    color=style.bbox.color,
                    thickness=style.bbox.thickness,
                ), img)

            if style.show_id:
                self._draw(TextDrawable(
                    text=f"ID: {str(track.id)[:8]}",
                    position=(bbox[0], bbox[1] - 10),
                    color=style.id_label.color,
                    font_scale=style.id_label.font_size,
                    thickness=style.id_label.thickness,
                ), img)

            if style.show_trail:
                self._draw_trail(track, style.trail, img)

            if det.keypoints:
                keypoints = det.keypoints.to_pixel_xy(self._w, self._h)
                self._draw(KeypointsDrawable(keypoints), img)

    def _draw_trail(self, track, style, img: np.ndarray) -> None:
        points = [
            (int(d.bbox.cx * self._w), int(d.bbox.cy * self._h))
            for d in track.detections
            if d.bbox
        ]
        if len(points) < 2:
            return

        n = len(points)
        for i in range(1, n):
            t     = i / (n - 1)
            color, alpha = self._trail_segment_style(style, t)

            if alpha < 1.0:
                overlay = img.copy()
                cv2.line(overlay, points[i - 1], points[i],
                         color, style.thickness)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            else:
                cv2.line(img, points[i - 1], points[i],
                         color, style.thickness)

    def _trail_segment_style(self, style,
                              t: float) -> Tuple[Tuple, float]:
        """Return (color, alpha) for trail segment at position t (0=tail, 1=head)."""
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
    # Analytics (dynamic)                                                  #
    # ------------------------------------------------------------------ #

    def _draw_analytics(self, img: np.ndarray,
                         ctx: FrameContext) -> None:
        analytics_style = self.styles.analytics
        if not analytics_style.show:
            return

        for source, data in ctx.analytics.items():
            panel_style: Optional[PanelStyle] = analytics_style.for_panel(source)
            if panel_style is None or not panel_style.show:
                continue
            if not isinstance(data, dict):
                continue
            self._draw_panel(source, panel_style, data, img)

    def _draw_panel(self, source: str,
                    style: PanelStyle,
                    data: dict,
                    img: np.ndarray) -> None:
        x = int(style.position[0] * self._w)
        y = int(style.position[1] * self._h)

        keys  = list(style.show_keys) if style.show_keys else list(data.keys())
        lines = [f"{k}: {data[k]}" for k in keys if k in data]
        if not lines:
            return

        # draw background panel if filled
        if style.panel.filled and lines:
            line_height = style.text.font_size * 30 + 4
            panel_h     = int(len(lines) * line_height + style.padding * 2)
            max_text_w  = max(len(l) for l in lines) * int(style.text.font_size * 12)
            panel_w     = max_text_w + style.padding * 2
            self._draw(PolygonDrawable(
                points=self._pixel_rect_to_normalized(
                    x, y, panel_w, panel_h
                ),
                color=style.panel.color,
                filled=True,
                alpha=style.panel.alpha,
                thickness=0,
            ), img)

        # draw text lines
        y_offset = y + style.padding
        for line in lines:
            self._draw(TextDrawable(
                text=line,
                position=(x + style.padding, y_offset),
                color=style.text.color,
                font_scale=style.text.font_size,
                thickness=style.text.thickness,
            ), img)
            y_offset += int(style.text.font_size * 30 + 4)

    def _pixel_rect_to_normalized(self, x: int, y: int,
                                   w: int, h: int) -> List:
        """Convert pixel rect to normalized points for PolygonDrawable."""
        return [
            (x / self._w,       y / self._h),
            ((x + w) / self._w, y / self._h),
            ((x + w) / self._w, (y + h) / self._h),
            (x / self._w,       (y + h) / self._h),
        ]

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

    def _draw_bbox(self, d: BBoxDrawable, img: np.ndarray) -> None:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), d.color, d.thickness)

    def _draw_text(self, d: TextDrawable, img: np.ndarray) -> None:
        font_scale = d.style.font_size * self._h / BASE_HEIGHT
        x, y = int(d.position[0]*self._w), int(d.position[1]*self._h)
        cv2.putText(
            img, d.text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, d.style.color, d.style.thickness,
            cv2.LINE_AA,
        )

    def _draw_polygon(self, d: PolygonDrawable, img: np.ndarray) -> None:
        pixel_points = [
            (int(x * self._w), int(y * self._h))
            for x, y in d.points
        ]
        if not pixel_points:
            return

        pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))

        if d.style.filled:
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], d.style.color)
            cv2.addWeighted(overlay, d.style.alpha, img, 1 - d.style.alpha, 0, img)
        else:
            cv2.polylines(img, [pts], True, d.style.color, d.style.thickness)
        

    def _draw_line(self, d: LineDrawable, img: np.ndarray) -> None:
        start = (int(d.start[0] * self._w), int(d.start[1] * self._h))
        end   = (int(d.end[0]   * self._w), int(d.end[1]   * self._h))
        cv2.line(img, start, end, d.color, d.thickness)

        if d.label:
            mid         = ((start[0] + end[0]) // 2,
                           (start[1] + end[1]) // 2)
            label_color = getattr(d, "label_color", d.color)
            cv2.putText(
                img, d.label,
                (mid[0], mid[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, label_color, 2, cv2.LINE_AA,
            )

    def _draw_keypoints(self, d: KeypointsDrawable,
                         img: np.ndarray) -> None:
        for pt in d.keypoints:
            cv2.circle(img, pt, 3, d.color, -1)