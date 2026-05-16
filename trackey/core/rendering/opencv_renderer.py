import numpy as np
from typing import Optional, List
from trackey.core.context import FrameContext
from trackey.core.interfaces.renderer import Renderer
from trackey.core.scene.scene import Scene
from trackey.data.schemas.frame import Frame
from trackey.core.register import register_renderer
from trackey.core.io.output.viewer.drawable import (
    Drawable,
    BBoxDrawable, 
    PointDrawable, 
    PolygonDrawable,
    LineDrawable,
    TextDrawable,
    KeypointsDrawable,
)


@register_renderer("opencv")
class OpenCVRenderer(Renderer):
    def __init__(self, scene: Optional[Scene] = None):
        self.scene = scene
        self._scene_drawables: List[Drawable] = []

    def initialize(self, frame: Frame) -> None:
        if not self.scene:
            return
        self._scene_drawables = self._build_scene_drawables(frame)

    def render(self, ctx: FrameContext) -> np.ndarray:
        img = ctx.frame.frame.copy()

        # static — pre-built at startup
        for drawable in self._scene_drawables:
            drawable.draw(img)

        # dynamic — built every frame
        self._draw_detections(img, ctx)
        self._draw_tracks(img, ctx)
        self._draw_analytics(img, ctx)

        return img

    # ------------------------------------------------------------------ #
    # Dynamic drawing                                                      #
    # ------------------------------------------------------------------ #

    def _draw_detections(self, img: np.ndarray,
                          ctx: FrameContext) -> None:
        for det in ctx.detections:
            if det.bbox is None:
                continue
            bbox = det.bbox.to_pixel_xyxy(ctx.frame.width, ctx.frame.height)
            BBoxDrawable(bbox, color=(0, 255, 0)).draw(img)

            if det.class_name:
                TextDrawable(
                    text=f"{det.class_name} {det.confidence:.2f}",
                    position=(bbox[0], bbox[1] - 10)
                ).draw(img)

    def _draw_tracks(self, img: np.ndarray,
                      ctx: FrameContext) -> None:
        for track in ctx.tracks:
            if "detections" not in track:
                continue

            det = track.detections[-1]
            if det.bbox is None:
                continue

            bbox = det.bbox.to_pixel_xyxy(ctx.frame.width, ctx.frame.height)

            # bounding box
            BBoxDrawable(bbox, color=(255, 165, 0)).draw(img)

            # track id above box
            TextDrawable(
                text=f"ID: {str(track.id)[:8]}",
                position=(bbox[0], bbox[1] - 10)
            ).draw(img)

            # class name if available
            if det.class_name:
                TextDrawable(
                    text=det.class_name,
                    position=(bbox[0], bbox[1] - 25)
                ).draw(img)

            # keypoints if available
            if det.keypoints:
                keypoints = det.keypoints.to_pixel_xy(
                    ctx.frame.width, ctx.frame.height
                )
                KeypointsDrawable(keypoints).draw(img)

    def _draw_analytics(self, img: np.ndarray,
                         ctx: FrameContext) -> None:
        if not ctx.analytics:
            return

        # draw each analytics result as overlay text
        y_offset = 30
        for name, result in ctx.analytics.items():
            if not isinstance(result, dict):
                continue

            count = result.get("count")
            if count is not None:
                TextDrawable(
                    text=f"{name}: {count}",
                    position=(10, y_offset),
                    color=(255, 255, 255),
                    font_scale=0.7,
                    thickness=2
                ).draw(img)
                y_offset += 25

    # ------------------------------------------------------------------ #
    # Static scene drawing                                                 #
    # ------------------------------------------------------------------ #

    def _build_scene_drawables(self, frame: Frame) -> List[Drawable]:
        drawables = []

        for zone in self.scene.zones.values():
            drawables.append(PolygonDrawable(
                points=zone.polygon.points,
                frame_width=frame.width,
                frame_height=frame.height,
                color=zone.color,
                filled=zone.filled,
                alpha=zone.alpha,
                label=zone.name
            ))

        for line in self.scene.lines.values():
            drawables.append(LineDrawable(
                start=line.start,
                end=line.end,
                frame_width=frame.width,
                frame_height=frame.height,
                color=line.color,
                label=line.name
            ))

        return drawables