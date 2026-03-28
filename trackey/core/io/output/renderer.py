from typing import List

from trackey.data.schemas.track import Track
from trackey.data.schemas.frame import Frame
from trackey.core.context import FrameContext
from trackey.core.io.output.viewer.drawable import (
    Drawable,
    BBoxDrawable,
    TextDrawable,
    PointDrawable,
    KeypointsDrawable
)


class Renderer:
    def render(self, ctx: FrameContext) -> None:
        drawables: List[Drawable] = []
        img = ctx.frame.frame
        for track in ctx.tracks:
            drawables.extend(self._track_drawables(track, ctx.frame))
            
        # for analytics in ctx.analytics:
        #     drawables.extend(self._analysis_drawables(analytics, ctx.frame))

        for drawable in drawables:
            drawable.draw(img)


    def _track_drawables(self, track: Track, frame: Frame) -> List[Drawable]:
        bbox = track.bbox.to_pixel_xyxy(frame.width, frame.height)
        return [
            BBoxDrawable(bbox),
            TextDrawable(f"track {track.id}", position=(bbox[0], bbox[1]))
        ]
    