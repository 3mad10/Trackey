import cv2
import logging
import numpy as np
from uuid import UUID
from typing import Union, Optional, List

from trackey.core.io.output.viewer.base import OutputViewer
from trackey.core.scene.scene import Scene
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.pipeline import PipelineResult
from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.view import GlobalStateBox, GlobalStateBoxPlacement
from trackey.core.io.output.viewer.drawable import (
    BBoxDrawable, 
    PointDrawable, 
    PolygonDrawable,
    LineDrawable,
    detection_to_drawables, 
    track_to_drawables
)


logger = logging.getLogger(__name__)

class OpenCVViewer(OutputViewer):
    def __init__(self,
                 window_name: str = "Trackey",
                 wait_ms: int = 1,
                 scene: Scene = None,
                 show_scene: bool = True,
                 show_lines: bool = True):
        """
        Args:
            window_name: OpenCV window name
            wait_ms: Milliseconds to wait between frames
            show_zones: If True, draw analyzer zones/areas of effect
        """
        if show_scene and not scene:
            logger.error(f"[OutputViewer][OpenCVViewer] scene shall be viewed but scene is not passed as input")
        self.window_name = window_name
        self.is_open = False
        self.scene = scene
        self.show_scene = show_scene
        self.static_layer = None


    def show(self, frame: Optional[Frame], data: PipelineResult):
        if frame is None:
            return

        base = frame.frame.copy()
    
        if self.static_layer is None and self.show_scene:
            self._build_static_layer(frame)

        img = base
        if self.static_layer is not None:
            img = cv2.addWeighted(img,1,self.static_layer,1,0)

        detections = data.detections
        for det in detections:
            # print("Detection : ", det)
            # print("class_name : ", det.class_name)
            drawables = detection_to_drawables(det, frame)
            for drawable in drawables:
                drawable.draw(img)

        tracks = data.tracks
        for track in tracks:
            if not track.view_track or not track.detections:
                continue
            det = track.detections[-1]
            for drawable in detection_to_drawables(det, frame):
                drawable.draw(img)

        analytics = data.analytics
        # draw activities, counters, heatmaps, etc.

        self._render(img)

    
    def _build_static_layer(self, frame):

        self.static_layer = np.zeros_like(frame.frame)
        if not self.scene:
            return
        for zone in self.scene.zones.values():
            PolygonDrawable(
                points=zone.polygon.points,
                frame_width=frame.width,
                frame_height=frame.height,
                color=zone.color,
                filled=zone.filled,
                alpha=zone.alpha,
                label=zone.name
            ).draw(self.static_layer)
        for line in self.scene.lines.values():

            LineDrawable(
                line=line,
                frame_width=frame.width,
                frame_height=frame.height
            ).draw(self.static_layer)
    def add_global_state_box(self, placement: GlobalStateBoxPlacement) -> UUID:
        pass

    def add_global_state(self, global_state_box: UUID, state_name: str, value: Union[int, float, str]):
        pass

    def open(self) -> bool:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.is_open = True
        return self.is_open

    def close(self):
        if self.is_open:
            cv2.destroyWindow(self.window_name)
        self.is_open = False


    def _render(self, img):
            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt


    def _draw_zones(self, img: np.ndarray, zones, width: int, height: int):
        """Draw analyzer zones/areas of effect"""
        for name, zone_info in zones.items():
            # print("=======================")
            # print(name)
            # print(zone_info)
            drawable = PolygonDrawable(
                points=zone_info.polygon.points,
                frame_width=width,
                frame_height=height,
                color=zone_info.color,
                filled=zone_info.filled,
                alpha=zone_info.alpha,
                label=zone_info.name,
                thickness=2
            )
            drawable.draw(img)
        
