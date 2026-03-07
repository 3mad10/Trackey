import cv2
import numpy as np
from uuid import UUID
from typing import Union, Optional, List

from trackey.core.io.output.viewer.base import OutputViewer
from trackey.core.io.output.viewer.drawable import BBoxDrawable, PointDrawable, detection_to_drawables, track_to_drawables
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.view import GlobalStateBox, GlobalStateBoxPlacement



class OpenCVViewer(OutputViewer):
    def __init__(self, window_name: str = "Trackey", wait_ms: int = 1):
        self.window_name = window_name
        self.is_open = False

    def _render(self, img):
        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt

    def show(self, frame: Optional[Frame], data: dict):
        if frame is None:
            return

        img = frame.frame.copy()

        detections = data.get("detections", [])

        for det in detections:
            print("Detection : ", det)
            print("class_name : ", det.class_name)
            drawables = detection_to_drawables(det, frame)
            for drawable in drawables:
                drawable.draw(img)

        tracks = data.get("tracks", [])
        for track in tracks:
            if not track.view_track or not track.detections:
                continue
            det = track.detections[-1]
            for drawable in detection_to_drawables(det, frame):
                drawable.draw(img)

        analytics = data.get("analytics", {})
        # draw activities, counters, heatmaps, etc.

        self._render(img)

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
