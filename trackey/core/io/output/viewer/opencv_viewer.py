import cv2
import logging

from trackey.core.interfaces.sink import OutputSink
from trackey.data.schemas.pipeline import PipelineResult
from trackey.core.register import register_sink

logger = logging.getLogger(__name__)


class OpenCVViewer(OutputSink):
    def __init__(self,
                 window_name: str = "Trackey"
                 ):
        """
        Args:
            window_name: OpenCV window name
        """
        self.window_name = window_name
        self.is_open = False

    def open(self) -> bool:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.is_open = True
        return self.is_open
    
    def write(self, result: PipelineResult) -> None:
        if result.rendered_frame is None:
            return
        cv2.imshow(self.window_name, result.rendered_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

    def close(self):
        if self.is_open:
            cv2.destroyWindow(self.window_name)
        self.is_open = False
    

