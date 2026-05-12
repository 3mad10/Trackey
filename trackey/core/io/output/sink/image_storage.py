import cv2
import logging

from trackey.core.interfaces.sink import OutputSink
from trackey.data.schemas.pipeline import PipelineResult

logger = logging.getLogger(__name__)

class ImageStorageSink(OutputSink):
    def __init__(self, path: str):
        self.path = path

    def open(self) -> bool:
        pass
    
    def write(self, result: PipelineResult) -> None:
        if result.raw_frame is None:
            return
        cv2.imwrite(
            f"{self.path}/{result.camera_id}_{result.frame_id}.jpg",
            result.raw_frame.frame
        )

    def close(self):
        pass

