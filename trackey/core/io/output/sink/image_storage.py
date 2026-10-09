import cv2
import logging

from trackey.core.interfaces.sink import OutputSink
from trackey.data.schemas.pipeline import PipelineResult

logger = logging.getLogger(__name__)

import os

class ImageStorageSink(OutputSink):
    def __init__(self, path: str):
        self.path = path

    def open(self) -> bool:
        os.makedirs(self.path, exist_ok=True)
        return True
    
    def write(self, result: PipelineResult) -> None:
        if result.raw_frame is None:
            return
        safe_cam_id = str(result.camera_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        cv2.imwrite(
            os.path.join(self.path, f"{safe_cam_id}_{result.frame_id}.jpg"),
            result.raw_frame.frame
        )

    def close(self):
        pass

