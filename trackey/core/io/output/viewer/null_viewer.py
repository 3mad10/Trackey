import cv2
import logging
import numpy as np
from uuid import UUID
from typing import Union, Optional, List

from trackey.core.interfaces.sink import OutputSink
from trackey.data.schemas.pipeline import PipelineResult

logger = logging.getLogger(__name__)

class NullViewer(OutputSink):
    def __init__(self):
        pass

    def open(self) -> bool:
        return True
    
    def write(self, result: PipelineResult) -> None:
        pass

    def close(self):
        pass
