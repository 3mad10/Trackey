from .base import OutputViewer
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.track import Track
from trackey.data.schemas.view import GlobalStateBox, GlobalStateBoxPlacement
import cv2
import numpy as np
from uuid import UUID, uuid4
from typing import Optional


class NullViewer(OutputViewer):

    def open(self) -> bool:
        return True

    def show(
        self,
        frame: Optional[Frame],
        tracks: list[Track]
    ):
        # Intentionally do nothing
        pass


    def close(self):
        pass
