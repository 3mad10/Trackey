from abc import ABC, abstractmethod
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.frame import Frame
from typing import TYPE_CHECKING, Annotated, List
import numpy as np


class DetectorBase(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        pass
