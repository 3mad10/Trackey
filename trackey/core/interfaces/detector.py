from abc import ABC, abstractmethod
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.frame import Frame
from typing import List


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        pass
