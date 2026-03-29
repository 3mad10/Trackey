from abc import ABC, abstractmethod
from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection
from trackey.data.schemas.frame import Frame
from typing import List
from typing import Optional


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection],
               frame: Optional[Frame] = None) -> List[Track]:
        pass

