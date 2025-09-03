from abc import ABC, abstractmethod
from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection
from typing import List
import numpy as np
from typing import Optional


class Frame(ABC):
    frame: np.ndarray

class TrackerBase(ABC):
    @abstractmethod
    def update(self, detections: List[Detection],
               frame: Optional[Frame] = None) -> List[Track]:
        pass

    @abstractmethod
    def get_tracks(self) -> List[Track]:
        pass
