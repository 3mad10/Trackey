from abc import ABC, abstractmethod
from trackey.data.schemas.track import Track
from trackey.data.schemas.detection import Detection
from typing import TYPE_CHECKING, Annotated, List
import numpy as np

class Frame(ABC):
    frame: np.ndarray

class TrackerBase(ABC):
    @abstractmethod
    def update(detections: List[Detection]) -> List[Track]:
        pass

    @abstractmethod
    def get_tracks() -> List[Track]:
        pass
