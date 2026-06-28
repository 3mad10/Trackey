from abc import ABC, abstractmethod
import numpy as np
from typing import List

from trackey.data.schemas.track import Track
from trackey.data.schemas.frame import Frame


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, tracks: List[Track], frame: Frame) -> List[np.ndarray]:
        pass
