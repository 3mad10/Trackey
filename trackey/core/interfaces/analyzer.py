from abc import ABC, abstractmethod
from typing import List, Dict, Any

from trackey.data.schemas.track import Track


class Analyzer(ABC):
    @abstractmethod
    def analyze(self, tracks: List[Track], frame=None) -> Dict[str, Any]:
        pass
