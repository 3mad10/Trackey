from abc import ABC, abstractmethod
from trackey.data.schemas.track import Track


class Analyzer(ABC):
    @abstractmethod
    def analyze(self, track: Track):
        pass
