from abc import ABC, abstractmethod
from trackey.core.interfaces.sink import OutputSink


class SinkPlugin(ABC):

    @classmethod
    @abstractmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    @abstractmethod
    def build(cls, cfg: dict) -> OutputSink:
        pass