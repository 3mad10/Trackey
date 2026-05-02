from abc import ABC, abstractmethod
from trackey.core.interfaces.source import InputSource


class SourcePlugin(ABC):

    @classmethod
    @abstractmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    @abstractmethod
    def build(cls, cfg: dict) -> InputSource:
        pass