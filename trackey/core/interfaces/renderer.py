from abc import ABC, abstractmethod

from trackey.data.schemas.pipeline import PipelineResult

class Renderer(ABC):
    @abstractmethod
    def open(self) -> bool:
        pass

    @abstractmethod
    def write(self, result: PipelineResult) -> None:
        pass

    @abstractmethod
    def release(self) -> None:
        pass