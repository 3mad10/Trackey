from abc import ABC, abstractmethod

from trackey.data.schemas.pipeline import PipelineResult

class OutputSink(ABC):
    @abstractmethod
    def open(self) -> bool:
        pass

    @abstractmethod
    def write(self, result: PipelineResult) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass