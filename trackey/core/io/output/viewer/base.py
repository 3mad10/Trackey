from abc import ABC, abstractmethod
from trackey.data.schemas.frame import Frame
from trackey.data.schemas.track import Track
from trackey.data.schemas.view import GlobalStateBox, GlobalStateBoxPlacement
from trackey.data.schemas.pipeline import PipelineResult
from uuid import UUID
from typing import Union, Optional


class OutputViewer(ABC):
    @abstractmethod
    def show(self, frame: Frame, result: PipelineResult) -> None:
        pass

    @abstractmethod
    def open(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass
