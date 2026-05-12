from abc import ABC, abstractmethod
import numpy as np

from trackey.data.schemas.frame import Frame
from trackey.core.context import FrameContext


class Renderer(ABC):
    @abstractmethod
    def initialize(self, frame: Frame) -> None:
        pass

    @abstractmethod
    def render(self, ctx: FrameContext) -> np.ndarray:
        pass
