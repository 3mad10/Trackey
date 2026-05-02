from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from trackey.core.context import FrameContext


@dataclass
class PipelineNode(ABC):
    name: str

    @abstractmethod
    def process(self, ctx: FrameContext) -> FrameContext:
        pass

    @abstractmethod
    def get_inputs(self) -> List[str]:
        pass

    @abstractmethod
    def get_outputs(self) -> List[str]:
        pass

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, PipelineNode) and self.name == other.name

