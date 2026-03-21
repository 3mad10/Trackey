from abc import ABC, abstractmethod

from trackey.core.context import FrameContext


class PipelineNode(ABC):

    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def process(self, ctx: FrameContext) -> FrameContext:
        """
        Receives a FrameContext (frame, detections, tracks, features, etc.)
        Returns updated FrameContext.
        """
        pass

