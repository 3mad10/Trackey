from abc import ABC, abstractmethod
from typing import List

from trackey.core.context import FrameContext
from trackey.core.interfaces.subscriber import Subscriber

class PipelineNode(ABC):

    @abstractmethod
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def process(self, ctx: FrameContext) -> FrameContext:
        """
        Receives a FrameContext (frame, detections, tracks, features, etc.)
        Returns updated FrameContext.
        """
        pass


class PublisherNode(PipelineNode):
    def __init__(self, name: str, subscribers:List[Subscriber]):
        super().__init__(name=name)
        self.subscribers = subscribers

    def add_subscriber(self, subscriber: Subscriber):
        self.subscribers.append(subscriber)
    
    def remove_subscriber(self, subscriber: Subscriber):
        self.subscribers.remove(subscriber)