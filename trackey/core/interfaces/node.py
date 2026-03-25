from abc import ABC, abstractmethod
from typing import List

from trackey.core.context import FrameContext
from trackey.core.interfaces.event import Subscriber

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


class PublisherNode(ABC):
    def __init__(self, subscribers:List[Subscriber]):
        self.subscribers = subscribers

    def add_subscriber(self, subscriber: Subscriber):
        self.subscribers.append(subscriber)
    
    def remove_subscriber(self, subscriber: Subscriber):
        self.subscribers.remove(subscriber)