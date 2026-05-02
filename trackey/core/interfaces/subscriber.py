from abc import ABC, abstractmethod

from trackey.data.schemas.event import BaseEvent


class Subscriber(ABC):
    @abstractmethod
    def on_event(self, event: BaseEvent) -> None:
        """
        Receives Event Signal
        Returns Nothing.
        """
        pass

