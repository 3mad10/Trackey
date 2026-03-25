from abc import ABC, abstractmethod

from trackey.data.schemas.event import Event


class Subscriber(ABC):
    @abstractmethod
    def on_event(self, event: Event) -> None:
        """
        Receives Event Signal
        Returns Nothing.
        """
        pass

