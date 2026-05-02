from abc import ABC, abstractmethod
from trackey.core.interfaces.subscriber import Subscriber


class SubscriberPlugin(ABC):

    @classmethod
    @abstractmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    @abstractmethod
    def build(cls, cfg: dict) -> Subscriber:
        pass