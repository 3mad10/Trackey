from abc import ABC, abstractmethod
from trackey.core.interfaces.store import EmbeddingRepository


class EmbeddingRepositoryPlugin(ABC):

    @classmethod
    @abstractmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    @abstractmethod
    def build(cls, cfg: dict) -> EmbeddingRepository:
        pass