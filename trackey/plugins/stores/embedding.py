from abc import ABC, abstractmethod
from trackey.core.interfaces.store import EmbeddingStore

class EmbeddingStorePlugin(ABC):
    @classmethod
    @abstractmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    @abstractmethod
    def build(cls, cfg: dict) -> EmbeddingStore:
        pass
