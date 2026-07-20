import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
from uuid import UUID
from datetime import datetime

from trackey.data.schemas.identity import Identity


class IdentificationStore(ABC):

    @abstractmethod
    def search(self, embedding, modality, threshold=0.75) -> Optional[Identity]:
        """Find best matching identity above threshold."""
        pass
    @abstractmethod
    def search_by_text(self, text) -> Optional[Identity]:
        """Find an identity by exact label match."""
        pass
    @abstractmethod
    def register(self, identity) -> None:
        """Create a new identity record."""
        pass
    @abstractmethod
    def upsert(self, identity) -> None:
        """Update an existing identity, or register it if not present."""
        pass
    @abstractmethod
    def load_watchlist(self, watchlist, modality) -> None:
        pass


class EmbeddingStore(ABC):

    @abstractmethod
    def search(self, embeddings: List[np.ndarray], threshold: float)-> List[Dict]:
        pass
    @abstractmethod
    def register(self, ids: List[int], embeddings: List[np.ndarray]) -> None:
        pass
    @abstractmethod
    def update(self, ids: List[int], embeddings: List[np.ndarray]) -> None:
        pass
    @abstractmethod
    def get(self, ids: List[int]) -> List[np.ndarray]:
        pass
    @abstractmethod
    def remove(self, ids: List[int]) -> None:
        pass

class IdentityRepository(ABC):
    """Durable storage for Identity metadata only — label, timestamps.
    Never touches vectors. Symmetric role to EmbeddingStore, different concern."""

    @abstractmethod
    def save(self, identity: Identity) -> None:
        pass
    @abstractmethod
    def load(self, global_id: UUID) -> Optional[Identity]:
        pass
    @abstractmethod
    def load_all(self) -> List[Identity]:
        pass
    @abstractmethod
    def find_by_label(self, label: str) -> Optional[Identity]:
        pass
    @abstractmethod
    def touch(self, global_id: UUID, last_seen: datetime) -> None:
        pass

class EmbeddingRepository(ABC):
    """Durable storage for (int_id, global_id, modality, vector).
    Symmetric to IdentityRepository, but persists vectors, not metadata.
    EmbeddingStore (FAISS/Memory) is always rebuilt from this at startup."""

    @abstractmethod
    def save(self, id: int, global_id: UUID, modality: str, vector: np.ndarray) -> None:
        pass

    @abstractmethod
    def delete(self, id: int, modality: str) -> None:
        pass

    @abstractmethod
    def load_all(self) -> List[Tuple[int, UUID, str, np.ndarray]]:
        pass