from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from uuid import UUID
import numpy as np

from trackey.data.schemas.feature import Embedding


class IdentificationStore(ABC):

    @abstractmethod
    def search(self, embedding: Embedding,
               threshold: float = 0.7) -> Optional[Identity]:
        """Find best matching identity above threshold."""
        pass

    @abstractmethod
    def register(self, embedding: Embedding,
                 label: Optional[str] = None) -> Identity:
        """Create new identity with first embedding."""
        pass

    @abstractmethod
    def update(self, global_id: UUID,
               embedding: np.ndarray) -> None:
        """Add new embedding observation to existing identity."""
        pass

    @abstractmethod
    def get(self, global_id: UUID) -> Optional[Identity]:
        """Retrieve identity by ID."""
        pass

    @abstractmethod
    def load_watchlist(self, watchlist: Dict[str, List[np.ndarray]]) -> None:
        """Pre-load known identities. label → list of embeddings."""
        pass