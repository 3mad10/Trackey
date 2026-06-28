from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Identity:
    global_id:  UUID                    = field(default_factory=dict)
    embeddings: List[np.ndarray]         = field(default_factory=list)
    label:      Optional[str]           = None
    metadata:   Dict[str, Any]          = field(default_factory=dict)
    max_embeddings: int                 = 10     # cap per identity

    def add_embedding(self, embedding: np.ndarray) -> None:
        """Add new embedding, drop oldest if at capacity."""
        if len(self.embeddings) >= self.max_embeddings:
            self.embeddings.pop(0)
        self.embeddings.append(embedding)

    def best_similarity(self, query: np.ndarray) -> float:
        """Return highest cosine similarity against all stored embeddings."""
        if not self.embeddings:
            return 0.0
        similarities = [
            self._cosine_similarity(query, emb)
            for emb in self.embeddings
        ]
        return max(similarities)

    def _cosine_similarity(self, a: np.ndarray,
                            b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))