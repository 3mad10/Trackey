import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class Identity:
    global_id:  UUID
    embeddings: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    label:      Optional[str] = None
    metadata:   Dict[str, str] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_embedding(self, modality: str, embedding: np.ndarray, max: int = 10) -> Optional[np.ndarray]:
        """Returns the evicted embedding, if FIFO eviction occurred, else None."""
        bucket = self.embeddings.setdefault(modality, [])
        bucket.append(embedding)
        evicted = None
        if len(bucket) > max:
            evicted = bucket.pop(0)
        return evicted

    def best_similarity(self, modality: str, query: np.ndarray) -> float:
        bucket = self.embeddings.get(modality, [])
        if not bucket:
            return 0.0
        return max(float(np.dot(query, e) / (np.linalg.norm(query) * np.linalg.norm(e))) for e in bucket)

@dataclass
class IdentityCandidate:
    global_id: UUID
    conf: float = 0.0
