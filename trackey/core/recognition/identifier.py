import numpy as np
from datetime import datetime
from uuid import uuid4

from trackey.core.interfaces.store import IdentificationStore
from trackey.data.schemas.identity import Identity


class Identifier:
    """Owns match/register decision for one modality. Store stays dumb."""

    def __init__(self, store: IdentificationStore, modality: str, threshold: float, read_only: bool = False):
        self.store = store
        self.modality = modality
        self.threshold = threshold
        self.read_only = read_only   # True for watchlist-only matching (e.g. FaceRecognitionNode)

    def identify(self, embedding: np.ndarray, now: datetime) -> Identity:
        match = self.store.search(embedding, self.modality, self.threshold)

        if match is not None:
            if not self.read_only:
                evicted = match.add_embedding(self.modality, embedding)
                match.last_seen = now
                self.store.upsert(match, modality=self.modality, added=embedding, evicted=evicted)
            return match

        if self.read_only:
            return None   # watchlist miss — no new identity created

        identity = Identity(global_id=uuid4(), first_seen=now, last_seen=now)
        identity.add_embedding(self.modality, embedding)
        self.store.register(identity)
        return identity