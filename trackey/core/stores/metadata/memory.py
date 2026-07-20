from datetime import datetime
from typing import List, Optional, Dict

from trackey.data.schemas.identity import Identity
from trackey.core.interfaces.store import IdentityRepository


class MemoryRepository(IdentityRepository):
    def __init__(self):
        self._identities: Dict[UUID, Identity] = {}
        self._label_to_identities = {}

    def save(self, identity: Identity) -> None:
        global_id = identity.global_id
        self._identities[global_id] = identity
        if identity.label:
            self._label_to_identities[identity.label] = identity

    def load(self, global_id: UUID) -> Optional[Identity]:
        return self._identities.get(global_id)
    
    def load_all(self) -> List[Identity]:
        return list(self._identities.values())

    def find_by_label(self, label: str) -> Optional[Identity]:
        if not label in self._label_to_identities:
            return None
        return self._label_to_identities[label]

    def touch(self, global_id: UUID, last_seen: datetime) -> None:
        self._identities[global_id].last_seen = last_seen

if __name__ == '__main__':
    from uuid import uuid4
    import numpy as np
    repo = MemoryRepository()
    print(repo.load_all())
    new_id1 = Identity(global_id=uuid4(), label = "Mohamed Emad")
    new_id1.add_embedding(modality = "body", embedding = np.random.randn(128).reshape(-1, 1))
    new_id1_uuid = new_id1.global_id

    new_id2 = Identity(global_id=uuid4(), label = "Ziad Emad")
    new_id2.add_embedding(modality = "body", embedding = np.random.randn(128).reshape(-1, 1))
    new_id2_uuid = new_id2.global_id

    repo.save(new_id1)

    repo.save(new_id2)

    print(repo.load_all())
    print("===============")
    print(repo.load(new_id1_uuid))
    print("===============")
    print(repo.load(new_id2_uuid))
    print("===============")
    print(repo.find_by_label("Ziad Emad"))
    print("===============")
    print(repo.find_by_label("Zsadsdaasd"))
