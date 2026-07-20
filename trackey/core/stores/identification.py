from typing import Dict, List, Tuple, Optional
from uuid import UUID, uuid4
import numpy as np

from trackey.core.interfaces.store import IdentificationStore, EmbeddingStore, IdentityRepository, EmbeddingRepository
from trackey.data.schemas.identity import Identity


class CompositeIdentificationStore(IdentificationStore):
    def __init__(self, embedding_stores: Dict[str, EmbeddingStore], repository: IdentityRepository,
                 embedding_repository: Optional[EmbeddingRepository] = None):
        self._embedding_stores = embedding_stores
        self._repository = repository
        self._embedding_repository = embedding_repository
        self._int_to_global: Dict[str, Dict[int, UUID]] = {} # modality -> int to global
        self._embedding_ids: Dict[UUID, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
        self._next_int_id: Dict[str, int] = {}

        if self._embedding_repository is not None:
            self._rebuild_from_repository()

    def search(self, embedding, modality, threshold):
        store = self._embedding_stores.get(modality)
        if store is None:
            return None
        results = store.search([embedding], threshold=threshold)[0]
        if not results:
            return None
        best = max(results, key=lambda r: r["score"])
        global_id = self._int_to_global.get(modality, {}).get(best["id"])
        return self._repository.load(global_id) if global_id else None

    def register(self, identity: Identity) -> None:
        self._repository.save(identity)
        self._embedding_ids[identity.global_id] = {}
        for modality, embs in identity.embeddings.items():
            self._add_embeddings(identity.global_id, modality, embs)

    def upsert(self, identity: Identity, modality: str = None, added=None, evicted=None) -> None:
        self._repository.save(identity)
        if modality is None:
            return
        vec_store = self._embedding_stores.get(modality)
        if vec_store is None:
            raise ValueError(f"No EmbeddingStore configured for modality '{modality}'")

        pairs = self._embedding_ids.setdefault(identity.global_id, {}).setdefault(modality, [])

        if evicted is not None:
            evicted_id = self._find_id_for_vector(pairs, evicted)
            if evicted_id is not None:
                vec_store.remove([evicted_id])
                self._int_to_global.get(modality, {}).pop(evicted_id, None)
                pairs[:] = [(i, v) for i, v in pairs if i != evicted_id]
                if self._embedding_repository is not None:
                    self._embedding_repository.delete(evicted_id, modality)

        if added is not None:
            new_id = self._next_int_id.setdefault(modality, 0)
            self._next_int_id[modality] += 1
            vec_store.register([new_id], [added])
            self._int_to_global.setdefault(modality, {})[new_id] = identity.global_id
            pairs.append((new_id, added))
            if self._embedding_repository is not None:
                self._embedding_repository.save(new_id, identity.global_id, modality, added)
            
    def get(self, global_id):
        return self._repository.load(global_id)

    def search_by_text(self, text):
        return self._repository.find_by_label(text)

    def load_watchlist(self, watchlist, modality):
        for label, embeddings in watchlist.items():
            identity = Identity(global_id=uuid4(), label=label)
            for e in embeddings:
                identity.add_embedding(modality, e)
            self.register(identity)
    
    def _find_id_for_vector(self, pairs: List[Tuple[int, np.ndarray]], target: np.ndarray) -> Optional[int]:
        for int_id, vec in pairs:
            if np.array_equal(vec, target):
                return int_id
        return None
    
    def _rebuild_from_repository(self):
        max_id_per_modality: Dict[str, int] = {}

        for id_, global_id, modality, vector in self._embedding_repository.load_all():
            store = self._embedding_stores.get(modality)
            if store is None:
                continue   # modality no longer configured — skip, don't crash startup

            store.register([id_], [vector])

            self._int_to_global.setdefault(modality, {})[id_] = global_id
            self._embedding_ids.setdefault(global_id, {}).setdefault(modality, []).append((id_, vector))

            max_id_per_modality[modality] = max(max_id_per_modality.get(modality, -1), id_)

        for modality in self._embedding_stores:
            self._next_int_id[modality] = max_id_per_modality.get(modality, -1) + 1
    
    def _add_embeddings(self, global_id: UUID, modality: str, embs: List[np.ndarray]) -> None:
        vec_store = self._embedding_stores.get(modality)
        if vec_store is None:
            raise ValueError(f"No EmbeddingStore configured for modality '{modality}'")
        pairs = self._embedding_ids.setdefault(global_id, {}).setdefault(modality, [])
        for emb in embs:
            new_id = self._next_int_id.setdefault(modality, 0)
            self._next_int_id[modality] += 1
            vec_store.register([new_id], [emb])
            self._int_to_global.setdefault(modality, {})[new_id] = global_id
            pairs.append((new_id, emb))
            if self._embedding_repository is not None:
                self._embedding_repository.save(new_id, global_id, modality, emb)

    
if __name__ == '__main__':
    import numpy as np
    from trackey.core.stores.embeddings.faiss import FaissEmbeddingStore
    from trackey.core.stores.metadata.postgresql import PostgresIdentityRepository
    from trackey.core.stores.embeddings.persist.postgresql import PostgresEmbeddingRepository

    d_body = 128
    d_face = 512
    nb = 100
    nq = 10

    xb_body = np.random.random((nb, d_body)).astype('float32')
    xq_body = np.random.random((nq, d_body)).astype('float32')

    xb_face = np.random.random((nb, d_face)).astype('float32')
    xq_face = np.random.random((nq, d_face)).astype('float32')

    b_store = FaissEmbeddingStore(d_body)
    f_store = FaissEmbeddingStore(d_face)
    # print(len(list(xb_body)))
    # b_store.register(np.array(np.arange(nb)).tolist(), list(xb_body))
    # f_store.register(np.array(np.arange(nb)).tolist(), list(xb_face))
    # res = b_store.search(xq_body)
    # print(res)
    # print(res[0][0]['id'])

    embedding_stores = {
        'body' : b_store,
        'face' : f_store
    }
    uuid_1 = uuid4()
    con = "postgresql://trackey:password@localhost:5432/trackey"
    metadata_repo = PostgresIdentityRepository(repo_name="people" , dsn=con)
    embedding_repository = PostgresEmbeddingRepository(dsn=con, table="people_body_embedding")
    comp = CompositeIdentificationStore(embedding_stores=embedding_stores, repository=metadata_repo, embedding_repository=embedding_repository)
    comp.register(Identity(uuid_1, ))