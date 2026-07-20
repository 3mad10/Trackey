import faiss
from typing import List, Dict
import numpy as np

from trackey.core.interfaces.store import EmbeddingStore
# from trackey.core.register import register_embedding_store


# @register_embedding_store("faiss")
class FaissEmbeddingStore(EmbeddingStore):
    """Raw id <-> vector index for one modality. No identity concept —
    the wrapping IdentificationStore owns global_id <-> int id mapping
    and modality routing."""

    def __init__(self, dims: int, k: int = 2):
        self.dims = dims
        self.k = k
        base_index = faiss.index_factory(dims, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap2(base_index)

    def _to_batch(self, embeddings: List[np.ndarray]) -> np.ndarray:
        if not embeddings:
            raise ValueError("embeddings list must not be empty")
        batch = np.stack(embeddings).astype(np.float32)
        if batch.shape[1] != self.dims:
            raise ValueError(f"expected dim {self.dims}, got {batch.shape[1]}")
        return batch

    def search(self, embeddings: List[np.ndarray], threshold: float = 0.75) -> List[List[Dict]]:
        batch = self._to_batch(embeddings)
        faiss.normalize_L2(batch)

        k = min(self.k, self.index.ntotal) or 1
        scores, indices = self.index.search(batch, k=k)

        results = []
        for i in range(len(batch)):
            query_results = []
            for j in range(k):
                if indices[i, j] == -1:
                    continue
                if scores[i, j] > threshold:
                    query_results.append({"id": int(indices[i, j]), "score": float(scores[i, j])})
            results.append(query_results)
        return results

    def register(self, ids: List[int], embeddings: List[np.ndarray]) -> None:
        batch = self._to_batch(embeddings)
        if len(ids) != len(batch):
            raise ValueError(f"embeddings ({len(batch)}) and ids ({len(ids)}) length mismatch")
        faiss.normalize_L2(batch)
        self.index.add_with_ids(batch, np.asarray(ids, dtype=np.int64))

    def update(self, ids: List[int], embeddings: List[np.ndarray]) -> None:
        ids_array = np.asarray(ids, dtype=np.int64)
        self.index.remove_ids(ids_array)   # drop stale vector(s) before re-adding
        batch = self._to_batch(embeddings)
        if len(ids) != len(batch):
            raise ValueError(f"embeddings ({len(batch)}) and ids ({len(ids)}) length mismatch")
        faiss.normalize_L2(batch)
        self.index.add_with_ids(batch, ids_array)

    def remove(self, ids: List[int]) -> None:
        self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def get(self, ids: List[int]) -> List[np.ndarray]:
        if not isinstance(ids, list):
            raise ValueError("Input to get must be a list of IDs")
        return [self.index.reconstruct(int(i)) for i in ids]


if __name__ == '__main__':
    d = 128
    nb = 100
    nq = 10

    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')

    store = FaissEmbeddingStore(d)
    print(store.search(list(xq)))          # empty index — should return [[], [], ...]

    store.register(list(xb), list(range(nb)))
    res = store.search(list(xq))
    print(res)
    print(res[0][0]['id'] if res[0] else "no match")
    # print(store.get([res[0][0]['id'], res[1][0]['id']]))