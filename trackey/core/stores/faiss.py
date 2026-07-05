import faiss
from uuid import UUID
from typing import List, Union, Dict
import numpy as np

from trackey.data.schemas.feature import Embedding
from trackey.core.interfaces.store import IdentificationStore


class FAISS(IdentificationStore):
    def __init__(self, dims, k = 2):
        base_index = faiss.index_factory(dims, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap2(base_index)
        self.k = k

    def search(self, embeddings: List[Union[UUID, int]], threshold: float = 0.75):
        embeddings_array = np.asarray(embeddings, dtype=np.float32)

        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)

        faiss.normalize_L2(embeddings_array)

        distances, indices = self.index.search(embeddings_array, k=self.k)

        results = []

        for i in range(len(embeddings_array)):
            query_results = []

            for j in range(self.k):
                if distances[i, j] > threshold:
                    query_results.append({
                        "id": int(indices[i, j]),
                        "score": float(distances[i, j])
                    })

            results.append(query_results)

        return results

    def register(self, embeddings, ids):
        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        self.index.add_with_ids(
            embeddings_array,
            np.asarray(ids, dtype=np.int64)
        )

    def update(self, ids, embeddings):
        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        self.index.add_with_ids(
            embeddings_array,
            np.asarray(ids, dtype=np.int64)
        )

    def get(self, ids: List[Union[UUID, int]]) -> List[Embedding]:
        """Retrieve identity by ID."""
        embeddings = []
        if not isinstance(ids, List):
            raise ValueError("Input to get must be a list of IDs") 
        for id in ids:
            embeddings.append(self.index.reconstruct(id))
        return embeddings


if __name__ == '__main__':
    d = 128
    nb = 100
    nq = 10

    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')

    store = FAISS(d)
    print(store.search(xq))
    
    store.register(xb, np.array(np.arange(nb)).tolist())
    res = store.search(xq)
    print(res)
    print(res[0][0]['id'])
    
    print(store.get([res[0][0]['id'], res[1][0]['id']]))