from trackey.core.stores.embeddings.faiss import FaissEmbeddingStore
from trackey.core.interfaces.store import EmbeddingStore
from trackey.plugins.stores.embedding import EmbeddingStorePlugin
from trackey.core.register import register_embedding_store


@register_embedding_store("faiss")
class FaissPlugin(EmbeddingStorePlugin):

    @classmethod
    def validate(cls, cfg: dict) -> None:
        if "dims" not in cfg:
            raise ValueError("FaissPlugin requires 'dims' in config")

    @classmethod
    def build(cls, cfg: dict) -> EmbeddingStore:
        return FaissEmbeddingStore(dims=cfg.get("dims", 512), k=cfg.get("k", 2))