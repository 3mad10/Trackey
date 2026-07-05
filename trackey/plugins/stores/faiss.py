from trackey.core.stores.faiss import FAISS
from trackey.core.interfaces.store import IdentificationStore
from trackey.plugins.stores.store import StorePlugin
from trackey.core.register import register_store


@register_store("faiss")
class FaissPlugin(StorePlugin):

    @classmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    def build(cls, cfg: dict) -> IdentificationStore:
        return FAISS