from trackey.core.interfaces.store import IdentityRepository
from trackey.plugins.stores.identity_repository import IdentityRepositoryPlugin
from trackey.core.register import register_identity_repository

@register_identity_repository("memory")
class MemoryRepositoryPlugin(IdentityRepositoryPlugin):
    @classmethod
    def validate(cls, cfg: dict) -> None:
        pass

    @classmethod
    def build(cls, cfg: dict) -> IdentityRepository:
        from trackey.core.stores.metadata.memory import MemoryRepository
        return MemoryRepository()
