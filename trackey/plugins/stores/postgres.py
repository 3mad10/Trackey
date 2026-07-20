from trackey.core.interfaces.store import IdentityRepository
from trackey.plugins.stores.identity_repository import IdentityRepositoryPlugin
from trackey.core.register import register_identity_repository

@register_identity_repository("postgres")
class PostgresRepositoryPlugin(IdentityRepositoryPlugin):
    @classmethod
    def validate(cls, cfg: dict) -> None:
        params = cfg.get("params") or {}
        required = ["host", "port", "database", "user", "password"]
        for r in required:
            if r not in params:
                raise ValueError(f"postgres repository requires '{r}' in params")

    @classmethod
    def build(cls, cfg: dict) -> IdentityRepository:
        from trackey.core.stores.metadata.postgresql import PostgresIdentityRepository
        params = cfg.get("params") or {}
        repo_name = cfg.get("table", "identities")
        dsn = f"postgresql://{params.get('user')}:{params.get('password')}@{params.get('host')}:{params.get('port')}/{params.get('database')}"
        return PostgresIdentityRepository(repo_name=repo_name, dsn=dsn)
