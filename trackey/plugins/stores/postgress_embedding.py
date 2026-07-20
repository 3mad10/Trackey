from trackey.core.interfaces.store import EmbeddingRepository
from trackey.plugins.stores.embedding_repository import EmbeddingRepositoryPlugin
from trackey.core.register import register_embedding_repository

@register_embedding_repository("postgres")
class PostgresEmbeddingRepositoryPlugin(EmbeddingRepositoryPlugin):
    @classmethod
    def validate(cls, cfg: dict) -> None:
        params = cfg.get("params") or {}
        required = ["host", "port", "database", "user", "password"]
        for r in required:
            if r not in params:
                raise ValueError(f"postgres repository requires '{r}' in params")

    @classmethod
    def build(cls, cfg: dict) -> EmbeddingRepository:
        from trackey.core.stores.embeddings.persist.postgresql import PostgresEmbeddingRepository
        params = cfg.get("params") or {}
        repo_name = cfg.get("table", "identities")
        dsn = f"postgresql://{params.get('user')}:{params.get('password')}@{params.get('host')}:{params.get('port')}/{params.get('database')}"
        return PostgresEmbeddingRepository(repo_name=repo_name, dsn=dsn)
