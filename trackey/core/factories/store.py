import logging
from typing import Dict
from trackey.core.factories.builder import Builder
from trackey.core.registries.store import REPOSITORY_REGISTRY, EMBEDDING_STORE_REGISTRY
from trackey.core.interfaces.store import IdentificationStore, IdentityRepository, EmbeddingStore
from trackey.core.stores.identification import CompositeIdentificationStore


logger = logging.getLogger(__name__)


class StoreBuilder(Builder):
    """
    stores:
      people:
        repository:
          backend: postgres        # sqlite | postgres | mongodb | memory
          table: people_identities  # repo_name — configurable per store
          params:
            host: localhost
            port: 5432
            database: trackey
            user: trackey
            password: ${DB_PASSWORD}
        modalities:
          body: {backend: faiss, dims: 512}
          face: {backend: faiss, dims: 512}
        watchlist:
          path: watchlists/vip.json
          modality: face
    """

    def __init__(self, cfg_path: str):
        self.cfg = self._load_yaml(cfg_path)

    def build(self) -> Dict[str, IdentificationStore]:
        stores_cfg = self.cfg.get("stores", {})
        if not isinstance(stores_cfg, dict):
            raise ValueError("[StoreBuilder] 'stores' must be a dict")

        stores: Dict[str, IdentificationStore] = {}
        for store_name, store_cfg in stores_cfg.items():
            stores[store_name] = self._build_store(store_name, store_cfg)
        return stores

    def _build_store(self, name: str, cfg: dict) -> IdentificationStore:
        if not isinstance(cfg, dict):
            raise ValueError(f"[StoreBuilder] Store '{name}' config must be a dict")

        repository       = self._build_repository(name, cfg.get("repository"))
        embedding_stores = self._build_embedding_stores(name, cfg.get("modalities"))

        embedding_repo_cfg = cfg.get("embedding_repository") or cfg.get("repository")
        embedding_repository = self._build_embedding_repository(name, embedding_repo_cfg) if embedding_repo_cfg else None

        store = CompositeIdentificationStore(embedding_stores=embedding_stores, repository=repository, embedding_repository=embedding_repository)

        watchlist_cfg = cfg.get("watchlist")
        if watchlist_cfg:
            self._load_watchlist(store, watchlist_cfg)

        logger.info(f"[StoreBuilder] Built store: {name} (repo={repository.__class__.__name__}, "
                    f"modalities={list(embedding_stores.keys())})")
        return store

    # ------------------------------------------------------------------ #
    # Repository (identity metadata persistence)                          #
    # ------------------------------------------------------------------ #

    def _build_repository(self, store_name: str, cfg: dict) -> IdentityRepository:
        if not cfg:
            raise ValueError(f"[StoreBuilder] Store '{store_name}' missing 'repository'")

        backend = cfg.get("backend")
        repo_plugin = REPOSITORY_REGISTRY.get(backend)
        if not repo_plugin:
            raise ValueError(
                f"[StoreBuilder] Unknown repository backend '{backend}' for store '{store_name}'. "
                f"Available: {list(REPOSITORY_REGISTRY.keys())}"
            )
            
        repo_plugin.validate(cfg)
        
        # Merge store_name into config as fallback for table
        merged_cfg = {**cfg, "table": cfg.get("table", store_name)}
        return repo_plugin.build(merged_cfg)

    # ------------------------------------------------------------------ #
    # Embedding Repository (vector persistence)                           #
    # ------------------------------------------------------------------ #

    def _build_embedding_repository(self, store_name: str, cfg: dict):
        if not cfg:
            return None
        backend = cfg.get("backend")
        repo_name = cfg.get("table", store_name)
        params = cfg.get("params") or {}
        
        if backend == "postgres":
            dsn = f"postgresql://{params.get('user')}:{params.get('password')}@{params.get('host')}:{params.get('port')}/{params.get('database')}"
            from trackey.core.stores.embeddings.persist.postgresql import PostgresEmbeddingRepository
            return PostgresEmbeddingRepository(table=f"{repo_name}_embeddings", dsn=dsn)
        
        return None

    # ------------------------------------------------------------------ #
    # Embedding stores (one per modality, vector search only)             #
    # ------------------------------------------------------------------ #

    def _build_embedding_stores(self, store_name: str, cfg: dict) -> Dict[str, EmbeddingStore]:
        if not cfg:
            raise ValueError(f"[StoreBuilder] Store '{store_name}' missing 'modalities'")

        stores: Dict[str, EmbeddingStore] = {}
        for modality, modality_cfg in cfg.items():
            backend = modality_cfg.get("backend")
            store_plugin = EMBEDDING_STORE_REGISTRY.get(backend)
            if not store_plugin:
                raise ValueError(
                    f"[StoreBuilder] Unknown embedding backend '{backend}' for "
                    f"store '{store_name}' modality '{modality}'. "
                    f"Available: {list(EMBEDDING_STORE_REGISTRY.keys())}"
                )
            store_plugin.validate(modality_cfg)
            stores[modality] = store_plugin.build(modality_cfg)
        return stores

    # ------------------------------------------------------------------ #
    # Watchlist                                                          #
    # ------------------------------------------------------------------ #

    def _load_watchlist(self, store: IdentificationStore, watchlist_cfg: dict) -> None:
        import json
        import numpy as np

        path     = watchlist_cfg["path"]
        modality = watchlist_cfg["modality"]

        with open(path) as f:
            raw = json.load(f)   # {label: [[float, ...], ...]}

        watchlist = {
            label: [np.array(vec, dtype=np.float32) for vec in vectors]
            for label, vectors in raw.items()
        }
        store.load_watchlist(watchlist, modality)