from typing import Dict, Type
from trackey.plugins.stores.embedding_repository import EmbeddingRepositoryPlugin
from trackey.plugins.stores.identity_repository import IdentityRepositoryPlugin
from trackey.plugins.stores.embedding import EmbeddingStorePlugin

STORE_REGISTRY: Dict[str, Type[EmbeddingRepositoryPlugin]] = {}
REPOSITORY_REGISTRY: Dict[str, Type[IdentityRepositoryPlugin]] = {}
EMBEDDING_STORE_REGISTRY: Dict[str, Type[EmbeddingStorePlugin]] = {}
