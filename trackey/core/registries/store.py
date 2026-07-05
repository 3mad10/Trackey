from typing import Dict, Type
from trackey.data.schemas.identity import Identity

STORE_REGISTRY: Dict[str, Type[Identity]] = {}
