from typing import Dict, Type
from trackey.core.interfaces.source import InputSource

SOURCE_REGISTRY: Dict[str, Type[InputSource]] = {}
