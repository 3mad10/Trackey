from typing import Dict, Type
from trackey.core.interfaces.extractor import FeatureExtractor

REID_REGISTRY: Dict[str, Type[FeatureExtractor]] = {}
