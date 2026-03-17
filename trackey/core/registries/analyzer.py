from typing import Dict, Type
from trackey.core.interfaces.analyzer import Analyzer

ANALYZER_REGISTRY: Dict[str, Type[Analyzer]] = {}
