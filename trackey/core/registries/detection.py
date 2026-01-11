from typing import Dict, Type
from trackey.core.interfaces.detector import Detector

DETECTOR_REGISTRY: Dict[str, Type[Detector]] = {}
