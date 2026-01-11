from typing import Dict, Type
from trackey.core.interfaces.tracker import Tracker

TRACKER_REGISTRY = {}
TRACKER_REGISTRY: Dict[str, Type[Tracker]] = {}
