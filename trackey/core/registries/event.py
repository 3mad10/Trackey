from typing import Dict, Type
from trackey.data.schemas.event import BaseEvent

EVENT_REGISTRY: Dict[str, Type[BaseEvent]] = {}
