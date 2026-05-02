from typing import Dict, Type
from trackey.core.interfaces.subscriber import Subscriber

SUBSCRIBER_REGISTRY: Dict[str, Type[Subscriber]] = {}
