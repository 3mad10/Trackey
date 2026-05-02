from typing import Dict, Type
from trackey.core.interfaces.sink import OutputSink

SINK_REGISTRY: Dict[str, Type[OutputSink]] = {}
