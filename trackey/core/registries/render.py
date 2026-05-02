from typing import Dict, Type
from trackey.core.interfaces.renderer import Renderer

RENDERER_REGISTRY: Dict[str, Type[Renderer]] = {}
