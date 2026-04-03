from typing import Any

from trackey.core.context import FrameContext

class PathExtractor:
    def __init__(self, path: str):
        self.parts = path.split(".")
    
    def extract(self, ctx: FrameContext) -> Any:
        value = ctx
        for part in self.parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                return None
        return value