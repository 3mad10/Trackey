from typing import Any


class PathExtractor:
    def __init__(self, path: str):
        self.path = path  # "analytics.counter.count.car"
        self.parts = path.split(".")

    def extract(self, ctx) -> Any:
        # start from context
        value = ctx
        
        for part in self.parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            
            if value is None:
                return None
        
        return value