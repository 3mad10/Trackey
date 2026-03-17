from trackey.core.registries.analyzer import ANALYZER_REGISTRY
from trackey.core.interfaces.analyzer import Analyzer
from trackey.data.schemas.geometry import Zone
from trackey.core.scene import Scene


def build_analyzer(analyzer_type: str, scene: Scene =None, **kwargs) -> Analyzer:
    cls = ANALYZER_REGISTRY.get(analyzer_type)
    if cls is None:
        raise ValueError(
            f"Unknown analyzer '{analyzer_type}'. "
            f"Available: {list(ANALYZER_REGISTRY.keys())}"
        )
    
    if "zone" in kwargs:
        if not scene:
            raise ValueError("Must pass scene object if a zone is configured")
        zone = scene.get_zone(kwargs["zone"])
        kwargs["zone"] = Zone.model_validate(zone)

    return cls(**kwargs)