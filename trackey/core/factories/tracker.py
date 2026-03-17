from trackey.core.registries.tracking import TRACKER_REGISTRY
from trackey.core.interfaces.tracker import Tracker
from trackey.core.scene import Scene


def build_tracker(tracker_type: str, scene: Scene =None,  **kwargs) -> Tracker:
    cls = TRACKER_REGISTRY.get(tracker_type)
    if cls is None:
        raise ValueError(
            f"Unknown detector '{tracker_type}'. "
            f"Available: {list(TRACKER_REGISTRY.keys())}"
        )

    return cls(**kwargs)
