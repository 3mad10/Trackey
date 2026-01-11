from trackey.core.registries.tracking import TRACKER_REGISTRY
from trackey.core.interfaces.tracker import Tracker


def build_tracker(tracker_type: str, **tracker_args) -> Tracker:
    cls = TRACKER_REGISTRY.get(tracker_type)
    if cls is None:
        raise ValueError(
            f"Unknown detector '{tracker_type}'. "
            f"Available: {list(TRACKER_REGISTRY.keys())}"
        )

    return cls(**tracker_args)
