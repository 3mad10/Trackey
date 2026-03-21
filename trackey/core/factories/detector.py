from trackey.core.registries.detection import DETECTOR_REGISTRY
from trackey.core.interfaces.detector import Detector


def build_detector(detector_type: str, **kwargs) -> Detector:
    cls = DETECTOR_REGISTRY.get(detector_type)
    if cls is None:
        raise ValueError(
            f"Unknown detector '{detector_type}'. "
            f"Available: {list(DETECTOR_REGISTRY.keys())}"
        )

    return cls(**kwargs)
