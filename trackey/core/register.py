from trackey.core.registries.detection import DETECTOR_REGISTRY
from trackey.core.registries.tracking import TRACKER_REGISTRY
from trackey.core.registries.analyzer import ANALYZER_REGISTRY
from trackey.core.interfaces.detector import Detector
from trackey.core.interfaces.tracker import Tracker
from trackey.core.interfaces.analyzer import Analyzer


def register_detector(name: str):
    def wrapper(cls: type[Detector]):
        if not issubclass(cls, Detector):
            raise TypeError(f"{cls.__name__} must extend Detector")
        DETECTOR_REGISTRY[name] = cls
        return cls
    return wrapper


def register_tracker(name: str):
    def wrapper(cls: type[Tracker]):
        if not issubclass(cls, Tracker):
            raise TypeError(f"{cls.__name__} must extend Tracker")
        TRACKER_REGISTRY[name] = cls
        return cls
    return wrapper


def register_analyzer(name: str):
    def wrapper(cls: type[Analyzer]):
        if not issubclass(cls, Analyzer):
            raise TypeError(f"{cls.__name__} must extend Analyzer")
        ANALYZER_REGISTRY[name] = cls
        return cls
    return wrapper
