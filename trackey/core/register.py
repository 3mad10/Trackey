from trackey.core.registries.detection  import DETECTOR_REGISTRY
from trackey.core.registries.tracking   import TRACKER_REGISTRY
from trackey.core.registries.analyzer   import ANALYZER_REGISTRY
from trackey.core.registries.event      import EVENT_REGISTRY
from trackey.core.registries.subscriber import SUBSCRIBER_REGISTRY
from trackey.core.registries.source     import SOURCE_REGISTRY
from trackey.core.registries.sink       import SINK_REGISTRY
from trackey.core.registries.node       import NODE_REGISTRY
from trackey.core.registries.render     import RENDERER_REGISTRY

from trackey.core.interfaces.detector   import Detector
from trackey.core.interfaces.tracker    import Tracker
from trackey.core.interfaces.analyzer   import Analyzer
from trackey.core.interfaces.node       import PipelineNode
from trackey.core.interfaces.renderer   import Renderer
from trackey.data.schemas.event         import BaseEvent

from trackey.plugins.io.source              import SourcePlugin
from trackey.plugins.io.sink                import SinkPlugin
from trackey.plugins.subscribers.subscriber import SubscriberPlugin

from typing import Type


# ------------------------------------------------------------------ #
# Processing components                                                #
# ------------------------------------------------------------------ #

def register_detector(name: str):
    def wrapper(cls: Type[Detector]):
        if not issubclass(cls, Detector):
            raise TypeError(f"{cls.__name__} must extend Detector")
        DETECTOR_REGISTRY[name] = cls
        return cls
    return wrapper


def register_tracker(name: str):
    def wrapper(cls: Type[Tracker]):
        if not issubclass(cls, Tracker):
            raise TypeError(f"{cls.__name__} must extend Tracker")
        TRACKER_REGISTRY[name] = cls
        return cls
    return wrapper


def register_analyzer(name: str):
    def wrapper(cls: Type[Analyzer]):
        if not issubclass(cls, Analyzer):
            raise TypeError(f"{cls.__name__} must extend Analyzer")
        ANALYZER_REGISTRY[name] = cls
        return cls
    return wrapper


# ------------------------------------------------------------------ #
# Pipeline nodes                                                       #
# ------------------------------------------------------------------ #

def register_node(name: str):
    def wrapper(cls: Type[PipelineNode]):
        if not issubclass(cls, PipelineNode):
            raise TypeError(f"{cls.__name__} must extend PipelineNode")
        NODE_REGISTRY[name] = cls
        return cls
    return wrapper


# ------------------------------------------------------------------ #
# Events                                                               #
# ------------------------------------------------------------------ #

def register_event(name: str):
    def wrapper(cls: Type[BaseEvent]):
        if not issubclass(cls, BaseEvent):
            raise TypeError(f"{cls.__name__} must extend BaseEvent")
        EVENT_REGISTRY[name] = cls
        return cls
    return wrapper


# ------------------------------------------------------------------ #
# Subscribers                                                          #
# ------------------------------------------------------------------ #

def register_subscriber(name: str):
    def wrapper(cls: Type[SubscriberPlugin]):
        if not issubclass(cls, SubscriberPlugin):
            raise TypeError(f"{cls.__name__} must extend SubscriberPlugin")
        SUBSCRIBER_REGISTRY[name] = cls
        return cls
    return wrapper


# ------------------------------------------------------------------ #
# Renderer                                                             #
# ------------------------------------------------------------------ #

def register_renderer(name: str):
    def wrapper(cls: Type[Renderer]):
        if not issubclass(cls, Renderer):
            raise TypeError(f"{cls.__name__} must extend Renderer")
        RENDERER_REGISTRY[name] = cls
        return cls
    return wrapper

# ------------------------------------------------------------------ #
# I/O                                                                  #
# ------------------------------------------------------------------ #

def register_source(name: str):
    def wrapper(cls: Type[SourcePlugin]):
        if not issubclass(cls, SourcePlugin):
            raise TypeError(f"{cls.__name__} must extend SourcePlugin")
        SOURCE_REGISTRY[name] = cls
        return cls
    return wrapper


def register_sink(name: str):
    def wrapper(cls: Type[SinkPlugin]):
        if not issubclass(cls, SinkPlugin):
            raise TypeError(f"{cls.__name__} must extend SinkPlugin")
        SINK_REGISTRY[name] = cls
        return cls
    return wrapper