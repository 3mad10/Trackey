from .detector import build_detector
from .tracker import build_tracker
from .analyzer import build_analyzer
from trackey.core.pipeline.nodes import (
    DetectorNode,
    TrackerNode,
    AnalyzerNode,
    PostprocessorNode,
    ReIDNode,
)

FACTORY_ROUTER = {
    "detector": build_detector,
    "tracker": build_tracker,
    "analyzer": build_analyzer,
}


NODE_WRAPPERS = {
    "detector": lambda component, cfg: DetectorNode(component),
    "tracker": lambda component, cfg: TrackerNode(component),
    "analyzer": lambda component, cfg: AnalyzerNode(
        analyzer=component,
        key=cfg["name"]
    ),
    "postprocessor": lambda component, cfg: PostprocessorNode(component),
    "reid": lambda component, cfg: ReIDNode(component),
}
