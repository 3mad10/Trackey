from .detector import build_detector
from .tracker import build_tracker
from .analyzer import build_analyzer
from trackey.core.pipeline.nodes import (
    DetectorNode,
    TrackerNode,
    AnalyzerNode,
    PostprocessorNode,
    ReIDNode,
    SpatialIndexNode,
)

FACTORY_ROUTER = {
    "detector": build_detector,
    "tracker": build_tracker,
    "analyzer": build_analyzer,
}


NODE_WRAPPERS = {
    "detector": lambda name, component, cfg: DetectorNode(name, component),
    "tracker": lambda name, component, cfg: TrackerNode(name, component),
    "analyzer": lambda name, component, cfg: AnalyzerNode(
        name=name,
        analyzer=component,
        node_cfg=cfg
    ),
    "postprocessor": lambda name, component, cfg: PostprocessorNode(name, component),
    "reid": lambda name, component, cfg: ReIDNode(name, component),

}

CONTROL_NODES = {
    "spatial_context": lambda name, scene: SpatialIndexNode(name, scene),
}
