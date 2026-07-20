from .detector import build_detector
from .tracker import build_tracker
from .analyzer import build_analyzer
from .builder import Builder
from .event_bus import EventBusBuilder
from .pipeline import PipelineBuilder
from .scene import SceneBuilder
from .renderer import RendererBuilder
from .sink import SinkBuilder
from .source import SourceBuilder
from .store import StoreBuilder
from trackey.core.pipeline.nodes import (
    DetectorNode,
    TrackerNode,
    AnalyzerNode,
    PostprocessorNode,
    EmbeddingNode,
    SpatialIndexNode,
    PublisherNode,
)

FACTORY_ROUTER = {
    "detector": build_detector,
    "tracker": build_tracker,
    "analyzer": build_analyzer,
}


NODE_WRAPPERS = {
    "detector": lambda node_name, component, cfg: DetectorNode(node_name, component),
    "tracker": lambda node_name, component, cfg: TrackerNode(node_name, component),
    "analyzer": lambda node_name, component, cfg: AnalyzerNode(
        node_name,
        component,
        **cfg
    ),
    "postprocessor": lambda node_name, component, cfg: PostprocessorNode(node_name, component),
    "reid": lambda node_name, component, cfg: EmbeddingNode(node_name, component),

}

CONTROL_NODES = {
    "spatial_context": lambda node_name, scene, cfg: SpatialIndexNode(node_name, scene),
    "publisher": lambda node_name, scene, cfg: PublisherNode(node_name, cfg.subscribers),
}

__all__ = [
    "SourceBuilder",
    "SceneBuilder",
    "PipelineBuilder",
    "EventBusBuilder",
    "RendererBuilder",
    "SinkBuilder",
    "StoreBuilder",
]
