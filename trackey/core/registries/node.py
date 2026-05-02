from trackey.core.pipeline.nodes import *

# one registry for all node types
NODE_REGISTRY = {
    # processing
    "detector":      DetectorNode,
    "tracker":       TrackerNode,
    "analyzer":      AnalyzerNode,
    # control
    "spatial_context": SpatialIndexNode,
    "publisher":     PublisherNode,
}