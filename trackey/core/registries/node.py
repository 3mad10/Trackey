from trackey.core.pipeline.nodes import *
from trackey.core.logic import *

# one registry for all node types
NODE_REGISTRY = {
    # processing
    "detector":      DetectorNode,
    "tracker":       TrackerNode,
    "analyzer":      AnalyzerNode,
    # control
    "spatial_context": SpatialIndexNode,
    "conditional":   ConditionalNode,
    # "publisher":     PublisherNode,
    # "branch":        BranchNode,
}