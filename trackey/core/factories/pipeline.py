import yaml
import logging
from pathlib import Path
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

from trackey.core.registries.detection import DETECTOR_REGISTRY
from trackey.core.registries.tracking  import TRACKER_REGISTRY
from trackey.core.registries.analyzer  import ANALYZER_REGISTRY
from trackey.core.registries.node      import NODE_REGISTRY
from trackey.core.scene.scene          import Scene
from trackey.core.events.bus           import EventBus
from trackey.core.pipeline.edge        import Edge
from trackey.core.interfaces.node      import PipelineNode
from trackey.core.factories.builder    import Builder
from trackey.core.utils.graph          import topological_sort

logger = logging.getLogger(__name__)


class PipelineBuilder(Builder):
    PROCESSING_NODES = ["detector", "tracker", "analyzer", "reid", "postprocessor"]
    CONTROL_NODES    = ["spatial_index", "condition", "switch", "publisher", "branch"]

    PIPELINE_STRUCTURE = (
        "pipeline:\n"
        "  nodes:\n"
        "    - name: <unique-node-name>\n"
        "      type: <detector|tracker|analyzer|spatial_index|condition|publisher>\n"
        "      processor: <implementation>   # processing nodes only\n"
        "      zone_name: <zone>             # analyzer only, optional\n"
        "      params:\n"
        "        param1: value\n"
        "  edges:\n"
        "    - from: <source-node-name>\n"
        "      to:   <target-node-name>\n"
    )

    COMPONENT_REGISTRY = {
        "detector":     DETECTOR_REGISTRY,
        "tracker":      TRACKER_REGISTRY,
        "analyzer":     ANALYZER_REGISTRY,
    }

    # which control nodes need scene injected
    SCENE_NODES = {"spatial_index"}

    # which control nodes need event_bus injected
    EVENT_BUS_NODES = {"publisher"}

    NODE_BUILDERS = {
        # processing
        "detector":      "_build_detector_node",
        "tracker":       "_build_tracker_node",
        "analyzer":      "_build_analyzer_node",
        "reid":          "_build_reid_node",
        "postprocessor": "_build_postprocessor_node",
        # control
        "spatial_index": "_build_spatial_index_node",
        "condition":     "_build_condition_node",
        "switch":        "_build_switch_node",
        "publisher":     "_build_publisher_node",
        "branch":        "_build_branch_node",
    }
                              
    def __init__(self, cfg_path: str,
                 scene:     Scene,
                 event_bus: EventBus):
        self.nodes:     Dict[str, PipelineNode] = {}
        self.edges:     List[Edge]              = []
        self.cfg        = self._load_yaml(cfg_path)
        self.scene      = scene
        self.event_bus  = event_bus

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build(self) -> Tuple[List[PipelineNode], List[Edge]]:
        self._build_nodes()
        self._build_edges()
        self._validate_wiring()
        logger.info(
            f"[PipelineBuilder] Built {len(self.nodes)} nodes "
            f"and {len(self.edges)} edges"
        )
        return list(self.nodes.values()), self.edges

    def get_node(self, node_name: str) -> Optional[PipelineNode]:
        return self.nodes.get(node_name)

    def get_pipeline_order(self) -> List[str]:
        return list(self.nodes.keys())

    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #

    def _build_nodes(self) -> None:
        pipeline_cfg = self.cfg.get("pipeline", {})
        nodes_cfg    = pipeline_cfg.get("nodes", [])

        if not isinstance(nodes_cfg, list):
            raise TypeError(
                f"[PipelineBuilder] pipeline.nodes must be a list.\n"
                f"{self.PIPELINE_STRUCTURE}"
            )

        for node_cfg in nodes_cfg:
            self._validate_node_cfg(node_cfg)
            node = self._build_node(node_cfg)
            self.nodes[node.name] = node

    def _build_edges(self) -> None:
        pipeline_cfg = self.cfg.get("pipeline", {})
        edges_cfg    = pipeline_cfg.get("edges", [])

        for edge_cfg in edges_cfg:
            self._validate_edge_cfg(edge_cfg)
            self.edges.append(Edge(
                source=edge_cfg["from"],
                target=edge_cfg["to"]
            ))
    

    def _build_node(self, node_cfg: dict) -> PipelineNode:
        node_type    = node_cfg["type"]
        builder_name = self.NODE_BUILDERS[node_type]
        builder      = getattr(self, builder_name)
        node         = builder(node_cfg)
        logger.info(
            f"[PipelineBuilder] Built node: "
            f"{node_cfg['name']} ({node_type})"
        )
        return node
    
    # ------------------------------------------------------------------ #
    # Processing node builders                                             #
    # ------------------------------------------------------------------ #

    def _build_detector_node(self, node_cfg: dict):
        from trackey.core.pipeline.nodes import DetectorNode
        component = self._build_component(node_cfg)
        return DetectorNode(name=node_cfg["name"], detector=component)

    def _build_tracker_node(self, node_cfg: dict):
        from trackey.core.pipeline.nodes import TrackerNode
        component = self._build_component(node_cfg)
        return TrackerNode(name=node_cfg["name"], tracker=component)

    def _build_analyzer_node(self, node_cfg: dict):
        from trackey.core.pipeline.nodes import AnalyzerNode
        component = self._build_component(node_cfg)
        return AnalyzerNode(
            name=node_cfg["name"],
            analyzer=component,
            zone_name=node_cfg.get("zone_name")
        )

    def _build_reid_node(self, node_cfg: dict):
        from trackey.core.pipeline.nodes import ReIDNode
        component = self._build_component(node_cfg)
        return ReIDNode(name=node_cfg["name"], reid_model=component)

    def _build_postprocessor_node(self, node_cfg: dict):
        from trackey.core.pipeline.nodes import PostprocessorNode
        component = self._build_component(node_cfg)
        return PostprocessorNode(
            name=node_cfg["name"],
            postprocessor=component
        )

    # ------------------------------------------------------------------ #
    # Control node builders                                                #
    # ------------------------------------------------------------------ #

    def _build_spatial_index_node(self, node_cfg: dict):
        from trackey.core.pipeline.nodes import SpatialIndexNode
        return SpatialIndexNode(
            name=node_cfg["name"],
            scene=self.scene
        )

    def _build_condition_node(self, node_cfg: dict):
        from trackey.core.pipeline.nodes import ConditionNode
        params = node_cfg.get("params") or {}
        return ConditionNode(name=node_cfg["name"], **params)

    def _build_switch_node(self, node_cfg: dict):
        from trackey.core.pipeline.nodes import SwitchNode
        params = node_cfg.get("params") or {}
        return SwitchNode(name=node_cfg["name"], **params)

    def _build_publisher_node(self, node_cfg: dict):
        from trackey.core.pipeline.nodes import PublisherNode
        definitions = self._build_event_definitions(
            node_cfg.get("events", [])
        )
        return PublisherNode(
            name=node_cfg["name"],
            definitions=definitions,
            event_bus=self.event_bus
        )


    # ------------------------------------------------------------------ #
    # Shared component builder                                             #
    # ------------------------------------------------------------------ #

    def _build_component(self, node_cfg: dict):
        node_type  = node_cfg["type"]
        processor  = node_cfg["processor"]
        params     = node_cfg.get("params") or {}

        registry = self.COMPONENT_REGISTRY.get(node_type)
        if registry is None:
            raise ValueError(
                f"[PipelineBuilder] No component registry for: '{node_type}'"
            )

        component_class = registry.get(processor)
        if component_class is None:
            raise ValueError(
                f"[PipelineBuilder] Unknown processor '{processor}' "
                f"for type '{node_type}'. "
                f"Available: {list(registry.keys())}"
            )

        return component_class(**params)

    def _build_event_definitions(self, events_cfg: list) -> list:
        from trackey.core.registries.event import EVENT_REGISTRY
        from trackey.data.schemas.event import EventDefinition

        definitions = []
        for event_cfg in events_cfg:
            event_type = EVENT_REGISTRY.get(event_cfg["type"])
            if not event_type:
                raise ValueError(
                    f"[PipelineBuilder] Unknown event type: "
                    f"'{event_cfg['type']}'. "
                    f"Available: {list(EVENT_REGISTRY.keys())}"
                )
            definitions.append(EventDefinition(
                event_type=event_type,
                extract=event_cfg.get("extract", {})
            ))
        return definitions
    
    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def _validate_node_cfg(self, node_cfg: dict) -> None:
        if not isinstance(node_cfg, dict):
            raise TypeError(
                f"[PipelineBuilder] Each node must be a dict.\n"
                f"{self.PIPELINE_STRUCTURE}"
            )

        # required fields
        for field in ("name", "type"):
            if field not in node_cfg:
                raise ValueError(
                    f"[PipelineBuilder] Node missing '{field}'.\n"
                    f"{self.PIPELINE_STRUCTURE}"
                )

        node_name = node_cfg["name"]
        node_type = node_cfg["type"]

        # duplicate name check
        if node_name in self.nodes:
            raise ValueError(
                f"[PipelineBuilder] Duplicate node name: '{node_name}'"
            )

        # valid type check
        all_types = self.PROCESSING_NODES + self.CONTROL_NODES
        if node_type not in all_types:
            raise ValueError(
                f"[PipelineBuilder] Unknown node type: '{node_type}'. "
                f"Available: {all_types}"
            )

        # type-specific validation
        if node_type in self.PROCESSING_NODES:
            self._validate_processing_node(node_cfg)
        else:
            self._validate_control_node(node_cfg)

    def _validate_processing_node(self, node_cfg: dict) -> None:
        if "processor" not in node_cfg:
            raise ValueError(
                f"[PipelineBuilder] Processing node '{node_cfg['type']}' "
                f"requires 'processor' field.\n"
                f"{self.PIPELINE_STRUCTURE}"
            )

    def _validate_control_node(self, node_cfg: dict) -> None:
        if "processor" in node_cfg:
            raise ValueError(
                f"[PipelineBuilder] Control node '{node_cfg['type']}' "
                f"must not have 'processor' field.\n"
                f"{self.PIPELINE_STRUCTURE}"
            )

    def _validate_edge_cfg(self, edge_cfg: dict) -> None:
        for field in ("from", "to"):
            if field not in edge_cfg:
                raise ValueError(
                    f"[PipelineBuilder] Edge missing '{field}'.\n"
                    f"edges:\n"
                    f"  - from: source_node\n"
                    f"    to:   target_node\n"
                )

        source = edge_cfg["from"]
        target = edge_cfg["to"]

        if source not in self.nodes:
            raise ValueError(
                f"[PipelineBuilder] Edge references unknown "
                f"source node: '{source}'"
            )
        if target not in self.nodes:
            raise ValueError(
                f"[PipelineBuilder] Edge references unknown "
                f"target node: '{target}'"
            )

    def _validate_wiring(self) -> None:
        available = {"frame", "frame_id", "camera_id", "timestamp"}
        order     = topological_sort(list(self.nodes.keys()), self.edges)

        for node_name in order:
            node = self.nodes[node_name]
            for required in node.get_inputs():
                if not self._is_satisfied(required, available):
                    producer = self._find_producer(required)
                    raise ValueError(
                        f"[PipelineBuilder] Node '{node_name}' requires "
                        f"'{required}' which is not available upstream.\n"
                        + (
                            f"Hint: add '{producer}' before '{node_name}'."
                            if producer else
                            f"No node produces '{required}'."
                        )
                    )
            for output in node.get_outputs():
                available.add(output)

    def _is_satisfied(self, required: str, available: set) -> bool:
        if required in available:
            return True
        # prefix match — "analytics.counter" satisfied by "analytics"
        return any(required.startswith(item) for item in available)

    def _find_producer(self, required):
        for node in self.nodes.values():
            for output in node.get_outputs():
                if required.startswith(output):
                    return node.name
        return None



if __name__ == '__main__':
    # Example: ensure detector/tracker classes are imported to register
    from trackey.core.detectors.yolo import YoloDetector
    from trackey.core.features.mediapipe import MPLandmarkDetector
    from trackey.core.trackers.deepsort import DeepSortTracker

    builder = PipelineBuilder('../base_pipeline.yaml')
    print("Pipeline order:", builder.get_pipeline_order())

    # Example: insert a new detector after 'detector1'
    new_node = {
        "name": "detector2",
        "component": "detector",
        "type": "yolo",
        "params": {"weights": "yolov8s.pt"}
    }
    builder.insert_node(new_node, after_node="detector")
    print("Pipeline order after insertion:", builder.get_pipeline_order())
    builder.remove_node(node_name="detector2")
    print("Pipeline order after removal:", builder.get_pipeline_order())
