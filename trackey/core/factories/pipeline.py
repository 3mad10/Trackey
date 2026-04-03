import yaml
import logging
from pathlib import Path
from trackey.core.registries.detection import DETECTOR_REGISTRY
from trackey.core.registries.tracking import TRACKER_REGISTRY
from trackey.core.registries.analyzer import ANALYZER_REGISTRY
from trackey.core.registries.node import NODE_REGISTRY
from trackey.core.scene.scene import Scene
from trackey.data.schemas.pipeline import NodeCfg

logger = logging.getLogger(__name__)

class PipelinBuilder:
    PROCESSING_NODES = ['detector', 'tracker', 'analyzer']
    CONTROL_NODES = ['spatial_context', 'branch', 'conditional']
    PIPELINE_STRUCUTURE = f"pipeline:\
                            - name: <unique-node-name> \
                              type: <catagory-of-node> \
                              <if node of type {PROCESSING_NODES}> \
                              processor: <actual-implementation>\
                              params: \
                                param1: <param-value> \
                                param2: <param-value> \
                              <end if> \
                            - name: <unique-node-name> \
                              type: <catagory-of-node>"
    COMPONENT_REGISTRY = {
        "detector": DETECTOR_REGISTRY,
        "tracker": TRACKER_REGISTRY,
        "analyzer": ANALYZER_REGISTRY,
    }
                              
    def __init__(self, cfg_path: str, scene: Scene):
        self.nodes = []
        self.cfg = self._load_yaml(cfg_path)
        self.scene = scene

    def build(self):
        self._build_pipeline()
        return self.nodes

    def _load_yaml(self, cfg_path: str):
        cfg_path = Path(cfg_path)

        if not cfg_path.exists():
            logger.error(f"[PipelineBuilder] Config file not found: {cfg_path.resolve()}")
            raise FileNotFoundError(f"[PipelineBuilder] Config file not found: {cfg_path.resolve()}")

        with cfg_path.open("r") as f:
            return yaml.safe_load(f)

    def _build_pipeline(self):
        """Build all nodes from the loaded YAML in order."""
        pipeline = self.cfg.get("pipeline", [])
        if not isinstance(pipeline, list):
            raise TypeError(f"[PipelineBuilder] Pipeline must be a list of nodes format of yaml must be \
                            {self.PIPELINE_STRUCUTURE}")

        for node_cfg in pipeline:
            self._build_node(node_cfg)

    def _build_node(self, node_cfg: NodeCfg):

        self._validate_node_cfg(node_cfg)

        node_type = node_cfg["type"]
        node_name = node_cfg["name"]
        if node_type in self.PROCESSING_NODES:
            # processing node
            node = self._build_processing_node(node_cfg)

        else:
            # control node
            node = self._build_control_node(node_cfg)

        self.nodes.append(node)
        logger.info(f"[PipelineBuilder] Built node: {node_name} ({node_type})")


    def _validate_node_cfg(self, node_cfg: NodeCfg) -> None:
        if "name" not in node_cfg or "type" not in node_cfg:
            logger.error(f"[PipelineBuilder] Node must have name AND type attributes : \
                             {self.PIPELINE_STRUCUTURE}")
            raise ValueError(f"[PipelineBuilder] Node must have name AND type attributes : \
                             {self.PIPELINE_STRUCUTURE}")
        
        node_name = node_cfg["name"]
        node_type = node_cfg["type"]

        if any(node.name == node_name for node in self.nodes):
            logger.error(f"[PipelineBuilder] Duplicate node name detected: '{node_name}'")
            raise ValueError(f"[PipelineBuilder] Duplicate node name detected: '{node_name}'")

        if node_type in self.PROCESSING_NODES:
            self._validate_processing_node(node_cfg)
        elif node_type in self.CONTROL_NODES:
            self._validate_control_node(node_cfg)
        else:
            logger.error(f"[PipelineBuilder] Invalid Node type: '{node_type}' \
                         Available node types: {self.PROCESSING_NODES + self.CONTROL_NODES}")
            raise ValueError(f"[PipelineBuilder] Duplicate node name detected: '{node_name}'")

    def _validate_processing_node(self, node_cfg: NodeCfg) -> None:
        if "processor" not in node_cfg:
            logger.error(f"[PipelineBuilder] Attribute processor must be added for node of type: '{node_cfg['type']}'")
            raise ValueError(f"[PipelineBuilder] Attribute processor must be added for node of type: '{node_cfg['type']}'")
    
    def _validate_control_node(self, node_cfg: NodeCfg) -> None:
        if "processor" in node_cfg:
            logger.error(f"[PipelineBuilder] Attribute processor must NOT be added for node of type: '{node_cfg['type']}'")
            raise ValueError(f"[PipelineBuilder] Attribute processor must NOT be added for node of type: '{node_cfg['type']}'")
    
    def _build_processing_node(self, node_cfg: NodeCfg):
        node_name = node_cfg["name"]
        node_type = node_cfg["type"]
        
        params = node_cfg.get("params") or {}
        processor = node_cfg["processor"]
        
        # node-level params (zone_name, event_name, etc.)
        node_params = {
            k: v for k, v in node_cfg.items()
            if k not in ("name", "type", "processor", "params")
        }
        
        component_class = self.COMPONENT_REGISTRY[node_type][processor]
        component = component_class(**params)
        
        node_class = NODE_REGISTRY[node_type]
        node = node_class(node_name, component, **node_params)
        return node

    def _build_control_node(self, node_cfg: NodeCfg):
        node_name = node_cfg["name"]
        node_type = node_cfg["type"]

        # node-level params (zone_name, event_name, etc.)
        node_params = {
            k: v for k, v in node_cfg.items()
            if k not in ("name", "type")
        }
        node_class = NODE_REGISTRY[node_type]
        node = node_class(name=node_name, scene=self.scene, **node_params)
        return node

    def _remove_node_by_position(self, pipeline, position):
        if not isinstance(position, int):
            raise TypeError("position must be an integer")

        if position < 0 or position >= len(self.nodes):
            raise IndexError("position out of range")

        removed_node = self.nodes.pop(position)
        pipeline.pop(position)

        print(f"[PipelineBuilder] Removed node at position {position}: {removed_node[0]}")

    def _remove_node_by_name(self, pipeline, node_name):
        if not isinstance(node_name, str):
            raise TypeError("node_name must be a string")

        for idx, (name, _) in enumerate(self.nodes):
            if name == node_name:
                self.nodes.pop(idx)
                pipeline.pop(idx)
                print(f"[PipelineBuilder] Removed node '{node_name}'")
                return

        raise ValueError(f"[PipelineBuilder] node '{node_name}' not found in pipeline.")

    def insert_node(self, new_node_cfg, position=None, after_node=None):
        """
        Insert a node at a specific position in the pipeline and nodes list.
        If after_node and position are not passed the node is added to the end of the pipeline

        Parameters:
        - position: integer index (0-based)
        - after_node: insert after this node name
        """
        pipeline = self.cfg.setdefault("pipeline", [])

        # Determine position in the YAML list
        if after_node:
            for idx, s in enumerate(pipeline):
                if s["name"] == after_node:
                    position = idx + 1
                    break
            else:
                raise ValueError(f"node '{after_node}' not found for insertion")

        if position is None:
            pipeline.append(new_node_cfg)
            position = len(self.nodes)  # append at the end
        else:
            pipeline.insert(position, new_node_cfg)

        # Build instance
        node_name = new_node_cfg["name"]
        component = new_node_cfg["component"]
        node_type = new_node_cfg["type"]
        factory = FACTORY_ROUTER.get(component)
        params = new_node_cfg.get("params") or {}
        self._build_node(new_node_cfg)

        # Move it to correct position if needed
        if position is not None and position < len(self.nodes) - 1:
            self.nodes.insert(position, self.nodes.pop())
        print(f"[PipelineBuilder] Inserted node: {node_name}")

    def remove_node(self, position=None, node_name=None):
        """
        Remove a node from the pipeline.
        Either position or node_name must be provided (not both).
        """

        if not self.nodes:
            raise ValueError("Cannot remove a node from an empty pipeline.")

        if position is None and node_name is None:
            raise ValueError("Either position or node_name must be provided.")

        if position is not None and node_name is not None:
            raise ValueError("Provide only one of position or node_name.")

        pipeline = self.cfg.get("pipeline", [])

        # --- Remove by position ---
        if position is not None:
            self._remove_node_by_position(pipeline=pipeline, position=position)
            return

        # --- Remove by node_name ---
        if node_name is not None:
            self._remove_node_by_name(pipeline=pipeline, node_name=node_name)

    def get_node(self, node_name):
        """Return the built instance of a node by name (first match)."""
        for name, inst in self.nodes:
            if name == node_name:
                return inst
        return None

    def get_pipeline_order(self):
        """Return the names of the nodes in order."""
        return [name for name, _ in self.nodes]


if __name__ == '__main__':
    # Example: ensure detector/tracker classes are imported to register
    from trackey.core.detectors.yolo import YoloDetector
    from trackey.core.features.mediapipe import MPLandmarkDetector
    from trackey.core.trackers.deepsort import DeepSortTracker

    builder = PipelinBuilder('../base_pipeline.yaml')
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
