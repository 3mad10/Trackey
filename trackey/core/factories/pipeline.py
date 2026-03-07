import yaml
from pathlib import Path
from trackey.core.factories import FACTORY_ROUTER, NODE_WRAPPERS


class PipelinBuilder:
    def __init__(self, cfg_path: str):
        self.nodes = []  # list of tuples: (step_name, instance)
        self.cfg = self._load_yaml(cfg_path)

    def build(self):
        self._build_pipeline()
        return self.nodes

    def _load_yaml(self, cfg_path: str):
        cfg_path = Path(cfg_path)

        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

        with cfg_path.open("r") as f:
            return yaml.safe_load(f)

    def _build_pipeline(self):
        """Build all steps from the loaded YAML in order."""
        pipeline = self.cfg.get("pipeline", [])
        if not isinstance(pipeline, list):
            raise TypeError("Pipeline must be a list of steps")

        for step_cfg in pipeline:
            self._build_node(step_cfg)

    def _build_node(self, step_cfg):
        step_name = step_cfg["name"]

        if any(name == step_name for name, _ in self.nodes):
            raise ValueError(f"Duplicate step name detected: '{step_name}'")

        kind = step_cfg["kind"]
        step_type = step_cfg["type"]
        params = step_cfg.get("params") or {}

        # Build raw component
        component_factory = FACTORY_ROUTER.get(kind)
        if not component_factory:
            raise ValueError(f"Unknown step kind: {kind}")

        component = component_factory(step_type, **params)

        # Wrap into node
        node_builder = NODE_WRAPPERS.get(kind)
        if not node_builder:
            raise ValueError(f"No Node wrapper for kind: {kind}")

        node = node_builder(component, step_cfg)

        self.nodes.append((step_name, node))
        print(f"[PipelineBuilder] Built step: {step_name} ({kind}/{step_type})")

    def _remove_node_by_position(self, pipeline, position):
        if not isinstance(position, int):
            raise TypeError("position must be an integer")

        if position < 0 or position >= len(self.nodes):
            raise IndexError("position out of range")

        removed_node = self.nodes.pop(position)
        pipeline.pop(position)

        print(f"[PipelineBuilder] Removed step at position {position}: {removed_node[0]}")

    def _remove_node_by_name(self, pipeline, step_name):
        if not isinstance(step_name, str):
            raise TypeError("step_name must be a string")

        for idx, (name, _) in enumerate(self.nodes):
            if name == step_name:
                self.nodes.pop(idx)
                pipeline.pop(idx)
                print(f"[PipelineBuilder] Removed step '{step_name}'")
                return

        raise ValueError(f"[PipelineBuilder] Step '{step_name}' not found in pipeline.")

    def insert_node(self, new_node_cfg, position=None, after_node=None):
        """
        Insert a step at a specific position in the pipeline and steps list.
        If after_node and position are not passed the step is added to the end of the pipeline

        Parameters:
        - position: integer index (0-based)
        - after_node: insert after this step name
        """
        pipeline = self.cfg.setdefault("pipeline", [])

        # Determine position in the YAML list
        if after_node:
            for idx, s in enumerate(pipeline):
                if s["name"] == after_node:
                    position = idx + 1
                    break
            else:
                raise ValueError(f"Step '{after_node}' not found for insertion")

        if position is None:
            pipeline.append(new_node_cfg)
            position = len(self.nodes)  # append at the end
        else:
            pipeline.insert(position, new_node_cfg)

        # Build instance
        step_name = new_node_cfg["name"]
        kind = new_node_cfg["kind"]
        step_type = new_node_cfg["type"]
        factory = FACTORY_ROUTER.get(kind)
        params = new_node_cfg.get("params") or {}
        self._build_node(new_node_cfg)

        # Move it to correct position if needed
        if position is not None and position < len(self.nodes) - 1:
            self.nodes.insert(position, self.nodes.pop())
        print(f"[PipelineBuilder] Inserted step: {step_name}")

    def remove_node(self, position=None, step_name=None):
        """
        Remove a step from the pipeline.
        Either position or step_name must be provided (not both).
        """

        if not self.nodes:
            raise ValueError("Cannot remove a step from an empty pipeline.")

        if position is None and step_name is None:
            raise ValueError("Either position or step_name must be provided.")

        if position is not None and step_name is not None:
            raise ValueError("Provide only one of position or step_name.")

        pipeline = self.cfg.get("pipeline", [])

        # --- Remove by position ---
        if position is not None:
            self._remove_node_by_position(pipeline=pipeline, position=position)
            return

        # --- Remove by step_name ---
        if step_name is not None:
            self._remove_node_by_name(pipeline=pipeline, step_name=step_name)

    def get_node(self, step_name):
        """Return the built instance of a step by name (first match)."""
        for name, inst in self.nodes:
            if name == step_name:
                return inst
        return None

    def get_pipeline_order(self):
        """Return the names of the steps in order."""
        return [name for name, _ in self.nodes]


if __name__ == '__main__':
    # Example: ensure detector/tracker classes are imported to register
    from trackey.core.detectors.yolo import YoloDetector
    from trackey.core.detectors.mediapipe import MPLandmarkDetector
    from trackey.core.trackers.deepsort import DeepSortTracker

    builder = PipelinBuilder('../base_pipeline.yaml')
    print("Pipeline order:", builder.get_pipeline_order())

    # Example: insert a new detector after 'detector1'
    new_node = {
        "name": "detector2",
        "kind": "detector",
        "type": "yolo",
        "params": {"weights": "yolov8s.pt"}
    }
    builder.insert_node(new_node, after_node="detector")
    print("Pipeline order after insertion:", builder.get_pipeline_order())
    builder.remove_node(step_name="detector2")
    print("Pipeline order after removal:", builder.get_pipeline_order())
