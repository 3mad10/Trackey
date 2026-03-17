import logging
from typing import Dict, Callable, List, Any
from collections import defaultdict
from trackey.core.pipeline.nodes import PipelineNode


logger = logging.getLogger(__name__)


class SwitchNode(PipelineNode):
    """
    Multi-class switch with flexible result storage.
    Results are ALWAYS stored - never lost.
    """
    
    def __init__(self,
                 class_key: str = "detections",
                 class_selector: Callable[[Any], Any] = None,
                 cases: Dict[Any, List[PipelineNode]] = None,
                 default: List[PipelineNode] = None,
                 merge_strategy: str = "nested"):
        """
        Args:
            class_key: Data key containing items with classes
            class_selector: Function to extract class from item
            cases: Dict mapping class values to node lists
            default: Nodes for unmatched classes
            merge_strategy: How to store results:
                - "nested" (default): Store in data["class_results"][class_id]
                - "flat": Merge into data["tracks"] and data["analytics"]
                - "both": Store in both formats
        """
        self.class_key = class_key
        self.class_selector = class_selector or (lambda item: item.class_id)
        self.cases = cases or {}
        self.default = default or []
        self.merge_strategy = merge_strategy
        
        # Validate merge_strategy
        valid_strategies = ["nested", "flat", "both"]
        if self.merge_strategy not in valid_strategies:
            logging.error(
                f"[SwitchNode] merge_strategy must be one of {valid_strategies}, "
                f"got '{self.merge_strategy}'"
            )
            raise ValueError(
                f"[SwitchNode] merge_strategy must be one of {valid_strategies}, "
                f"got '{self.merge_strategy}'"
            )
    
    def process(self, data: Dict) -> Dict:
        items = data.get(self.class_key, [])
        
        if not items:
            return data
        
        # Group items by class
        grouped = defaultdict(list)
        for item in items:
            class_value = self.class_selector(item)
            grouped[class_value].append(item)
        
        # Process each class
        class_results = {}
        
        for class_value, class_items in grouped.items():
            # Get nodes for this class
            nodes = self.cases.get(class_value, self.default)
            
            if not nodes:
                # No nodes for this class, but still store the items
                class_results[class_value] = {
                    self.class_key: class_items
                }
                continue
            
            # Create class-specific data
            class_data = data.copy()
            class_data[self.class_key] = class_items
            
            # Execute nodes for this class
            for node in nodes:
                class_data = node.process(class_data)
            
            # Store processed result
            class_results[class_value] = class_data
        
        # Store results based on strategy
        if self.merge_strategy in ["nested", "both"]:
            data["class_results"] = class_results
        
        if self.merge_strategy in ["flat", "both"]:
            self._merge_flat(data, class_results)
        
        return data
    
    def _merge_flat(self, data: Dict, class_results: Dict[Any, Dict]):
        """Merge class results into main data structure"""
        
        # Merge tracks from all classes
        all_tracks = []
        for class_value, class_data in class_results.items():
            tracks = class_data.get("tracks", [])
            
            # Tag tracks with source class
            for track in tracks:
                if not hasattr(track, 'metadata'):
                    track.metadata = {}
                track.metadata['source_class'] = class_value
            
            all_tracks.extend(tracks)
        
        if all_tracks:
            data["tracks"] = all_tracks
        
        # Merge analytics with class prefixes
        if "analytics" not in data:
            data["analytics"] = {}
        
        for class_value, class_data in class_results.items():
            class_name = self._get_class_name(class_value)
            analytics = class_data.get("analytics", {})
            
            for key, value in analytics.items():
                prefixed_key = f"{class_name}_{key}"
                data["analytics"][prefixed_key] = value
    
    def _get_class_name(self, class_id: int) -> str:
        """Get human-readable class name"""
        names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            5: "bus", 7: "truck", 16: "dog", 17: "cat"
        }
        return names.get(class_id, f"class_{class_id}")