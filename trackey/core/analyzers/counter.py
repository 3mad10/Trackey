from typing import List, Dict, Any, Tuple
from trackey.core.analyzers.base import BaseAnalyzer
from trackey.data.schemas.geometry import Zone
from trackey.data.schemas.track import Track
from trackey.data.schemas.frame import Frame
from trackey.core.register import register_analyzer


@register_analyzer('counter')
class Counter(BaseAnalyzer):
    """
    Count object in a specific zone.
    Zone is part of the analyzer configuration.
    """
    
    def __init__(self,
                 object: str,
                 zone: Zone = None,
                 count_type: str = "current"):
        """
        Args:
            zone: Zone to count object in. If None, counts all.
            count_type: Type of count to return
        """
        super().__init__(zone=zone)
        self.object = object
        self.count_type = count_type
        self.cumulative_count = 0
        self.max_count = 0
    
    def _analyze_impl(self, tracks: List[Track], frame: Frame) -> Dict[str, Any]:
        """Count object in filtered tracks"""
        current_count = 0
        for track in tracks:
            if track.detections[-1].class_name.lower() == self.object.lower():
                current_count+=1
        
        if self.count_type == "cumulative":
            self.cumulative_count += current_count
            count = self.cumulative_count
        elif self.count_type == "max":
            self.max_count = max(self.max_count, current_count)
            count = self.max_count
        else:
            count = current_count
        
        return {
            "count": count,
            "count_type": self.count_type,
        }
