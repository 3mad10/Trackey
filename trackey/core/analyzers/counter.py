from typing import List, Dict, Any
from collections import defaultdict
import logging


from trackey.core.interfaces import Analyzer
from trackey.data.schemas.track import Track
from trackey.data.schemas.frame import Frame
from trackey.core.register import register_analyzer


logger = logging.getLogger(__name__)

@register_analyzer('counter')
class Counter(Analyzer):
    """
    Count metrices per object.
    """
    VALID_METRICS = {
        "current",
        "peak",
        "cumulative",
    }

    def __init__(
        self,
        target_classes: List[str] = None,
        metrics: List[str] = None
    ):

        self.target_classes = (
            {c.lower() for c in target_classes}
            if target_classes else None
        )
        self.metrics = set(metrics or ["current"])
        for metric in self.metrics:
            if metric not in self.VALID_METRICS:
                raise ValueError(f"[Analyzer][Counter] Unknown metric {metric}")

        self.peak_counts = defaultdict(int)
        self.unique_tracks = defaultdict(set)

    def analyze(
        self,
        tracks: List[Track],
        frame: Frame = None
    ) -> Dict[str, Any]:

        current_counts = defaultdict(int)
        # print("tracks : ", tracks)
        for track in tracks:
            # print(track)
            if not track.history:
                continue
            track_class = track.class_name.lower()
            # print(track_class)
            # print(self.target_classes)
            if self.target_classes is None or track_class in self.target_classes:
                current_counts[track_class] += 1
                self.unique_tracks[track_class].add(track.id)
        

        result = {}
        for class_name, count in current_counts.items():
            result[class_name] = {}
            if "current" in self.metrics:
                result[class_name]["current"] = count
            if "peak" in self.metrics:
                self.peak_counts[class_name] = max(self.peak_counts[class_name], count)
                result[class_name]["peak"] = self.peak_counts[class_name]
            if "cumulative" in self.metrics:
                result[class_name]["cumulative"] = len(self.unique_tracks[class_name])
        print(self.unique_tracks)
        return result
