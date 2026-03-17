from abc import abstractmethod
from typing import List, Optional, Dict, Any
from trackey.data.schemas.track import Track
from trackey.data.schemas.geometry import Zone
from trackey.data.schemas.frame import Frame
from trackey.core.interfaces.analyzer import Analyzer

class BaseAnalyzer(Analyzer):
    """
    Base class for all analyzers.
    Analyzers can optionally filter by area of effect.
    """
    def __init__(self, zone: Optional[Zone] = None):
        """
        Args:
            zone: Optional zone/region name to limit analysis.
                          If None, analyzes all tracks.
        """
        self.zone = zone
    
    def analyze(self, tracks: List[Track], frame:Frame=None) -> Dict[str, Any]:
        """
        Analyze tracks, optionally filtered by area of effect.
        
        Args:
            tracks: All tracks in frame
            frame: Optional frame data (Must be added IF there is a zone for analysis)
        
        Returns:
            Analysis results dict
        """
        # Filter tracks by area of effect if specified
        if self.zone:
            filtered_tracks = self._filter_by_area(tracks)
        else:
            filtered_tracks = tracks
        
        # Perform actual analysis
        return self._analyze_impl(filtered_tracks, frame)
    
    @abstractmethod
    def _analyze_impl(self, tracks: List[Track], frame) -> Dict[str, Any]:
        """
        Implement actual analysis logic.
        Subclasses override this, not analyze().
        """
        pass
    
    def _filter_by_area(self, tracks: List[Track]) -> List[Track]:
        """Filter tracks that are within area of effect"""
        filtered = []
        
        for track in tracks:
            if not track.detections:
                continue
            
            # Get latest detection position
            latest_detection = track.detections[-1]
            position = latest_detection.bbox.center  # (cx, cy) normalized
            
            # Check if position is in area of effect
            if self.zone.contains(position):
                filtered.append(track)
        
        return filtered