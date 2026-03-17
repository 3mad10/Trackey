from abc import ABC, abstractmethod
from typing import Dict


class PipelineNode(ABC):
    @abstractmethod
    def process(self, data: Dict) -> Dict:
        """
        Receives a data dictionary (frame, detections, tracks, features, etc.)
        Returns updated data dictionary.
        """
        pass


class DetectorNode(PipelineNode):
    def __init__(self, detector):
        """
        detector: object with method detect(frame) -> list[Detection]
        """
        self.detector = detector

    def process(self, data: Dict) -> Dict:
        frame = data.get("frame")
        if frame is None:
            return data

        detections = self.detector.detect(frame)
        data["detections"] = detections
        return data


class TrackerNode(PipelineNode):
    def __init__(self, tracker):
        """
        tracker: object with method update(frame, detections) -> list[Track]
        """
        self.tracker = tracker

    def process(self, data: Dict) -> Dict:
        frame = data.get("frame")
        detections = data.get("detections", [])
        if not detections:
            return data
        tracks = self.tracker.update(detections, frame)
        data["tracks"] = tracks
        return data


class AnalyzerNode(PipelineNode):
    def __init__(self, analyzer, key: str):
        """
        analyzer: object with method analyze(frame, tracks) -> any
        key: name to store results in data['analytics']
        """
        self.analyzer = analyzer
        self.key = key

    def process(self, data: Dict) -> Dict:
        frame = data.get("frame")
        tracks = data.get("tracks", [])
        if not tracks:
            return data

        result = self.analyzer.analyze(tracks, frame)
        if "analytics" not in data:
            data["analytics"] = {}
        data["analytics"][self.key] = result
        return data


class ReIDNode(PipelineNode):
    def __init__(self, reid_model):
        """
        reid_model: object with method assign_ids(tracks) -> list[Track]
        """
        self.reid_model = reid_model

    def process(self, data: Dict) -> Dict:
        tracks = data.get("tracks", [])
        if not tracks:
            return data

        enriched_tracks = self.reid_model.assign_ids(tracks)
        data["tracks"] = enriched_tracks
        return data


class PostprocessorNode(PipelineNode):
    def __init__(self, postprocessor):
        """
        postprocessor: object with method process(tracks) -> list[Track]
        """
        self.postprocessor = postprocessor

    def process(self, data: Dict) -> Dict:
        tracks = data.get("tracks", [])
        if not tracks:
            return data

        processed_tracks = self.postprocessor.process(tracks)
        data["tracks"] = processed_tracks
        return data


class AnalyzerNode(PipelineNode):
    def __init__(self, analyzer, key: str):
        self.analyzer = analyzer
        self.key = key
    
    def process(self, data: Dict) -> Dict:
        frame = data.get("frame")
        tracks = data.get("tracks", [])
        
        if not tracks:
            return data
        
        result = self.analyzer.analyze(tracks, frame)
        
        # Store analytics result
        if "analytics" not in data:
            data["analytics"] = {}
        data["analytics"][self.key] = result
        
        return data