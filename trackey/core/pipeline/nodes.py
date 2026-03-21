from typing import List
from trackey.data.schemas.track import Track

from trackey.core.context import FrameContext
from trackey.core.interfaces.node import PipelineNode
from trackey.core.context import FrameContext
from trackey.core.interfaces.node import PipelineNode
from trackey.core.scene.scene import Scene
from trackey.core.scene.mappings import ZoneMemberships


class ZoneFilterMixin:
    def filter_tracks(self, ctx: FrameContext) -> List[Track]:
        if self.zone_name:
            track_ids = ctx.zone_memberships.by_zone.get(self.zone_name, [])
            return [t for t in ctx.tracks if t.id in track_ids]
        return ctx.tracks


class DetectorNode(PipelineNode):
    def __init__(self, name: str,  detector):
        """
        detector: object with method detect(frame) -> list[Detection]
        """
        super().__init__(name)
        self.detector = detector

    def process(self, ctx: FrameContext) -> FrameContext:
        frame = ctx.frame
        if frame is None:
            return ctx

        detections = self.detector.detect(frame)
        ctx.detections = detections
        return ctx


class TrackerNode(PipelineNode):
    def __init__(self, name: str,  tracker):
        """
        tracker: object with method update(frame, detections) -> list[Track]
        """
        super().__init__(name)
        self.tracker = tracker

    def process(self, ctx: FrameContext) -> FrameContext:
        frame = ctx.frame
        detections = ctx.detections
        tracks = self.tracker.update(detections, frame)
        ctx.tracks = tracks
        return ctx


class AnalyzerNode(PipelineNode, ZoneFilterMixin):
    def __init__(self, name: str, analyzer, **node_cfg):
        """
        analyzer: object with method analyze(tracks) -> list[Track]
        """
        super().__init__(name)
        self.analyzer = analyzer
        if 'zone' in node_cfg:
            self.zone_name = node_cfg['zone']
        else:
            self.zone_name = None

    def process(self, ctx: FrameContext) -> FrameContext:

        tracks = self.filter_tracks(ctx)

        ctx.analytics[self.name] = self.analyzer.analyze(tracks)

        return ctx


class ReIDNode(PipelineNode, ZoneFilterMixin):
    def __init__(self, name: str,  reid_model):
        """
        reid_model: object with method assign_ids(tracks) -> list[Track]
        """
        super().__init__(name)
        self.reid_model = reid_model

    def process(self, ctx: FrameContext) -> FrameContext:

        tracks = self.filter_tracks(ctx)

        enriched_tracks = self.reid_model.assign_ids(tracks)
        ctx.tracks = enriched_tracks
        return ctx


class PostprocessorNode(PipelineNode):
    def __init__(self, name: str,  postprocessor):
        """
        postprocessor: object with method process(tracks) -> list[Track]
        """
        super().__init__(name)
        self.postprocessor = postprocessor

    def process(self, ctx: FrameContext) -> FrameContext:
        tracks = ctx.tracks

        processed_tracks = self.postprocessor.process(tracks)
        ctx.tracks = processed_tracks
        return ctx


class SpatialIndexNode(PipelineNode):
    def __init__(self, name: str, scene: Scene):
        """
        SpatialIndexNode: object with method FrameContext -> FrameContext
        """
        super().__init__(name)
        self.scene = scene

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.zone_memberships = ZoneMemberships.build(ctx.tracks, self.scene)
        return ctx