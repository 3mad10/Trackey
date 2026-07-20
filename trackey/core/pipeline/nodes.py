from typing import List, Any, Optional, Dict
from dataclasses import replace
from datetime import datetime, timezone

from trackey.data.schemas.track import Track
from trackey.core.context import FrameContext
from trackey.core.interfaces.node import PipelineNode
from trackey.core.scene.mappings import ZoneMemberships
from trackey.core.scene import Scene
from trackey.data.schemas.event import EventDefinition
from trackey.data.schemas.detection import DetectionSource
from trackey.core.events.bus import EventBus
from trackey.core.utils.path import PathExtractor
from trackey.core.interfaces import *
from trackey.core.pipeline.constants import SKIP_BRANCH
from trackey.core.recognition.identifier import Identifier


class ZoneFilterMixin:
    def filter_tracks(self, ctx: FrameContext) -> List[Track]:
        if self.zone_name:
            track_ids = ctx.zone_memberships.by_zone.get(self.zone_name, [])
            return [t for t in ctx.tracks if t.tracker_id in track_ids]
        return ctx.tracks


class DetectorNode(PipelineNode):
    def __init__(self, name: str, detector: Detector, detection_source: str):
        super().__init__(name)
        self.detector = detector
        self.detection_source = detection_source

    def process(self, ctx: FrameContext) -> FrameContext:
        detections = self.detector.detect(ctx.frame)
        return ctx.with_detections(source=self.detection_source, detections=detections)

    def get_inputs(self) -> List[str]:
        return ["frame"]

    def get_outputs(self) -> List[str]:
        return [f"detections.{self.detection_source}"]


class TrackerNode(PipelineNode):
    def __init__(self, name: str, tracker: Tracker):
        super().__init__(name)
        self.tracker = tracker

    def process(self, ctx: FrameContext) -> FrameContext:
        primary = ctx.get_detections(DetectionSource.PRIMARY)  # [] if not yet populated, not a KeyError
        tracks = self.tracker.update(primary, ctx.frame)
        return ctx.with_tracks(tracks)

    def get_inputs(self) -> List[str]:
        return [f"detections.{DetectionSource.PRIMARY}", "frame"]

    def get_outputs(self) -> List[str]:
        return ["tracks"]


class AnalyzerNode(PipelineNode, ZoneFilterMixin):
    def __init__(self, name: str, analyzer: Analyzer, zone_name: Optional[str] = None):
        super().__init__(name)
        self.analyzer = analyzer
        self.zone_name = zone_name

    def process(self, ctx: FrameContext) -> FrameContext:
        tracks = self.filter_tracks(ctx)
        result = self.analyzer.analyze(tracks)
        if self.zone_name:
            result["zone_name"] = self.zone_name
        return ctx.with_analytics(self.name, result)

    def get_inputs(self) -> List[str]:
        return ["tracks", "zone_memberships"]

    def get_outputs(self) -> List[str]:
        return [f"analytics.{self.name}"]


class EmbeddingNode(PipelineNode, ZoneFilterMixin):
    """Feature extraction only — writes per_track.embeddings[tracker_id][modality].
    Does NOT assign identity; that's ReIdentificationNode's job, downstream."""

    def __init__(self, name: str, extractor: FeatureExtractor, modality: str, zone_name: Optional[str] = None):
        super().__init__(name)
        self.extractor = extractor
        self.modality = modality
        self.zone_name = zone_name

    def process(self, ctx: FrameContext) -> FrameContext:
        tracks = self.filter_tracks(ctx)
        embeddings = self.extractor.extract(tracks, ctx.frame)  # same length/order as tracks; None per failed extraction

        existing = ctx.per_track.get("embeddings", {})
        updated = {**existing}
        for track, emb in zip(tracks, embeddings):
            if emb is None:
                continue
            entry = {**updated.get(track.tracker_id, {}), self.modality: emb}
            updated[track.tracker_id] = entry

        return replace(ctx, per_track={**ctx.per_track, "embeddings": updated})

    def get_inputs(self) -> List[str]:
        return ["tracks", "frame", "zone_memberships"]

    def get_outputs(self) -> List[str]:
        return ["per_track.embeddings"]


class SpatialIndexNode(PipelineNode):
    def __init__(self, name: str, scene: Scene):
        super().__init__(name)
        self.scene = scene

    def process(self, ctx: FrameContext) -> FrameContext:
        memberships = ZoneMemberships.build(ctx.tracks, self.scene)
        return ctx.with_memberships(memberships)

    def get_inputs(self) -> List[str]:
        return ["tracks"]

    def get_outputs(self) -> List[str]:
        return ["zone_memberships"]


class SwitchNode(PipelineNode):
    def __init__(self, name: str, path: str, cases: Dict[Any, str], default: Optional[str] = None):
        super().__init__(name)
        self.path = path
        self.cases = cases
        self.default = default
        self._extractor = PathExtractor(self.path)

    def process(self, ctx: FrameContext) -> FrameContext:
        value = self._extractor.extract(ctx)
        target = self.cases.get(value, self.default)
        return ctx.with_branch(target)

    def get_inputs(self) -> List[str]:
        return [self.path]

    def get_outputs(self) -> List[str]:
        return ["triggered_conditions"]


class PublisherNode(PipelineNode):
    def __init__(self, name: str, definitions: List[EventDefinition], event_bus: EventBus):
        super().__init__(name)
        self.definitions = definitions
        self.event_bus = event_bus

    def process(self, ctx: FrameContext) -> FrameContext:
        for definition in self.definitions:
            event = definition.build(ctx)
            self.event_bus.publish(event)
        return ctx

    def get_inputs(self) -> List[str]:
        return []

    def get_outputs(self) -> List[str]:
        return []


class ConditionNode(PipelineNode):
    def __init__(self, name: str, path: str, operator: str, threshold: Any,
                 true_output: str, false_output: Optional[str] = None):
        super().__init__(name)
        self.path = path
        self.operator = operator
        self.threshold = threshold
        self.true_output = true_output
        self.false_output = false_output
        self._extractor = PathExtractor(self.path)

    def process(self, ctx: FrameContext) -> FrameContext:
        extracted = self._extractor.extract(ctx)
        if extracted is None:
            return replace(ctx, active_branch=SKIP_BRANCH)
        if self._evaluate(extracted):
            return replace(ctx, active_branch=self.true_output)
        return replace(ctx, active_branch=self.false_output or SKIP_BRANCH)

    def get_inputs(self) -> List[str]:
        return [self.path]

    def get_outputs(self) -> List[str]:
        return ["triggered_conditions"]

    def _evaluate(self, value: Any) -> bool:
        ops = {
            "gt":  lambda a, b: a > b,
            "lt":  lambda a, b: a < b,
            "eq":  lambda a, b: a == b,
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
            "ne":  lambda a, b: a != b,
        }
        op = ops.get(self.operator)
        if not op:
            raise ValueError(f"Unknown operator: {self.operator}")
        return op(value, self.threshold)


class PostprocessorNode(PipelineNode):
    def __init__(self, name: str, postprocessor: Any):
        super().__init__(name)
        self.postprocessor = postprocessor

    def process(self, ctx: FrameContext) -> FrameContext:
        processed_tracks = self.postprocessor.process(ctx.tracks)
        return ctx.with_tracks(processed_tracks)  # FrameContext is frozen — never assign ctx.tracks directly

    def get_inputs(self) -> List[str]:
        return ["tracks"]

    def get_outputs(self) -> List[str]:
        return ["tracks"]


class IdentificationNode(PipelineNode):
    def __init__(self, name: str, identifier: Identifier):
        super().__init__(name)
        self.identifier = identifier

    def process(self, ctx: FrameContext) -> FrameContext:
        embeddings = ctx.per_track.get("embeddings", {})
        now = datetime.now(timezone.utc)
        
        existing_identities = ctx.per_track.get("identities", {})
        updated_identities = {**existing_identities}
        
        for track in ctx.tracks:
            emb = embeddings.get(track.tracker_id, {}).get(self.identifier.modality)
            if emb is None:
                continue
            identity = self.identifier.identify(emb, now=now)
            if identity:
                entry = {**updated_identities.get(track.tracker_id, {}), self.identifier.modality: identity}
                updated_identities[track.tracker_id] = entry
                
        return replace(ctx, per_track={**ctx.per_track, "identities": updated_identities})

    def get_inputs(self) -> List[str]:
        return ["tracks", "per_track.embeddings"]

    def get_outputs(self) -> List[str]:
        return ["per_track.identities"]


class UnifiedIdentityManagerNode(PipelineNode):
    def __init__(self, name: str):
        super().__init__(name)

    def process(self, ctx: FrameContext) -> FrameContext:
        identities_map = ctx.per_track.get("identities", {})
        updated_tracks = []
        for track in ctx.tracks:
            candidates = identities_map.get(track.tracker_id, {})
            # Strategy: Prioritize face, fallback to body
            final_identity = candidates.get("face") or candidates.get("body") or None
            
            if final_identity:
                updated_tracks.append(replace(track, identity=final_identity))
            else:
                updated_tracks.append(track)
                
        return ctx.with_tracks(updated_tracks)

    def get_inputs(self) -> List[str]:
        return ["tracks", "per_track.identities"]

    def get_outputs(self) -> List[str]:
        return ["tracks"]
