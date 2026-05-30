from typing import List, Any, Optional, Dict
from dataclasses import dataclass, field, replace

from trackey.data.schemas.track import Track
from trackey.core.context import FrameContext
from trackey.core.interfaces.node import PipelineNode
from trackey.core.interfaces.node import PipelineNode
from trackey.core.scene.mappings import ZoneMemberships
from trackey.core.scene import Scene
from trackey.data.schemas.event import EventDefinition
from trackey.core.events.bus import EventBus
from trackey.core.utils.path import PathExtractor
from trackey.core.interfaces import *
from trackey.core.pipeline.constants import SKIP_BRANCH


class ZoneFilterMixin:
    def filter_tracks(self, ctx: FrameContext) -> List[Track]:
        if self.zone_name:
            track_ids = ctx.zone_memberships.by_zone.get(self.zone_name, [])
            return [t for t in ctx.tracks if t.id in track_ids]
        return ctx.tracks

@dataclass
class DetectorNode(PipelineNode):
    detector: Detector

    def process(self, ctx: FrameContext) -> FrameContext:
        detections = self.detector.detect(ctx.frame)
        return ctx.with_detections(detections)
    
    def get_inputs(self) -> List[str]:
        return ["frame"]

    def get_outputs(self) -> List[str]:
        return ["detections"]

@dataclass
class TrackerNode(PipelineNode):
    tracker: Tracker

    def process(self, ctx: FrameContext) -> FrameContext:
        tracks = self.tracker.update(ctx.detections, ctx.frame)
        return ctx.with_tracks(tracks)
    
    def get_inputs(self) -> List[str]:
        return ["detections", "frame"]

    def get_outputs(self) -> List[str]:
        return ["tracks"]

@dataclass
class AnalyzerNode(PipelineNode, ZoneFilterMixin):
    analyzer:  Analyzer
    zone_name: Optional[str] = None

    def process(self, ctx: FrameContext) -> FrameContext:
        tracks = self.filter_tracks(ctx)
        result = self.analyzer.analyze(tracks)
        if self.zone_name:
            result["zone_name"] = self.zone_name
        # print(f"result : {result}")
        return ctx.with_analytics(self.name, result)
    
    def get_inputs(self) -> List[str]:
        return ["tracks", "zone_memberships"]

    def get_outputs(self) -> List[str]:
        return [f"analytics.{self.name}"]

@dataclass
class SpatialIndexNode(PipelineNode):
    scene: Scene

    def process(self, ctx: FrameContext) -> FrameContext:
        memberships = ZoneMemberships.build(ctx.tracks, self.scene)
        return ctx.with_memberships(memberships)
    
    def get_inputs(self) -> List[str]:
        return ["tracks"]

    def get_outputs(self) -> List[str]:
        return ["zone_memberships"]

@dataclass
class SwitchNode(PipelineNode):
    path: str
    cases: Dict[Any, str]
    default: Optional[str] = None

    def __post_init__(self):
        self._extractor = PathExtractor(self.path)

    def process(self, ctx: FrameContext) -> FrameContext:
        value  = self._extractor.extract(ctx)
        target = self.cases.get(value, self.default)
        return ctx.with_branch(target)
    
    def get_inputs(self) -> List[str]:
        return [self.path]

    def get_outputs(self) -> List[str]:
        return ["triggered_conditions"]

@dataclass
class PublisherNode(PipelineNode):
    def __init__(self, name: str,
                 definitions: List[EventDefinition],
                 event_bus: EventBus):
        super().__init__(name)
        self.definitions = definitions
        self.event_bus   = event_bus

    def process(self, ctx: FrameContext) -> FrameContext:
        for definition in self.definitions:
            event = definition.build(ctx)
            self.event_bus.publish(event)
        return ctx

    def get_inputs(self) -> List[str]:
        return []

    def get_outputs(self) -> List[str]:
        return []

@dataclass
class ConditionNode(PipelineNode):
    path:         str
    operator:     str
    threshold:    Any
    true_output:  str
    false_output: Optional[str] = None

    def __post_init__(self):
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
    

@dataclass
class ReIDNode(PipelineNode, ZoneFilterMixin):
    reid_model: None

    def process(self, ctx: FrameContext) -> FrameContext:

        tracks = self.filter_tracks(ctx)

        enriched_tracks = self.reid_model.assign_ids(tracks)
        ctx.tracks = enriched_tracks
        return ctx


class PostprocessorNode(PipelineNode):
    postprocessor: None

    def process(self, ctx: FrameContext) -> FrameContext:
        tracks = ctx.tracks

        processed_tracks = self.postprocessor.process(tracks)
        ctx.tracks = processed_tracks
        return ctx