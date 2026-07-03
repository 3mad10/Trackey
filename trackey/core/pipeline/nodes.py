from typing import List, Any, Optional, Dict
from dataclasses import dataclass, field, replace

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


class ZoneFilterMixin:
    def filter_tracks(self, ctx: FrameContext) -> List[Track]:
        if self.zone_name:
            track_ids = ctx.zone_memberships.by_zone.get(self.zone_name, [])
            return [t for t in ctx.tracks if t.id in track_ids]
        return ctx.tracks

@dataclass
class DetectorNode(PipelineNode):
    detector: Detector
    detection_source: DetectionSource

    def process(self, ctx: FrameContext) -> FrameContext:
        detections = self.detector.detect(ctx.frame)
        return ctx.with_detections(source=self.detection_source , detections=detections)
    
    def get_inputs(self) -> List[str]:
        return ["frame"]

    def get_outputs(self) -> List[str]:
        return [f"detections.{self.detection_source}"]

@dataclass
class TrackerNode(PipelineNode):
    tracker: Tracker

    def process(self, ctx: FrameContext) -> FrameContext:
        tracks = self.tracker.update(ctx.detections[DetectionSource.PRIMARY], ctx.frame)
        return ctx.with_tracks(tracks)
    
    def get_inputs(self) -> List[str]:
        return [f"detections.{DetectionSource.PRIMARY}", "frame"]

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
class ReIDNode(PipelineNode, ZoneFilterMixin):
    reid_model: FeatureExtractor
    zone_name: Optional[str] = None

    def process(self, ctx: FrameContext) -> FrameContext:
        
        tracks = self.filter_tracks(ctx)

        embeddings = self.reid_model.extract(tracks, ctx.frame)
        
        for i in range(len(embeddings)):
            tracks[i].embedding = embeddings[i]
        return ctx.with_tracks(tracks)
    
    def get_inputs(self) -> List[str]:
        return ["tracks", "frame", "zone_memberships"]

    def get_outputs(self) -> List[str]:
        return [f"tracks.embedding"]


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


class PostprocessorNode(PipelineNode):
    postprocessor: None

    def process(self, ctx: FrameContext) -> FrameContext:
        tracks = ctx.tracks

        processed_tracks = self.postprocessor.process(tracks)
        ctx.tracks = processed_tracks
        return ctx


# @dataclass
# class FaceAssociationNode(PipelineNode):
#     """
#     Associates independently-detected faces (whole-frame detector
#     output, source="face" in ctx.detections) with person tracks
#     using geometric containment.

#     A face is considered to belong to a track if the face bbox
#     center falls within the track's bbox, weighted toward the
#     upper portion of the body (where heads are).

#     Use only with high track density where per-track face detection
#     (RetinaFaceCroppedNode) would be too expensive. Carries real
#     failure modes in crowded/occluded scenes — validate against
#     real footage before production use.
#     """

#     def __init__(self,
#                  name:               str,
#                  face_source:        str = "face",
#                  head_region_ratio:  float = 0.4):
#         """
#         Args:
#             face_source:       Key under ctx.detections where the
#                                 whole-frame face detector wrote results.
#             head_region_ratio:  Fraction of the track bbox height (from
#                                  the top) considered the "head region".
#                                  A face must fall within this band to
#                                  be associated. 0.4 means top 40%.
#         """
#         super().__init__(name)
#         self.face_source       = face_source
#         self.head_region_ratio = head_region_ratio

#     def process(self, ctx: FrameContext) -> FrameContext:
#         faces = ctx.get_detections(self.face_source)
#         if not faces:
#             return ctx

#         updated_tracks = [
#             self._associate(track, faces)
#             for track in ctx.tracks
#         ]
#         return ctx.with_tracks(updated_tracks)

#     def _associate(self, track: Track,
#                    faces: List[Detection]) -> Track:
#         if track.bbox is None:
#             return track

#         best_face = None
#         best_score = 0.0

#         for face in faces:
#             if face.bbox is None:
#                 continue
#             if not self._face_in_head_region(face.bbox, track.bbox):
#                 continue

#             score = self._containment_score(face.bbox, track.bbox)
#             if score > best_score:
#                 best_score = score
#                 best_face  = face

#         if best_face is None:
#             return track

#         return track.model_copy(update={"face_bbox": best_face.bbox})

#     def _face_in_head_region(self, face_bbox: BoundingBox,
#                               track_bbox: BoundingBox) -> bool:
#         tx1 = track_bbox.cx - track_bbox.w / 2
#         tx2 = track_bbox.cx + track_bbox.w / 2
#         ty1 = track_bbox.cy - track_bbox.h / 2
#         ty2 = ty1 + track_bbox.h * self.head_region_ratio

#         in_x = tx1 <= face_bbox.cx <= tx2
#         in_y = ty1 <= face_bbox.cy <= ty2
#         return in_x and in_y

#     def _containment_score(self, face_bbox: BoundingBox,
#                             track_bbox: BoundingBox) -> float:
#         """Closer to top-center of the track bbox scores higher."""
#         head_center_x = track_bbox.cx
#         head_center_y = (track_bbox.cy - track_bbox.h / 2) + (
#             track_bbox.h * self.head_region_ratio / 2
#         )
#         dx = abs(face_bbox.cx - head_center_x)
#         dy = abs(face_bbox.cy - head_center_y)
#         return 1.0 / (1.0 + dx + dy)

#     def get_inputs(self) -> List[str]:
#         return ["tracks", "detections"]

#     def get_outputs(self) -> List[str]:
#         return ["tracks"]