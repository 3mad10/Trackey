"""Tests for PipelineExecutor DAG execution."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from typing import List
import numpy as np

from trackey.data.schemas.frame import Frame
from trackey.data.schemas.detection import BoundingBox
from trackey.data.schemas.track import Track
from trackey.core.context import FrameContext
from trackey.core.interfaces.node import PipelineNode
from trackey.core.pipeline.executer import PipelineExecutor
from trackey.core.pipeline.edge import Edge
from trackey.core.scene.mappings import ZoneMemberships


def make_frame() -> Frame:
    return Frame(frame=np.zeros((100, 100, 3), dtype=np.uint8))


def make_bbox(cx=0.5, cy=0.5, w=0.1, h=0.1) -> BoundingBox:
    return BoundingBox(cx=cx, cy=cy, w=w, h=h)


def make_track(tracker_id: int, class_name: str = "person", last_seen: datetime = None) -> Track:
    t = Track(
        tracker_id=tracker_id,
        bbox=make_bbox(),
        class_name=class_name,
    )
    if last_seen is not None:
        object.__setattr__(t, "last_seen", last_seen)
    return t


def make_ctx(**kwargs) -> FrameContext:
    defaults = dict(
        frame=make_frame(),
        frame_id=0,
        camera_id="cam0",
        timestamp=0.0,
    )
    defaults.update(kwargs)
    return FrameContext(**defaults)


class PassthroughNode(PipelineNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        return ctx

    def get_inputs(self) -> List[str]:
        return []

    def get_outputs(self) -> List[str]:
        return []


class AddTrackNode(PipelineNode):
    def __init__(self, name: str, track: Track):
        super().__init__(name=name)
        self._track = track

    def process(self, ctx: FrameContext) -> FrameContext:
        return replace(ctx, tracks=[*ctx.tracks, self._track])

    def get_inputs(self) -> List[str]:
        return []

    def get_outputs(self) -> List[str]:
        return []


class DoubleAnalyticsNode(PipelineNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        doubled = {
            k: v * 2 if isinstance(v, (int, float)) else v
            for k, v in ctx.analytics.items()
        }
        return replace(ctx, analytics=doubled)

    def get_inputs(self) -> List[str]:
        return []

    def get_outputs(self) -> List[str]:
        return []


def test_linear_pipeline_passthrough():
    A = PassthroughNode(name="A")
    B = PassthroughNode(name="B")
    C = PassthroughNode(name="C")
    executor = PipelineExecutor(
        nodes=[A, B, C],
        edges=[Edge("A", "B"), Edge("B", "C")],
    )
    ctx = make_ctx(analytics={"count": 7})
    result = executor.run(ctx)
    assert result.analytics["count"] == 7


def test_linear_pipeline_adds_tracks():
    t1 = make_track(1)
    t2 = make_track(2)
    t3 = make_track(3)
    A = AddTrackNode(name="A", track=t1)
    B = AddTrackNode(name="B", track=t2)
    C = AddTrackNode(name="C", track=t3)
    executor = PipelineExecutor(
        nodes=[A, B, C],
        edges=[Edge("A", "B"), Edge("B", "C")],
    )
    ctx = make_ctx()
    result = executor.run(ctx)
    assert len(result.tracks) == 3
    ids = {t.tracker_id for t in result.tracks}
    assert ids == {1, 2, 3}


def test_linear_pipeline_double_analytics():
    A = PassthroughNode(name="A")
    B = DoubleAnalyticsNode(name="B")
    C = PassthroughNode(name="C")
    executor = PipelineExecutor(
        nodes=[A, B, C],
        edges=[Edge("A", "B"), Edge("B", "C")],
    )
    ctx = make_ctx(analytics={"count": 5})
    result = executor.run(ctx)
    assert result.analytics["count"] == 10


def test_branching_dag_zone_memberships_merge():
    class ZoneNode(PipelineNode):
        def __init__(self, name, track_id, zone_name):
            super().__init__(name=name)
            self._track_id = track_id
            self._zone_name = zone_name

        def process(self, ctx: FrameContext) -> FrameContext:
            zm = ZoneMemberships(
                by_track={self._track_id: [self._zone_name]},
                by_zone={self._zone_name: [self._track_id]},
            )
            return replace(ctx, zone_memberships=zm)

        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

    A = PassthroughNode(name="A")
    B = ZoneNode("B", track_id=1, zone_name="zone_b")
    C = ZoneNode("C", track_id=2, zone_name="zone_c")
    D = PassthroughNode(name="D")

    executor = PipelineExecutor(
        nodes=[A, B, C, D],
        edges=[Edge("A", "B"), Edge("A", "C"), Edge("B", "D"), Edge("C", "D")],
    )
    result = executor.run(make_ctx())
    zm = result.zone_memberships
    assert 1 in zm.by_track
    assert 2 in zm.by_track
    assert "zone_b" in zm.by_zone
    assert "zone_c" in zm.by_zone


def test_per_track_merge_across_branches():
    class PerTrackNode(PipelineNode):
        def __init__(self, name, track_id, key, value):
            super().__init__(name=name)
            self._track_id = track_id
            self._key = key
            self._value = value

        def process(self, ctx: FrameContext) -> FrameContext:
            pt = dict(ctx.per_track)
            pt[self._track_id] = {self._key: self._value}
            return replace(ctx, per_track=pt)

        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

    A = PassthroughNode(name="A")
    B = PerTrackNode("B", track_id=1, key="speed", value=3)
    C = PerTrackNode("C", track_id=2, key="dwell", value=5)
    D = PassthroughNode(name="D")

    executor = PipelineExecutor(
        nodes=[A, B, C, D],
        edges=[Edge("A", "B"), Edge("A", "C"), Edge("B", "D"), Edge("C", "D")],
    )
    result = executor.run(make_ctx())
    assert result.per_track[1] == {"speed": 3}
    assert result.per_track[2] == {"dwell": 5}


def test_track_dedup_keeps_most_recent():
    class TrackNode(PipelineNode):
        def __init__(self, name, track):
            super().__init__(name=name)
            self._track = track

        def process(self, ctx: FrameContext) -> FrameContext:
            return replace(ctx, tracks=[self._track])

        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

    old_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    new_time = old_time + timedelta(seconds=10)

    t_old = make_track(42, last_seen=old_time)
    t_new = make_track(42, last_seen=new_time)

    A = PassthroughNode(name="A")
    B = TrackNode("B", t_old)
    C = TrackNode("C", t_new)
    D = PassthroughNode(name="D")

    executor = PipelineExecutor(
        nodes=[A, B, C, D],
        edges=[Edge("A", "B"), Edge("A", "C"), Edge("B", "D"), Edge("C", "D")],
    )
    result = executor.run(make_ctx())
    assert len(result.tracks) == 1
    assert result.tracks[0].last_seen == new_time
