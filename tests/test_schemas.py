"""Unit tests for all data schemas in trackey.data.schemas."""
import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime, timezone

from trackey.data.schemas.frame import Frame
from trackey.data.schemas.detection import (
    BoundingBox,
    Detection,
    DetectionSource,
)
from trackey.data.schemas.track import Track
from trackey.data.schemas.identity import Identity
from trackey.data.schemas.event import BaseEvent, EventDefinition
from trackey.data.schemas.pipeline import PipelineResult
from trackey.data.schemas.geometry import Point as GeoPoint, Polygon, Line, Zone
from trackey.data.schemas.feature import Point as FeatPoint, Keypoint, Keypoints, Features
from trackey.core.context import FrameContext


def test_frame_schema():
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(frame=arr)
    assert frame.height == 480
    assert frame.width == 640
    assert frame.channels == 3
    assert frame.resolution == (640, 480)

    # 2D grayscale frame
    gray_arr = np.zeros((100, 200), dtype=np.uint8)
    gray_frame = Frame(frame=gray_arr)
    assert gray_frame.channels == 1


def test_bounding_box_schema():
    bbox = BoundingBox(cx=0.5, cy=0.5, w=0.2, h=0.4)
    assert bbox.center == (0.5, 0.5)
    assert bbox.area == pytest.approx(0.08)

    # xyxy conversion
    x1, y1, x2, y2 = bbox.to_pixel_xyxy(1000, 500)
    assert x1 == 400
    assert y1 == 150
    assert x2 == 600
    assert y2 == 350

    # xywh conversion
    x1, y1, w, h = bbox.to_pixel_xywh(1000, 500)
    assert x1 == 400
    assert y1 == 150
    assert w == 200
    assert h == 200

    # validation errors
    with pytest.raises(ValueError):
        BoundingBox(cx=-0.1, cy=0.5, w=0.2, h=0.2)
    with pytest.raises(ValueError):
        BoundingBox(cx=0.5, cy=1.5, w=0.2, h=0.2)
    with pytest.raises(ValueError):
        BoundingBox(cx=0.5, cy=0.5, w=0.0, h=0.2)


def test_feature_keypoint_and_keypoints():
    p1 = FeatPoint(x=0.2, y=0.3)
    p2 = FeatPoint(x=0.6, y=0.7)
    assert p1.to_pixel(100, 200) == (20, 60)
    assert np.array_equal(p1.to_numpy(), np.array([0.2, 0.3]))
    p_from = FeatPoint.from_pixel(20, 60, 100, 200)
    assert p_from.x == pytest.approx(0.2) and p_from.y == pytest.approx(0.3)

    kp1 = Keypoint(name="nose", point=p1, confidence=0.9)
    kp2 = Keypoint(name="left_eye", point=p2, confidence=0.8)
    assert kp1.to_pixel(100, 200) == (20, 60)

    kps = Keypoints(items=[kp1, kp2])
    arr = kps.as_numpy()
    assert arr.shape == (2, 3)

    # compute bbox from keypoints
    computed_bbox = kps.as_bbox()
    assert isinstance(computed_bbox, BoundingBox)
    assert computed_bbox.w > 0 and computed_bbox.h > 0

    # Features container
    features = Features(keypoints=kps, depth=2.5, attributes={"gender": "male"})
    assert features.depth == 2.5
    assert features.attributes["gender"] == "male"


def test_detection_schema():
    bbox = BoundingBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
    det = Detection(bbox=bbox, confidence=0.85, class_id=0, class_name="person")
    assert det.class_name == "person"
    assert det.class_id == 0
    assert det.confidence == 0.85

    with pytest.raises(ValueError):
        Detection(bbox=bbox, confidence=-0.1, class_id=0, class_name="person")


def test_detection_source():
    DetectionSource.register("thermal")
    DetectionSource.register("rgb")
    with pytest.raises(ValueError):
        DetectionSource.validate("thermal")
    assert DetectionSource.validate("unique_source_test") == "unique_source_test"


def test_track_schema():
    bbox = BoundingBox(cx=0.5, cy=0.5, w=0.1, h=0.1)
    gid = uuid4()
    ident = Identity(global_id=gid, label="Alice")
    track = Track(tracker_id=1, bbox=bbox, class_name="person", identity=ident, confidence=0.9, age=5)
    assert track.global_id == gid
    assert track.label == "Alice"

    # Trajectory
    track.history.append(bbox)
    assert track.trajectory == [(0.5, 0.5)]

    # Invalid post init
    with pytest.raises(ValueError):
        Track(tracker_id=2, bbox=bbox, class_name="person", confidence=2.0)
    with pytest.raises(ValueError):
        Track(tracker_id=3, bbox=bbox, class_name="person", age=-1)


def test_identity_schema():
    gid = uuid4()
    ident = Identity(global_id=gid, label="Bob")
    vec = np.ones((128,), dtype=np.float32)
    evicted = ident.add_embedding("body", vec)
    assert evicted is None
    assert "body" in ident.embeddings
    assert len(ident.embeddings["body"]) == 1
    assert np.array_equal(ident.embeddings["body"][0], vec)

    # best_similarity
    sim = ident.best_similarity("body", vec)
    assert sim == pytest.approx(1.0)
    assert ident.best_similarity("non_existent", vec) == 0.0

    # FIFO eviction
    for i in range(15):
        v = np.full((128,), float(i), dtype=np.float32)
        ident.add_embedding("body", v, max=5)
    assert len(ident.embeddings["body"]) == 5


def test_event_schema():
    from trackey.core.events.types import CountExceededEvent
    now = datetime.now(timezone.utc)
    ev = BaseEvent(frame_id=10, camera_id="c1", timestamp=now)
    assert ev.frame_id == 10
    assert ev.camera_id == "c1"
    assert ev.schema_version == "1.0"
    assert isinstance(ev.event_id, str)

    # EventDefinition.build
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    ctx = FrameContext(frame=Frame(frame=arr), frame_id=42, camera_id="cam_main", analytics={"count": 5})
    defn = EventDefinition(event_type=CountExceededEvent, extract={"count": "analytics.count"})
    built = defn.build(ctx)
    assert built.frame_id == 42
    assert built.camera_id == "cam_main"
    assert built.count == 5


def test_pipeline_result():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    ctx = FrameContext(frame=Frame(frame=arr), frame_id=1, camera_id="cam1")
    res = PipelineResult.from_context(ctx, rendered=arr)
    assert res.frame_id == 1
    assert res.camera_id == "cam1"
    assert res.rendered_frame is not None
    assert res.raw_frame is not None


def test_geometry_schemas():
    p = GeoPoint(x=0.5, y=0.5)
    x, y = p
    assert x == 0.5 and y == 0.5

    poly = Polygon(points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    zone = Zone(name="Z1", polygon=poly)
    assert zone.contains((0.5, 0.5)) is True
    assert zone.contains((1.5, 1.5)) is False

    line = Line(name="L1", start=(0.0, 0.5), end=(1.0, 0.5))
    # line crossed from (0.5, 0.0) to (0.5, 1.0)
    assert line.crossed((0.5, 0.0), (0.5, 1.0)) is True
    # no cross
    assert line.crossed((0.5, 0.1), (0.5, 0.2)) is False
    # parallel / collinear / zero denom
    assert line.crossed((0.0, 0.5), (1.0, 0.5)) is False
