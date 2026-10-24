"""Tests for the Counter analyzer."""
import numpy as np
import pytest

from trackey.data.schemas.frame import Frame
from trackey.data.schemas.detection import BoundingBox
from trackey.data.schemas.track import Track
from trackey.core.analyzers.counter import Counter


def make_frame() -> Frame:
    return Frame(frame=np.zeros((100, 100, 3), dtype=np.uint8))


def make_track(tracker_id: int, class_name: str = "person", with_history: bool = True) -> Track:
    bbox = BoundingBox(cx=0.5, cy=0.5, w=0.1, h=0.1)
    t = Track(tracker_id=tracker_id, bbox=bbox, class_name=class_name)
    if with_history:
        t.history.append(bbox)
    return t


def test_counter_counts_single_class():
    counter = Counter(target_classes=["person"])
    tracks = [make_track(1), make_track(2), make_track(3)]
    result = counter.analyze(tracks, frame=make_frame())
    assert result["person"]["current"] == 3


def test_counter_ignores_other_classes():
    counter = Counter(target_classes=["person"])
    tracks = [make_track(1, "person"), make_track(2, "car"), make_track(3, "car")]
    result = counter.analyze(tracks, frame=make_frame())
    assert result.get("person", {}).get("current", 0) == 1
    assert "car" not in result


def test_counter_no_filter_counts_all():
    counter = Counter(target_classes=None)
    tracks = [
        make_track(1, "person"),
        make_track(2, "car"),
        make_track(3, "person"),
    ]
    result = counter.analyze(tracks, frame=make_frame())
    assert result["person"]["current"] == 2
    assert result["car"]["current"] == 1


def test_counter_skips_tracks_without_history():
    counter = Counter()
    tracks = [
        make_track(1, with_history=True),
        make_track(2, with_history=False),
    ]
    result = counter.analyze(tracks, frame=make_frame())
    assert result.get("person", {}).get("current", 0) == 1


def test_counter_peak_metric():
    counter = Counter(metrics=["current", "peak"])
    tracks_a = [make_track(i) for i in range(3)]
    counter.analyze(tracks_a, frame=make_frame())
    tracks_b = [make_track(10)]
    result = counter.analyze(tracks_b, frame=make_frame())
    assert result["person"]["peak"] == 3
    assert result["person"]["current"] == 1


def test_counter_cumulative_metric():
    counter = Counter(metrics=["current", "cumulative"])
    counter.analyze([make_track(1), make_track(2)], frame=make_frame())
    result = counter.analyze([make_track(2), make_track(3)], frame=make_frame())
    assert result["person"]["cumulative"] == 3


def test_counter_empty_tracks():
    counter = Counter()
    result = counter.analyze([], frame=make_frame())
    assert result == {}


def test_counter_invalid_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        Counter(metrics=["invalid_metric"])


def test_counter_case_insensitive_class():
    counter = Counter(target_classes=["Person"])
    tracks = [make_track(1, "person"), make_track(2, "PERSON")]
    result = counter.analyze(tracks, frame=make_frame())
    assert result["person"]["current"] == 2
