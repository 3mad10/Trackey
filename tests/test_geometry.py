"""Tests for Polygon.contains() ray-casting implementation."""
from trackey.data.schemas.geometry import Polygon

SQUARE = Polygon(points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def test_point_inside():
    assert SQUARE.contains((0.5, 0.5)) is True


def test_point_outside():
    assert SQUARE.contains((2.0, 2.0)) is False


def test_point_outside_negative():
    assert SQUARE.contains((-0.1, 0.5)) is False


def test_point_on_top_edge():
    # Verify no crash on horizontal or edge boundary
    result = SQUARE.contains((0.5, 1.0))
    assert isinstance(result, bool)


def test_point_on_bottom_edge():
    result = SQUARE.contains((0.5, 0.0))
    assert isinstance(result, bool)


def test_point_on_left_edge():
    result = SQUARE.contains((0.0, 0.5))
    assert isinstance(result, bool)


def test_far_outside():
    assert SQUARE.contains((100.0, 100.0)) is False


def test_triangle_inside():
    tri = Polygon(points=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
    assert tri.contains((0.5, 0.4)) is True


def test_triangle_outside():
    tri = Polygon(points=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
    assert tri.contains((0.9, 0.9)) is False
