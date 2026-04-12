"""Tests for core overlay primitives."""

import numpy as np

from skating_ml.visualization.core.overlay import draw_overlay_rect


def _make_frame(h=100, w=200):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_draw_overlay_rect_modifies_roi():
    frame = _make_frame()
    result = draw_overlay_rect(frame, (10, 20, 60, 40), color=(0, 0, 0), alpha=0.5)
    assert result is frame  # in-place
    # ROI should be darker than original
    roi = frame[20:60, 10:70]
    assert roi.mean() < 128


def test_draw_overlay_rect_no_full_frame_copy():
    frame = _make_frame()
    original_id = id(frame)
    draw_overlay_rect(frame, (0, 0, 50, 50), color=(255, 255, 255), alpha=0.5)
    assert id(frame) == original_id  # no frame replacement


def test_draw_overlay_rect_clips_to_bounds():
    frame = _make_frame()
    # Rect extends beyond frame
    draw_overlay_rect(frame, (-10, -10, 300, 200), color=(0, 0, 0), alpha=1.0)
    assert frame.mean() == 0  # entire frame should be black


def test_draw_overlay_rect_alpha_zero_no_change():
    frame = _make_frame()
    original = frame.copy()
    draw_overlay_rect(frame, (10, 10, 50, 50), color=(255, 255, 255), alpha=0.0)
    np.testing.assert_array_equal(frame, original)


def test_draw_overlay_rect_with_border():
    frame = _make_frame()
    draw_overlay_rect(
        frame,
        (10, 20, 60, 40),
        color=(0, 0, 0),
        alpha=0.5,
        border_color=(0, 255, 0),
        border_thickness=2,
    )
    # Border pixels should be green
    assert not np.array_equal(frame[20, 10], [0, 0, 0])
