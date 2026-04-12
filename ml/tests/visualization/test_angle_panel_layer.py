"""Tests for angle panel layer."""

import numpy as np

from skating_ml.visualization.layers.angle_panel_layer import AnglePanelLayer
from skating_ml.visualization.layers.base import LayerContext


def _make_context(angles=None, w=640, h=480):
    ctx = LayerContext(frame_width=w, frame_height=h)
    if angles:
        ctx.custom_data["angles"] = angles
    return ctx


class TestAnglePanelLayer:
    def test_renders_empty_panel(self):
        """Should render without error even with no angles."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = AnglePanelLayer()
        ctx = _make_context()
        result = layer.render(frame, ctx)
        assert result is frame

    def test_renders_with_angles(self):
        """Should render panel with angle values."""
        angles = {"L Knee": 120.5, "R Knee": 95.0, "L Hip": 45.0}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = AnglePanelLayer()
        ctx = _make_context(angles=angles)
        result = layer.render(frame, ctx)
        assert result is frame

    def test_modifies_frame_with_progress_bar(self):
        """Should modify frame when angles present (progress bars add pixels)."""
        angles = {"L Knee": 120.5, "R Knee": 95.0}
        frame_before = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_after = frame_before.copy()
        layer = AnglePanelLayer()
        ctx = _make_context(angles=angles)
        layer.render(frame_after, ctx)
        assert not np.array_equal(frame_before, frame_after)
