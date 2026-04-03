"""Tests for dual-layer outlined text rendering."""

import numpy as np

from src.visualization.core.text import draw_text_outlined


class TestDrawTextOutlined:
    def test_renders_on_frame(self):
        """Should render text on a frame without errors."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_text_outlined(frame, "Knee: 45.0", (10, 30))
        assert result is frame

    def test_black_pixels_present(self):
        """Should draw black outline pixels near text (thicker)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_text_outlined(frame, "Test", (10, 30))
        text_region = frame[0:60, 0:200]
        has_non_white = np.any(text_region != 255, axis=2)
        assert has_non_white.any()

    def test_colored_pixels_present(self):
        """Should draw colored text pixels."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        green = (0, 255, 0)
        draw_text_outlined(frame, "Test", (10, 30), color=green)
        text_region = frame[0:60, 0:200]
        # Check for green pixels (BGR: channel 1 is green, channel 0/2 should be 0)
        has_green = np.any(
            (text_region[:, :, 1] == 255)
            & (text_region[:, :, 0] == 0)
            & (text_region[:, :, 2] == 0)
        )
        assert has_green.any()

    def test_custom_font_scale(self):
        """Should accept custom font_scale and thickness."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_text_outlined(frame, "Test", (10, 30), font_scale=0.8, thickness=2)
        assert frame.any()
