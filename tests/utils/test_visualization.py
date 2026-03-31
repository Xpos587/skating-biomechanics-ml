"""Tests for visualization utilities.

Updated for modular visualization API (H3.6M 17kp format).
"""

import numpy as np

from src.types import H36Key
from src.visualization import (
    BladeLayer,
    LayerContext,
    TrailLayer,
    VelocityLayer,
    draw_skeleton,
    draw_text_box,
    render_cyrillic_text,
)


class TestDrawSkeleton:
    """Tests for draw_skeleton function."""

    def test_draw_skeleton_normalized_coords(self):
        """Test skeleton drawing with normalized coordinates (H3.6M 17kp)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.random.rand(17, 2).astype(np.float32)

        result = draw_skeleton(frame, keypoints, 480, 640)

        assert np.any(result > 0)

    def test_draw_skeleton_pixel_coords(self):
        """Test skeleton drawing with pixel coordinates."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.random.rand(17, 3) * [640, 480, 1]
        keypoints = keypoints.astype(np.float32)

        result = draw_skeleton(frame, keypoints, 480, 640)

        assert np.any(result > 0)

    def test_draw_skeleton_invalid_points(self):
        """Test that invalid points (at origin) are skipped."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.zeros((17, 2), dtype=np.float32)
        keypoints[H36Key.LSHOULDER] = [0.5, 0.5]
        keypoints[H36Key.RSHOULDER] = [0.6, 0.5]

        result = draw_skeleton(frame, keypoints, 480, 640)

        assert np.any(result > 0)


class TestVelocityLayer:
    """Tests for VelocityLayer."""

    def test_velocity_layer_middle_frame(self):
        """Test velocity rendering for middle frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VelocityLayer()

        # First frame (no previous) — should not crash
        ctx1 = LayerContext(
            frame_width=640, frame_height=480, fps=25.0,
            pose_2d=np.random.rand(17, 2).astype(np.float32),
            normalized=True,
        )
        result = layer.render(frame, ctx1)
        assert result.shape == (480, 640, 3)

        # Second frame — should draw velocity
        ctx2 = LayerContext(
            frame_width=640, frame_height=480, fps=25.0,
            pose_2d=np.random.rand(17, 2).astype(np.float32),
            normalized=True,
        )
        result = layer.render(frame, ctx2)
        assert np.any(result > 0)

    def test_velocity_layer_first_frame(self):
        """Test velocity rendering for first frame (no velocity)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VelocityLayer()

        ctx = LayerContext(
            frame_width=640, frame_height=480, fps=25.0,
            pose_2d=np.random.rand(17, 2).astype(np.float32),
            normalized=True,
        )
        result = layer.render(frame, ctx)
        assert result.shape == (480, 640, 3)


class TestTrailLayer:
    """Tests for TrailLayer."""

    def test_trail_layer(self):
        """Test trail rendering."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = TrailLayer(length=10)

        for i in range(10):
            ctx = LayerContext(
                frame_width=640, frame_height=480, fps=25.0,
                pose_2d=np.random.rand(17, 2).astype(np.float32),
                normalized=True,
            )
            frame = layer.render(frame, ctx)

        assert frame.shape == (480, 640, 3)


class TestBladeLayer:
    """Tests for BladeLayer."""

    def test_blade_layer(self):
        """Test blade layer rendering."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = BladeLayer()

        ctx = LayerContext(
            frame_width=640, frame_height=480, fps=25.0,
        )
        result = layer.render(frame, ctx)
        assert result.shape == (480, 640, 3)


class TestDrawTextBox:
    """Tests for draw_text_box function."""

    def test_draw_text_box(self):
        """Test text box drawing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_text_box(
            frame,
            text="Test Text",
            position=(50, 50),
            font_scale=1.0,
        )

        assert np.any(frame > 0)
        assert frame.shape == (480, 640, 3)


class TestRenderCyrillicText:
    """Tests for render_cyrillic_text function."""

    def test_render_cyrillic_text(self):
        """Test Cyrillic text rendering."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = render_cyrillic_text(
            frame,
            text="Тестовый текст",
            position=(50, 50),
        )

        assert result.shape == (480, 640, 3)

    def test_render_cyrillic_empty_text(self):
        """Test with empty text."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = render_cyrillic_text(
            frame,
            text="",
            position=(50, 50),
        )

        assert result.shape == (480, 640, 3)
