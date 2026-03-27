"""Tests for visualization utilities."""

import numpy as np

from skating_biomechanics_ml.types import BKey
from skating_biomechanics_ml.utils import (
    draw_debug_hud,
    draw_edge_indicators,
    draw_skeleton,
    draw_subtitle_cyrillic,
    draw_text_box,
    draw_trails,
    draw_velocity_vectors,
)


class TestDrawSkeleton:
    """Tests for draw_skeleton function."""

    def test_draw_skeleton_normalized_coords(self):
        """Test skeleton drawing with normalized coordinates."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.random.rand(33, 2).astype(np.float32)

        result = draw_skeleton(frame, keypoints, 480, 640)

        # Check that skeleton was drawn (non-zero pixels)
        assert np.any(result > 0)

    def test_draw_skeleton_pixel_coords(self):
        """Test skeleton drawing with pixel coordinates."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.random.rand(33, 3) * [640, 480, 1]
        keypoints = keypoints.astype(np.float32)

        result = draw_skeleton(frame, keypoints, 480, 640)

        assert np.any(result > 0)

    def test_draw_skeleton_invalid_points(self):
        """Test that invalid points (at origin) are skipped."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.zeros((33, 2), dtype=np.float32)
        # Set only a few valid points
        keypoints[BKey.LEFT_SHOULDER] = [0.5, 0.5]
        keypoints[BKey.RIGHT_SHOULDER] = [0.6, 0.5]

        result = draw_skeleton(frame, keypoints, 480, 640)

        # Should still draw something for the valid points
        assert np.any(result > 0)


class TestDrawVelocityVectors:
    """Tests for draw_velocity_vectors function."""

    def test_draw_velocity_vectors_middle_frame(self):
        """Test velocity vector drawing for middle frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = np.random.rand(100, 33, 2).astype(np.float32)

        result = draw_velocity_vectors(frame, poses, 50, 25.0, 480, 640)

        # Velocity vectors should be drawn (check for non-zero pixels)
        assert np.any(result > 0)

    def test_draw_velocity_vectors_first_frame(self):
        """Test velocity vector drawing for first frame (no velocity)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = np.random.rand(100, 33, 2).astype(np.float32)

        result = draw_velocity_vectors(frame, poses, 0, 25.0, 480, 640)

        # First frame has no velocity (no previous frame)
        # Should still return valid frame
        assert result.shape == (480, 640, 3)

    def test_draw_velocity_vectors_custom_joints(self):
        """Test velocity vectors for specific joints."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = np.random.rand(100, 33, 2).astype(np.float32)

        result = draw_velocity_vectors(
            frame, poses, 50, 25.0, 480, 640, joint_indices=[BKey.LEFT_HIP]
        )

        assert np.any(result > 0)


class TestDrawTrails:
    """Tests for draw_trails function."""

    def test_draw_trails(self):
        """Test trail drawing."""
        from collections import deque  # noqa: PLC0415

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trail_history = deque(maxlen=10)

        # Add some poses to history
        for _ in range(5):
            trail_history.append(np.random.rand(33, 2).astype(np.float32))

        result = draw_trails(frame, trail_history, BKey.LEFT_ANKLE, 480, 640)

        assert np.any(result > 0)

    def test_draw_trails_empty_history(self):
        """Test trail drawing with empty history."""
        from collections import deque  # noqa: PLC0415

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trail_history = deque(maxlen=10)

        result = draw_trails(frame, trail_history, BKey.LEFT_ANKLE, 480, 640)

        # Should return unchanged frame
        assert np.array_equal(result, frame)

    def test_draw_trails_single_point(self):
        """Test trail drawing with only one point."""
        from collections import deque  # noqa: PLC0415

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trail_history = deque(maxlen=10)
        trail_history.append(np.random.rand(33, 2).astype(np.float32))

        result = draw_trails(frame, trail_history, BKey.LEFT_ANKLE, 480, 640)

        # Should return unchanged frame (need at least 2 points)
        assert np.array_equal(result, frame)


class TestDrawEdgeIndicators:
    """Tests for draw_edge_indicators function."""

    def test_draw_edge_indicators(self):
        """Test edge indicator drawing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = np.random.rand(100, 33, 2).astype(np.float32)

        result = draw_edge_indicators(frame, poses, 50, 480, 640)

        assert np.any(result > 0)

    def test_draw_edge_indicators_specific_edge(self):
        """Test edge indicator with specific edge value."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = np.zeros((10, 33, 2), dtype=np.float32)

        # Set up left foot for inside edge (positive x-component)
        poses[5, BKey.LEFT_HEEL] = [0.3, 0.7]
        poses[5, BKey.LEFT_FOOT_INDEX] = [0.4, 0.7]

        result = draw_edge_indicators(frame, poses, 5, 480, 640)

        assert np.any(result > 0)


class TestDrawSubtitleCyrillic:
    """Tests for draw_subtitle_cyrillic function."""

    def test_draw_subtitle_latin(self):
        """Test drawing Latin text."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_subtitle_cyrillic(frame, "Test Text", (50, 400))

        assert not np.array_equal(result, frame)

    def test_draw_subtitle_cyrillic(self):
        """Test drawing Cyrillic text."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_subtitle_cyrillic(frame, "Теперь делаем тройку", (50, 400))

        assert not np.array_equal(result, frame)

    def test_draw_subtitle_with_custom_font_size(self):
        """Test drawing with custom font size."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_subtitle_cyrillic(frame, "Test", (50, 400), font_size=50)

        assert not np.array_equal(result, frame)


class TestDrawDebugHud:
    """Tests for draw_debug_hud function."""

    def test_draw_hud_full(self):
        """Test HUD with all information."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        element_info = {"type": "three_turn", "start": 10, "end": 50, "confidence": 0.85}
        kinematics = {"hip_velocity": 1.2, "left_knee": 125.5, "right_knee": 122.3}

        result = draw_debug_hud(
            frame, element_info, kinematics, 25, 100, 25.0, 480, 640
        )

        assert np.any(result > 0)

    def test_draw_hud_minimal(self):
        """Test HUD with minimal information."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_debug_hud(frame, {}, {}, 25, 100, 25.0, 480, 640)

        assert np.any(result > 0)

    def test_draw_hud_no_kinematics(self):
        """Test HUD with element info but no kinematics."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        element_info = {"type": "waltz_jump", "start": 0, "end": 30, "confidence": 0.9}

        result = draw_debug_hud(frame, element_info, {}, 15, 100, 25.0, 480, 640)

        assert np.any(result > 0)


class TestDrawTextBox:
    """Tests for draw_text_box helper function."""

    def test_draw_text_box(self):
        """Test drawing text with background box."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_text_box(frame, "Test Label", (50, 100))

        assert not np.array_equal(frame, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_draw_text_box_custom_alpha(self):
        """Test drawing with custom background alpha."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_text_box(frame, "Test", (50, 100), bg_alpha=0.8)

        assert not np.array_equal(frame, np.zeros((480, 640, 3), dtype=np.uint8))
