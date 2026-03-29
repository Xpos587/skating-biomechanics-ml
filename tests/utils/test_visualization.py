"""Tests for visualization utilities.

Updated for H3.6M 17kp format.
"""

import numpy as np

from src.types import H36Key
from src.visualization import (
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
        """Test skeleton drawing with normalized coordinates (H3.6M 17kp)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.random.rand(17, 2).astype(np.float32)  # H3.6M format

        result = draw_skeleton(frame, keypoints, 480, 640)

        # Check that skeleton was drawn (non-zero pixels)
        assert np.any(result > 0)

    def test_draw_skeleton_pixel_coords(self):
        """Test skeleton drawing with pixel coordinates."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.random.rand(17, 3) * [640, 480, 1]  # H3.6M format
        keypoints = keypoints.astype(np.float32)

        result = draw_skeleton(frame, keypoints, 480, 640)

        assert np.any(result > 0)

    def test_draw_skeleton_invalid_points(self):
        """Test that invalid points (at origin) are skipped."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.zeros((17, 2), dtype=np.float32)  # H3.6M format
        # Set only a few valid points
        keypoints[H36Key.LSHOULDER] = [0.5, 0.5]
        keypoints[H36Key.RSHOULDER] = [0.6, 0.5]

        result = draw_skeleton(frame, keypoints, 480, 640)

        # Should still draw something for the valid points
        assert np.any(result > 0)


class TestDrawVelocityVectors:
    """Tests for draw_velocity_vectors function."""

    def test_draw_velocity_vectors_middle_frame(self):
        """Test velocity vector drawing for middle frame (H3.6M 17kp)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = np.random.rand(100, 17, 2).astype(np.float32)  # H3.6M format

        result = draw_velocity_vectors(frame, poses, 50, 25.0, 480, 640)

        # Velocity vectors should be drawn (check for non-zero pixels)
        assert np.any(result > 0)

    def test_draw_velocity_vectors_first_frame(self):
        """Test velocity vector drawing for first frame (no velocity)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = np.random.rand(100, 17, 2).astype(np.float32)  # H3.6M format

        result = draw_velocity_vectors(frame, poses, 0, 25.0, 480, 640)

        # First frame has no velocity (no previous frame)
        # Should still return valid frame
        assert result.shape == (480, 640, 3)

    def test_draw_velocity_vectors_custom_joints(self):
        """Test velocity vectors for specific joints."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = np.random.rand(100, 17, 2).astype(np.float32)  # H3.6M format

        result = draw_velocity_vectors(
            frame, poses, 50, 25.0, 480, 640, joint_indices=[H36Key.LHIP]
        )

        assert np.any(result > 0)


class TestDrawTrails:
    """Tests for draw_trails function."""

    def test_draw_trails(self):
        """Test trail drawing."""
        from collections import deque  # noqa: PLC0415

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trail_history = deque(maxlen=10)

        # Add some poses to history (H3.6M 17kp format)
        for i in range(10):
            pose = np.random.rand(17, 2).astype(np.float32)
            trail_history.append(pose)

        # Correct argument order: frame, pose_history, joint_idx, height, width
        result = draw_trails(frame, trail_history, joint_idx=0, height=480, width=640)

        assert result.shape == (480, 640, 3)


class TestDrawEdgeIndicators:
    """Tests for draw_edge_indicators function."""

    def test_draw_edge_indicators(self):
        """Test edge indicator drawing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create mock poses (H3.6M 17kp format)
        poses = np.random.rand(10, 17, 2).astype(np.float32)

        # Function computes edges from poses automatically
        result = draw_edge_indicators(
            frame,
            poses,
            frame_idx=5,
            height=480,
            width=640,
        )

        assert result.shape == (480, 640, 3)


class TestDrawDebugHud:
    """Tests for draw_debug_hud function."""

    def test_draw_debug_hud(self):
        """Test debug HUD drawing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        element_info = {
            "type": "waltz_jump",
            "start": 0,
            "end": 100,
            "confidence": 0.9,
        }

        kinematics = {
            "airtime": 0.5,
            "height": 0.3,
            "knee_angle": 120.0,
        }

        result = draw_debug_hud(
            frame,
            element_info=element_info,
            kinematics=kinematics,
            frame_idx=50,
            total_frames=100,
            fps=30.0,
            height=480,
            width=640,
        )

        assert result.shape == (480, 640, 3)


class TestDrawTextBox:
    """Tests for draw_text_box function."""

    def test_draw_text_box(self):
        """Test text box drawing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # draw_text_box modifies frame in-place and returns None
        draw_text_box(
            frame,
            text="Test Text",
            position=(50, 50),
            font_scale=1.0,
        )

        # Frame should be modified (non-zero pixels from text)
        assert np.any(frame > 0)
        assert frame.shape == (480, 640, 3)


class TestDrawSubtitleCyrillic:
    """Tests for draw_subtitle_cyrillic function."""

    def test_draw_subtitle_cyrillic(self):
        """Test Cyrillic subtitle drawing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_subtitle_cyrillic(
            frame,
            text="Тестовый текст",  # "Test text" in Russian
            position=(50, 50),
        )

        assert result.shape == (480, 640, 3)

    def test_draw_subtitle_empty_text(self):
        """Test with empty text."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_subtitle_cyrillic(
            frame,
            text="",
            position=(50, 50),
        )

        assert result.shape == (480, 640, 3)
