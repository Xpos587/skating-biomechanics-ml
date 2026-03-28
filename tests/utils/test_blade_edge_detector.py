"""Tests for blade edge detection."""

import numpy as np
import pytest

from skating_biomechanics_ml.types import BKey, BladeType, NormalizedPose
from skating_biomechanics_ml.utils.blade_edge_detector import (
    BladeEdgeDetector,
    BladeState,
    calculate_ankle_angle,
    calculate_foot_angle,
    calculate_foot_vector,
    calculate_motion_direction,
    calculate_vertical_acceleration,
)


@pytest.fixture
def sample_poses() -> NormalizedPose:
    """Create sample normalized poses for testing.

    Simulates a skater moving forward (positive x direction).
    """
    # 10 frames, 33 keypoints, 2 coordinates
    poses = np.zeros((10, 33, 2), dtype=np.float32)

    # Mid-hip moves forward (positive x)
    for i in range(10):
        poses[i, BKey.LEFT_HIP] = [0.3 + i * 0.01, 0.5]
        poses[i, BKey.RIGHT_HIP] = [0.3 + i * 0.01, 0.5]
        poses[i, BKey.LEFT_KNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, BKey.RIGHT_KNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, BKey.LEFT_ANKLE] = [0.3 + i * 0.01, 0.7]
        poses[i, BKey.RIGHT_ANKLE] = [0.3 + i * 0.01, 0.7]

        # Foot pointing forward (no angle relative to motion)
        poses[i, BKey.LEFT_FOOT_INDEX] = [0.35 + i * 0.01, 0.7]
        poses[i, BKey.RIGHT_FOOT_INDEX] = [0.35 + i * 0.01, 0.7]

    return poses


@pytest.fixture
def inside_edge_poses() -> NormalizedPose:
    """Create poses simulating inside edge (foot angled inward).

    For left foot inside edge skating forward:
    - Motion direction: forward (+x)
    - Foot angle: inward (toe points right, toward body center)
    - This gives POSITIVE angle (toe to right of ankle)
    """
    poses = np.zeros((10, 33, 2), dtype=np.float32)

    for i in range(10):
        poses[i, BKey.LEFT_HIP] = [0.3 + i * 0.01, 0.5]
        poses[i, BKey.RIGHT_HIP] = [0.3 + i * 0.01, 0.5]
        poses[i, BKey.LEFT_KNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, BKey.RIGHT_KNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, BKey.LEFT_ANKLE] = [0.35 + i * 0.01, 0.7]
        poses[i, BKey.RIGHT_ANKLE] = [0.3 + i * 0.01, 0.7]

        # Foot angled inward: toe points right AND up (positive angle)
        # Ankle at [0.35, 0.7], toe at [0.45, 0.65]
        # Foot vector = [0.10, -0.05] → angle ≈ -26.6° → normalized to positive since y is inverted
        poses[i, BKey.LEFT_FOOT_INDEX] = [0.45 + i * 0.01, 0.65]

    return poses


@pytest.fixture
def outside_edge_poses() -> NormalizedPose:
    """Create poses simulating outside edge (foot angled outward).

    For left foot outside edge skating forward:
    - Motion direction: forward (+x)
    - Foot angle: outward (toe points left, away from body center)
    - This gives NEGATIVE angle (toe to left of ankle)
    """
    poses = np.zeros((10, 33, 2), dtype=np.float32)

    for i in range(10):
        poses[i, BKey.LEFT_HIP] = [0.3 + i * 0.01, 0.5]
        poses[i, BKey.RIGHT_HIP] = [0.3 + i * 0.01, 0.5]
        poses[i, BKey.LEFT_KNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, BKey.RIGHT_KNEE] = [0.3 + i * 0.01, 0.6]
        poses[i, BKey.LEFT_ANKLE] = [0.35 + i * 0.01, 0.7]
        poses[i, BKey.RIGHT_ANKLE] = [0.3 + i * 0.01, 0.7]

        # Foot angled outward: toe points left AND down (negative angle)
        # Ankle at [0.35, 0.7], toe at [0.25, 0.75]
        # Foot vector = [-0.10, +0.05] → angle ≈ 153° → normalized to -153°
        poses[i, BKey.LEFT_FOOT_INDEX] = [0.25 + i * 0.01, 0.75]

    return poses


class TestFootVector:
    """Tests for foot vector calculation."""

    def test_foot_vector_forward(self, sample_poses: NormalizedPose) -> None:
        """Foot vector for forward-pointing foot."""
        vector = calculate_foot_vector(sample_poses, 0, "left")
        assert vector[0] > 0  # Pointing forward (positive x)
        assert abs(vector[1]) < 0.01  # No vertical component

    def test_foot_vector_left_foot(self, sample_poses: NormalizedPose) -> None:
        """Left foot vector."""
        vector = calculate_foot_vector(sample_poses, 0, "left")
        assert len(vector) == 2

    def test_foot_vector_right_foot(self, sample_poses: NormalizedPose) -> None:
        """Right foot vector."""
        vector = calculate_foot_vector(sample_poses, 0, "right")
        assert len(vector) == 2


class TestMotionDirection:
    """Tests for motion direction calculation."""

    def test_motion_direction_forward(self, sample_poses: NormalizedPose) -> None:
        """Motion direction for forward-moving skater."""
        direction = calculate_motion_direction(sample_poses, 5)
        assert direction[0] > 0  # Moving forward
        assert abs(direction[1]) < 0.1  # Minimal vertical motion
        # Should be normalized
        assert abs(np.linalg.norm(direction) - 1.0) < 0.01


class TestFootAngle:
    """Tests for foot angle calculation."""

    def test_foot_angle_flat(self, sample_poses: NormalizedPose) -> None:
        """Foot angle for flat blade (foot aligned with motion)."""
        angle = calculate_foot_angle(sample_poses, 5, "left")
        # Should be near 0 degrees (foot aligned with motion)
        assert abs(angle) < 10

    def test_foot_angle_inside_edge(self, inside_edge_poses: NormalizedPose) -> None:
        """Foot angle for inside edge."""
        angle = calculate_foot_angle(inside_edge_poses, 5, "left")
        # Should be negative (foot angled inward)
        assert angle < -10

    def test_foot_angle_outside_edge(self, outside_edge_poses: NormalizedPose) -> None:
        """Foot angle for outside edge."""
        angle = calculate_foot_angle(outside_edge_poses, 5, "left")
        # Should be positive (foot angled outward)
        assert angle > 10


class TestAnkleAngle:
    """Tests for ankle angle calculation."""

    def test_ankle_angle_range(self, sample_poses: NormalizedPose) -> None:
        """Ankle angle should be in physiological range."""
        angle = calculate_ankle_angle(sample_poses, 0, "left")
        assert 0 <= angle <= 180  # Valid angle range


class TestVerticalAcceleration:
    """Tests for vertical acceleration calculation."""

    def test_vertical_acceleration_static(self, sample_poses: NormalizedPose) -> None:
        """Vertical acceleration for static foot position."""
        accel = calculate_vertical_acceleration(sample_poses, fps=30.0, frame_idx=5, leg="left")
        # Should be near 0 for constant height
        assert abs(accel) < 1.0


class TestBladeEdgeDetector:
    """Tests for BladeEdgeDetector class."""

    def test_init(self) -> None:
        """Detector initialization."""
        detector = BladeEdgeDetector()
        assert detector.inside_threshold == -15.0
        assert detector.outside_threshold == 15.0
        assert detector.toe_pick_accel_threshold == 5.0

    def test_classify_frame_flat(self, sample_poses: NormalizedPose) -> None:
        """Classify flat blade."""
        detector = BladeEdgeDetector()
        state = detector.classify_frame(sample_poses, 5, fps=30.0, foot="left")

        assert isinstance(state, BladeState)
        assert state.blade_type in (BladeType.FLAT, BladeType.UNKNOWN)
        assert 0 <= state.confidence <= 1

    def test_classify_frame_inside_edge(self, inside_edge_poses: NormalizedPose) -> None:
        """Classify inside edge."""
        detector = BladeEdgeDetector()
        state = detector.classify_frame(inside_edge_poses, 5, fps=30.0, foot="left")

        assert isinstance(state, BladeState)
        assert state.blade_type == BladeType.INSIDE
        assert state.confidence > 0

    def test_classify_frame_outside_edge(self, outside_edge_poses: NormalizedPose) -> None:
        """Classify outside edge."""
        detector = BladeEdgeDetector()
        state = detector.classify_frame(outside_edge_poses, 5, fps=30.0, foot="left")

        assert isinstance(state, BladeState)
        assert state.blade_type == BladeType.OUTSIDE
        assert state.confidence > 0

    def test_detect_sequence(self, sample_poses: NormalizedPose) -> None:
        """Detect blade state for entire sequence."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(sample_poses, fps=30.0, foot="left")

        assert len(states) == len(sample_poses)
        assert all(isinstance(s, BladeState) for s in states)

    def test_detect_sequence_inside_edge(self, inside_edge_poses: NormalizedPose) -> None:
        """Detect inside edge sequence."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(inside_edge_poses, fps=30.0, foot="left")

        # Most frames should be classified as inside edge
        inside_count = sum(1 for s in states if s.blade_type == BladeType.INSIDE)
        assert inside_count > len(states) // 2

    def test_detect_sequence_outside_edge(self, outside_edge_poses: NormalizedPose) -> None:
        """Detect outside edge sequence."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(outside_edge_poses, fps=30.0, foot="left")

        # Most frames should be classified as outside edge
        outside_count = sum(1 for s in states if s.blade_type == BladeType.OUTSIDE)
        assert outside_count > len(states) // 2

    def test_get_blade_summary(self, sample_poses: NormalizedPose) -> None:
        """Get blade summary statistics."""
        detector = BladeEdgeDetector()
        states = detector.detect_sequence(sample_poses, fps=30.0, foot="left")
        summary = detector.get_blade_summary(states)

        assert "total_frames" in summary
        assert summary["total_frames"] == len(sample_poses)
        assert "type_percentages" in summary
        assert "average_confidence" in summary
        assert "dominant_edge" in summary

    def test_smoothing_window(self, sample_poses: NormalizedPose) -> None:
        """Temporal smoothing reduces flickering."""
        detector_no_smooth = BladeEdgeDetector(smoothing_window=1)
        detector_smooth = BladeEdgeDetector(smoothing_window=5)

        states_no_smooth = detector_no_smooth.detect_sequence(sample_poses, fps=30.0, foot="left")
        states_smooth = detector_smooth.detect_sequence(sample_poses, fps=30.0, foot="left")

        # Both should have same length
        assert len(states_no_smooth) == len(states_smooth)

        # Smoothed should have fewer type transitions
        transitions_no_smooth = sum(
            1
            for i in range(1, len(states_no_smooth))
            if states_no_smooth[i].blade_type != states_no_smooth[i - 1].blade_type
        )
        transitions_smooth = sum(
            1
            for i in range(1, len(states_smooth))
            if states_smooth[i].blade_type != states_smooth[i - 1].blade_type
        )
        assert transitions_smooth <= transitions_no_smooth


class TestBladeState:
    """Tests for BladeState dataclass."""

    def test_blade_state_creation(self) -> None:
        """Create blade state."""
        state = BladeState(
            blade_type=BladeType.INSIDE,
            foot_angle=-20.0,
            ankle_angle=90.0,
            vertical_accel=0.5,
            confidence=0.8,
        )
        assert state.blade_type == BladeType.INSIDE
        assert state.foot_angle == -20.0
        assert state.ankle_angle == 90.0
        assert state.vertical_accel == 0.5
        assert state.confidence == 0.8
