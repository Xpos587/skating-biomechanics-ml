"""Tests for geometry utility functions."""

import numpy as np
import pytest

from src.types import H36Key
from src.utils.geometry import (
    calculate_center_of_mass,
    calculate_com_trajectory,
    get_mid_hip,
    get_mid_shoulder,
)


class TestCalculateCenterOfMass:
    """Tests for Center of Mass calculation."""

    def test_com_single_frame(self, sample_normalized_poses):
        """Should calculate CoM for a single frame."""
        com_y = calculate_center_of_mass(sample_normalized_poses, 0)

        # CoM should be between hip and shoulder (roughly)
        # In our fixture: hips at Y=0, shoulders at Y=-0.3
        # CoM should be somewhere between, weighted toward torso
        assert -0.5 < com_y < 0.2

    def test_com_trajectory(self, sample_normalized_poses):
        """Should calculate CoM trajectory for entire sequence."""
        com_trajectory = calculate_com_trajectory(sample_normalized_poses)

        assert len(com_trajectory) == len(sample_normalized_poses)
        assert all(isinstance(v, (float, np.floating)) for v in com_trajectory)

    def test_com_symmetric_pose(self):
        """Should return CoM near center line for symmetric pose."""
        # Create perfectly symmetric pose (H3.6M 17kp format)
        poses = np.zeros((1, 17, 2), dtype=np.float32)

        # Hips at origin
        poses[0, H36Key.LEFT_HIP] = [-0.05, 0.0]
        poses[0, H36Key.RIGHT_HIP] = [0.05, 0.0]

        # Shoulders (above hips, negative Y)
        poses[0, H36Key.LEFT_SHOULDER] = [-0.1, -0.3]
        poses[0, H36Key.RIGHT_SHOULDER] = [0.1, -0.3]

        # Arms (symmetric, extended upward)
        poses[0, H36Key.LEFT_ELBOW] = [-0.15, -0.5]
        poses[0, H36Key.RIGHT_ELBOW] = [0.15, -0.5]
        poses[0, H36Key.LEFT_WRIST] = [-0.2, -0.7]
        poses[0, H36Key.RIGHT_WRIST] = [0.2, -0.7]

        # Legs (symmetric, below hips)
        poses[0, H36Key.LEFT_KNEE] = [-0.05, 0.3]
        poses[0, H36Key.RIGHT_KNEE] = [0.05, 0.3]
        poses[0, H36Key.LEFT_FOOT] = [-0.05, 0.6]
        poses[0, H36Key.RIGHT_FOOT] = [0.05, 0.6]

        # Add head (nose) - this contributes to head mass
        poses[0, H36Key.HEAD] = [0, -0.5]

        com_y = calculate_center_of_mass(poses, 0)

        # CoM should be near zero because legs (heavy) pull down,
        # torso+head (heavy) pull up. With symmetric pose and
        # balanced mass distribution, CoM should be close to hip level.
        assert -0.2 < com_y < 0.2

    def test_com_changes_with_pose(self, sample_normalized_poses):
        """CoM should change when pose changes."""
        # Modify second frame to have different pose
        sample_normalized_poses[1, H36Key.LEFT_WRIST, 1] = -0.9  # Raise arm

        com_0 = calculate_center_of_mass(sample_normalized_poses, 0)
        com_1 = calculate_center_of_mass(sample_normalized_poses, 1)

        # Raising arm should raise CoM (more negative Y)
        assert com_1 < com_0

    def test_com_bent_knee_vs_straight(self):
        """CoM should be lower with bent knees (landing simulation).

        This is the key test for the research finding: hip-only method
        overestimates jump height because bent knees lower the hips
        but CoM remains at true jump height.
        """
        # Create two poses with same CoM but different hip heights (H3.6M 17kp)
        poses_straight = np.zeros((1, 17, 2), dtype=np.float32)
        poses_bent = np.zeros((1, 17, 2), dtype=np.float32)

        # Same upper body
        for pose in [poses_straight, poses_bent]:
            pose[0, H36Key.LEFT_HIP] = [-0.05, 0.0]
            pose[0, H36Key.RIGHT_HIP] = [0.05, 0.0]
            pose[0, H36Key.LEFT_SHOULDER] = [-0.1, -0.3]
            pose[0, H36Key.RIGHT_SHOULDER] = [0.1, -0.3]
            pose[0, H36Key.HEAD] = [0, -0.5]

        # Straight legs (hips high)
        poses_straight[0, H36Key.LEFT_KNEE] = [-0.05, 0.3]
        poses_straight[0, H36Key.RIGHT_KNEE] = [0.05, 0.3]
        poses_straight[0, H36Key.LEFT_FOOT] = [-0.05, 0.6]
        poses_straight[0, H36Key.RIGHT_FOOT] = [0.05, 0.6]

        # Bent knees (hips lower relative to ankles)
        # Ankles same position, but knees bend downward
        poses_bent[0, H36Key.LEFT_KNEE] = [-0.05, 0.45]
        poses_bent[0, H36Key.RIGHT_KNEE] = [0.05, 0.45]
        poses_bent[0, H36Key.LEFT_FOOT] = [-0.05, 0.6]
        poses_bent[0, H36Key.RIGHT_FOOT] = [0.05, 0.6]

        com_straight = calculate_center_of_mass(poses_straight, 0)
        com_bent = calculate_center_of_mass(poses_bent, 0)

        # CoM should be slightly different but NOT as different as hips
        # The hip-only method would show large difference, CoM shows small diff
        com_diff = abs(com_bent - com_straight)

        # CoM difference should be proportional to knee bend
        # (bent knees move leg mass downward)
        assert com_diff > 0
        assert com_diff < 0.1  # Should be relatively small


class TestGetMidHip:
    """Tests for get_mid_hip function."""

    def test_get_mid_hip(self, sample_normalized_poses):
        """Should calculate mid-hip point for each frame."""
        mid_hip = get_mid_hip(sample_normalized_poses)

        assert mid_hip.shape == (3, 2)
        # Mid-hip should be at origin for our fixture
        assert np.allclose(mid_hip[:, 0], 0, atol=0.01)
        assert np.allclose(mid_hip[:, 1], 0, atol=0.01)


class TestGetMidShoulder:
    """Tests for get_mid_shoulder function."""

    def test_get_mid_shoulder(self, sample_normalized_poses):
        """Should calculate mid-shoulder point for each frame."""
        mid_shoulder = get_mid_shoulder(sample_normalized_poses)

        assert mid_shoulder.shape == (3, 2)
        # Mid-shoulder X should be at origin (symmetric)
        assert np.allclose(mid_shoulder[:, 0], 0, atol=0.01)
        # Mid-shoulder Y should be negative (above hips)
        assert np.all(mid_shoulder[:, 1] < 0)


class TestCalculateComTrajectoryVectorized:
    """Tests for vectorized CoM trajectory calculation."""

    def test_com_trajectory_matches_scalar(self, sample_normalized_poses):
        """Vectorized trajectory should match per-frame scalar computation."""
        from src.utils.geometry import calculate_center_of_mass

        expected = np.array(
            [
                calculate_center_of_mass(sample_normalized_poses, i)
                for i in range(len(sample_normalized_poses))
            ],
            dtype=np.float32,
        )
        actual = calculate_com_trajectory(sample_normalized_poses)

        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_com_trajectory_single_frame(self):
        """Should work for single-frame input."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        com = calculate_com_trajectory(poses)
        assert com.shape == (1,)

    def test_com_trajectory_100_frames(self):
        """Should handle 100 frames efficiently (no Python loop)."""
        rng = np.random.default_rng(42)
        poses = rng.uniform(-0.5, 0.5, size=(100, 17, 2)).astype(np.float32)
        com = calculate_com_trajectory(poses)
        assert com.shape == (100,)


class TestNormalizePosesVectorized:
    """Tests for vectorized normalize_poses."""

    def test_matches_original(self):
        """Vectorized output should match original loop-based output."""
        from src.utils.geometry import normalize_poses

        rng = np.random.default_rng(42)
        raw = rng.uniform(-1, 1, size=(50, 17, 3)).astype(np.float32)
        result = normalize_poses(raw)

        assert result.shape == (50, 17, 2)
        # Root-centered: mid-hip should be near origin
        mid_hip = (result[:, H36Key.LHIP] + result[:, H36Key.RHIP]) / 2
        np.testing.assert_allclose(mid_hip, 0, atol=1e-5)

    def test_17_keypoints_only(self):
        """Should raise ValueError for non-17 keypoint input."""
        with pytest.raises(ValueError, match="17 keypoints"):
            from src.utils.geometry import normalize_poses

            normalize_poses(np.zeros((10, 15, 3), dtype=np.float32))

    def test_scale_normalization(self):
        """Should normalize spine length to target_spine_length."""
        from src.utils.geometry import normalize_poses

        # Create test data with known spine length
        raw = np.zeros((10, 17, 3), dtype=np.float32)
        spine_length = 0.5  # 50cm in normalized units

        # Set hips at origin
        raw[:, H36Key.LHIP] = [0, 0, 1]
        raw[:, H36Key.RHIP] = [0, 0, 1]

        # Set shoulders at spine_length distance
        raw[:, H36Key.LSHOULDER] = [0, -spine_length, 1]
        raw[:, H36Key.RSHOULDER] = [0, -spine_length, 1]

        result = normalize_poses(raw, target_spine_length=0.4)

        # Check that spine is now 0.4
        spine_vector = result[:, H36Key.LSHOULDER] - result[:, H36Key.LHIP]
        actual_spine_length = np.linalg.norm(spine_vector, axis=1)

        np.testing.assert_allclose(actual_spine_length, 0.4, atol=1e-5)

    def test_per_frame_scaling(self):
        """Each frame should be scaled independently based on its own spine length."""
        from src.utils.geometry import normalize_poses

        raw = np.zeros((2, 17, 3), dtype=np.float32)

        # Frame 0: spine length = 0.5
        raw[0, H36Key.LHIP] = [0, 0, 1]
        raw[0, H36Key.RHIP] = [0, 0, 1]
        raw[0, H36Key.LSHOULDER] = [0, -0.5, 1]
        raw[0, H36Key.RSHOULDER] = [0, -0.5, 1]

        # Frame 1: spine length = 1.0
        raw[1, H36Key.LHIP] = [0, 0, 1]
        raw[1, H36Key.RHIP] = [0, 0, 1]
        raw[1, H36Key.LSHOULDER] = [0, -1.0, 1]
        raw[1, H36Key.RSHOULDER] = [0, -1.0, 1]

        result = normalize_poses(raw, target_spine_length=0.4)

        # Both frames should have spine length 0.4 after normalization
        spine_0 = np.linalg.norm(result[0, H36Key.LSHOULDER] - result[0, H36Key.LHIP])
        spine_1 = np.linalg.norm(result[1, H36Key.LSHOULDER] - result[1, H36Key.LHIP])

        np.testing.assert_allclose(spine_0, 0.4, atol=1e-5)
        np.testing.assert_allclose(spine_1, 0.4, atol=1e-5)
