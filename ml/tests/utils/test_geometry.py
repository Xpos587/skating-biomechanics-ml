"""Tests for geometry utility functions."""

import numpy as np

from skating_ml.types import H36Key
from skating_ml.utils.geometry import (
    calculate_center_of_mass,
    calculate_com_trajectory,
    detect_visible_side,
    estimate_floor_angle,
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
        # Create perfectly symmetric pose
        poses = np.zeros((1, 33, 2), dtype=np.float32)

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
        poses[0, H36Key.LEFT_ANKLE] = [-0.05, 0.6]
        poses[0, H36Key.RIGHT_ANKLE] = [0.05, 0.6]

        # Add head (nose) - this contributes to head mass
        poses[0, H36Key.NOSE] = [0, -0.5]

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
        # Create two poses with same CoM but different hip heights
        poses_straight = np.zeros((1, 33, 2), dtype=np.float32)
        poses_bent = np.zeros((1, 33, 2), dtype=np.float32)

        # Same upper body
        for pose in [poses_straight, poses_bent]:
            pose[0, H36Key.LEFT_HIP] = [-0.05, 0.0]
            pose[0, H36Key.RIGHT_HIP] = [0.05, 0.0]
            pose[0, H36Key.LEFT_SHOULDER] = [-0.1, -0.3]
            pose[0, H36Key.RIGHT_SHOULDER] = [0.1, -0.3]
            pose[0, H36Key.NOSE] = [0, -0.5]

        # Straight legs (hips high)
        poses_straight[0, H36Key.LEFT_KNEE] = [-0.05, 0.3]
        poses_straight[0, H36Key.RIGHT_KNEE] = [0.05, 0.3]
        poses_straight[0, H36Key.LEFT_ANKLE] = [-0.05, 0.6]
        poses_straight[0, H36Key.RIGHT_ANKLE] = [0.05, 0.6]

        # Bent knees (hips lower relative to ankles)
        # Ankles same position, but knees bend downward
        poses_bent[0, H36Key.LEFT_KNEE] = [-0.05, 0.45]
        poses_bent[0, H36Key.RIGHT_KNEE] = [0.05, 0.45]
        poses_bent[0, H36Key.LEFT_ANKLE] = [-0.05, 0.6]
        poses_bent[0, H36Key.RIGHT_ANKLE] = [0.05, 0.6]

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


class TestDetectVisibleSide:
    """Tests for detect_visible_side function."""

    def test_right_side_visible(self):
        """Right side visible when toes are right of heels."""
        # RBigToe.x > RHeel.x → right side
        foot_kp = np.array(
            [
                [
                    [100, 200, 0.9],  # L_Heel
                    [150, 200, 0.9],  # L_BigToe
                    [140, 195, 0.5],  # L_SmallToe
                    [300, 200, 0.9],  # R_Heel
                    [350, 200, 0.9],  # R_BigToe  → toe right of heel
                    [340, 195, 0.5],  # R_SmallToe
                ]
            ],
            dtype=np.float32,
        )  # shape (1, 6, 3)
        assert detect_visible_side(foot_kp) == "right"

    def test_left_side_visible(self):
        """Left side visible when toes are left of heels."""
        foot_kp = np.array(
            [
                [
                    [300, 200, 0.9],  # L_Heel
                    [250, 200, 0.9],  # L_BigToe  → toe left of heel
                    [260, 195, 0.5],  # L_SmallToe
                    [100, 200, 0.9],  # R_Heel
                    [50, 200, 0.9],  # R_BigToe  → toe left of heel
                    [60, 195, 0.5],  # R_SmallToe
                ]
            ],
            dtype=np.float32,
        )
        assert detect_visible_side(foot_kp) == "left"

    def test_no_confidence_returns_none(self):
        """Should return None when all foot keypoints have low confidence."""
        foot_kp = np.zeros((1, 6, 3), dtype=np.float32)
        assert detect_visible_side(foot_kp) is None

    def test_multi_frame_aggregation(self):
        """Should aggregate across multiple frames using median."""
        foot_kp = np.array(
            [
                [  # Frame 0: right side
                    [100, 200, 0.9],  # L_Heel
                    [150, 200, 0.9],  # L_BigToe
                    [140, 195, 0.5],  # L_SmallToe
                    [300, 200, 0.9],  # R_Heel
                    [350, 200, 0.9],  # R_BigToe
                    [340, 195, 0.5],  # R_SmallToe
                ],
                [  # Frame 1: right side
                    [110, 210, 0.9],  # L_Heel
                    [160, 210, 0.9],  # L_BigToe
                    [150, 205, 0.5],  # L_SmallToe
                    [310, 210, 0.9],  # R_Heel
                    [360, 210, 0.9],  # R_BigToe
                    [350, 205, 0.5],  # R_SmallToe
                ],
                [  # Frame 2: left side (outlier)
                    [300, 200, 0.9],  # L_Heel
                    [250, 200, 0.9],  # L_BigToe
                    [260, 195, 0.5],  # L_SmallToe
                    [100, 200, 0.9],  # R_Heel
                    [50, 200, 0.9],  # R_BigToe
                    [60, 195, 0.5],  # R_SmallToe
                ],
            ],
            dtype=np.float32,
        )  # shape (3, 6, 3)
        # 2/3 frames say right → median should be right
        assert detect_visible_side(foot_kp) == "right"


class TestEstimateFloorAngle:
    """Tests for estimate_floor_angle function."""

    def test_level_floor(self):
        """Horizontal foot positions should give ~0° angle."""
        positions = np.array(
            [
                [100.0, 200.0],
                [200.0, 200.0],
                [300.0, 200.0],
                [400.0, 200.0],
            ]
        )
        angle = estimate_floor_angle(positions)
        assert abs(angle) < 1.0

    def test_tilted_floor(self):
        """Consistently rising Y should give positive angle."""
        positions = np.array(
            [
                [100.0, 200.0],
                [200.0, 210.0],
                [300.0, 220.0],
                [400.0, 230.0],
            ]
        )
        angle = estimate_floor_angle(positions)
        assert angle > 0

    def test_single_point_returns_zero(self):
        """Single point should return 0 (no line to fit)."""
        positions = np.array([[100.0, 200.0]])
        angle = estimate_floor_angle(positions)
        assert angle == 0.0

    def test_negative_tilt(self):
        """Consistently falling Y should give negative angle."""
        positions = np.array(
            [
                [100.0, 230.0],
                [200.0, 220.0],
                [300.0, 210.0],
                [400.0, 200.0],
            ]
        )
        angle = estimate_floor_angle(positions)
        assert angle < 0

    def test_steep_positive_tilt(self):
        """Steep upward slope should give large positive angle."""
        positions = np.array(
            [
                [100.0, 200.0],
                [200.0, 300.0],
                [300.0, 400.0],
                [400.0, 500.0],
            ]
        )
        angle = estimate_floor_angle(positions)
        # 45° slope (dy/dx = 1) should give ~45° angle
        assert 40 < angle < 50
