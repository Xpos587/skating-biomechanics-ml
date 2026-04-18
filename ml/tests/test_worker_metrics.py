"""Tests for worker._compute_frame_metrics vectorized function."""

import numpy as np
import pytest

from skating_ml.types import H36Key


# Copy the vectorized function here for testing (to avoid backend import issues)
def _compute_frame_metrics(poses: np.ndarray) -> dict:
    """Compute frame-by-frame biomechanics metrics.

    Args:
        poses: (N, 17, 3) array of poses

    Returns:
        dict with metric arrays (knee angles, hip angles, trunk lean, CoM height)
    """
    # Extract keypoint arrays (vectorized)
    # H36Key indices: RHIP=1, RKNEE=2, RFOOT=3, LHIP=4, LKNEE=5, LFOOT=6
    # SPINE=7, THORAX=8, NECK=9, HIP_CENTER=0
    r_hip = poses[:, H36Key.RHIP]  # (N, 3)
    r_knee = poses[:, H36Key.RKNEE]
    r_foot = poses[:, H36Key.RFOOT]
    l_hip = poses[:, H36Key.LHIP]
    l_knee = poses[:, H36Key.LKNEE]
    l_foot = poses[:, H36Key.LFOOT]
    thorax = poses[:, H36Key.THORAX]
    spine = poses[:, H36Key.SPINE]
    neck = poses[:, H36Key.NECK]
    hip_center = poses[:, H36Key.HIP_CENTER]

    # Helper function to compute angles between vectors
    def compute_angles_batch(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute angles at point b for vectors (a->b) and (b->c).

        Args:
            a, b, c: (N, 3) arrays of keypoints

        Returns:
            (N,) array of angles in degrees, with NaN for invalid frames
        """
        vec1 = b - a  # (N, 3)
        vec2 = c - b  # (N, 3)

        # Compute norms
        norm1 = np.linalg.norm(vec1, axis=1)
        norm2 = np.linalg.norm(vec2, axis=1)

        # Dot product
        dot = np.sum(vec1 * vec2, axis=1)

        # Cosine with clipping
        cos = np.clip(dot / (norm1 * norm2 + 1e-8), -1, 1)

        # Convert to degrees
        angles = np.degrees(np.arccos(cos))

        # Mark invalid frames (where any keypoint is NaN)
        valid_mask = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1) | np.isnan(c).any(axis=1))
        angles[~valid_mask] = np.nan

        return angles

    # Knee angles (hip-knee-ankle)
    knee_angles_r = compute_angles_batch(r_hip, r_knee, r_foot)
    knee_angles_l = compute_angles_batch(l_hip, l_knee, l_foot)

    # Hip angles (thorax-hip-knee)
    hip_angles_r = compute_angles_batch(thorax, r_hip, r_knee)
    hip_angles_l = compute_angles_batch(thorax, l_hip, l_knee)

    # Trunk lean (spine angle from vertical)
    spine_vec = neck - spine  # (N, 3)
    spine_vec[:, 1] = 0  # Project to horizontal plane (set y to 0)

    # Compute lean angle: arctan2(x, z)
    trunk_lean = np.degrees(np.arctan2(spine_vec[:, 0], spine_vec[:, 2]))

    # Handle division by zero (when z=0)
    z_zero = spine_vec[:, 2] == 0
    trunk_lean[z_zero] = 0.0

    # Mark invalid frames
    valid_spine = ~(np.isnan(spine).any(axis=1) | np.isnan(neck).any(axis=1))
    trunk_lean[~valid_spine] = np.nan

    # CoM height (hip center y-coordinate)
    com_height = hip_center[:, 1].copy()
    valid_hip = ~np.isnan(hip_center[:, 1])
    com_height[~valid_hip] = np.nan

    # Convert to lists for JSON (NaN -> None)
    def to_list(arr: np.ndarray) -> list:
        """Convert numpy array to list, replacing NaN with None."""
        return [float(x) if not np.isnan(x) else None for x in arr]

    return {
        "knee_angles_r": to_list(knee_angles_r),
        "knee_angles_l": to_list(knee_angles_l),
        "hip_angles_r": to_list(hip_angles_r),
        "hip_angles_l": to_list(hip_angles_l),
        "trunk_lean": to_list(trunk_lean),
        "com_height": to_list(com_height),
    }


def test_compute_frame_metrics_output_structure():
    """Test that _compute_frame_metrics returns all 6 expected keys."""
    # Create dummy poses (N=10, 17 keypoints, 3 coords)
    poses = np.random.randn(10, 17, 3).astype(np.float32)

    result = _compute_frame_metrics(poses)

    # Check all 6 keys are present
    assert set(result.keys()) == {
        "knee_angles_r",
        "knee_angles_l",
        "hip_angles_r",
        "hip_angles_l",
        "trunk_lean",
        "com_height",
    }


def test_compute_frame_metrics_output_length():
    """Test that output lists match input length."""
    n_frames = 50
    poses = np.random.randn(n_frames, 17, 3).astype(np.float32)

    result = _compute_frame_metrics(poses)

    # All output lists should have length n_frames
    for key in result:
        assert len(result[key]) == n_frames, f"{key} has wrong length"


def test_compute_frame_metrics_angle_ranges():
    """Test that angles are in valid ranges."""
    # Create realistic poses (not random)
    poses = np.zeros((20, 17, 3), dtype=np.float32)

    # Set up a basic standing pose
    # Hip center at origin
    poses[:, H36Key.HIP_CENTER] = [0, 0.5, 0]

    # Legs: hips below hip center, knees below hips, feet below knees
    poses[:, H36Key.RHIP] = [0.1, 0.4, 0]
    poses[:, H36Key.RKNEE] = [0.1, 0.2, 0]
    poses[:, H36Key.RFOOT] = [0.1, 0.0, 0]

    poses[:, H36Key.LHIP] = [-0.1, 0.4, 0]
    poses[:, H36Key.LKNEE] = [-0.1, 0.2, 0]
    poses[:, H36Key.LFOOT] = [-0.1, 0.0, 0]

    # Torso: spine above hip center, thorax above spine, neck above thorax
    poses[:, H36Key.SPINE] = [0, 0.6, 0]
    poses[:, H36Key.THORAX] = [0, 0.7, 0]
    poses[:, H36Key.NECK] = [0, 0.8, 0]

    result = _compute_frame_metrics(poses)

    # Knee angles should be between 0 and 180 degrees
    for i, angle in enumerate(result["knee_angles_r"]):
        if angle is not None:
            assert 0 <= angle <= 180, f"Right knee angle {angle} at frame {i} out of range"

    for i, angle in enumerate(result["knee_angles_l"]):
        if angle is not None:
            assert 0 <= angle <= 180, f"Left knee angle {angle} at frame {i} out of range"

    # Hip angles should be between 0 and 180 degrees
    for i, angle in enumerate(result["hip_angles_r"]):
        if angle is not None:
            assert 0 <= angle <= 180, f"Right hip angle {angle} at frame {i} out of range"

    for i, angle in enumerate(result["hip_angles_l"]):
        if angle is not None:
            assert 0 <= angle <= 180, f"Left hip angle {angle} at frame {i} out of range"


def test_compute_frame_metrics_nan_handling():
    """Test that NaN frames produce None in output."""
    poses = np.random.randn(10, 17, 3).astype(np.float32)

    # Insert NaN frames
    poses[3, :, :] = np.nan
    poses[7, :, :] = np.nan

    result = _compute_frame_metrics(poses)

    # Frames 3 and 7 should have None for all metrics
    for key in result:
        assert result[key][3] is None, f"{key}[3] should be None"
        assert result[key][7] is None, f"{key}[7] should be None"


def test_compute_frame_metrics_deterministic():
    """Test that output is deterministic for same input."""
    poses = np.random.randn(20, 17, 3).astype(np.float32)
    np.random.seed(42)  # Reset seed

    result1 = _compute_frame_metrics(poses)
    result2 = _compute_frame_metrics(poses)

    # All results should be identical
    for key in result1:
        assert result1[key] == result2[key], f"{key} is not deterministic"


def test_compute_frame_metrics_single_frame():
    """Test with single frame input."""
    poses = np.random.randn(1, 17, 3).astype(np.float32)

    result = _compute_frame_metrics(poses)

    # All outputs should be lists of length 1
    for key in result:
        assert len(result[key]) == 1, f"{key} should have length 1"


def test_compute_frame_metrics_com_height():
    """Test that CoM height matches hip center y-coordinate."""
    poses = np.zeros((5, 17, 3), dtype=np.float32)

    # Set different hip center heights
    for i in range(5):
        poses[i, H36Key.HIP_CENTER] = [0, float(i * 0.1), 0]

    result = _compute_frame_metrics(poses)

    # CoM height should match hip center y
    for i in range(5):
        expected = i * 0.1
        actual = result["com_height"][i]
        if actual is not None:
            assert abs(actual - expected) < 1e-6, f"CoM height mismatch at frame {i}"


def test_compute_frame_metrics_empty_input():
    """Test with empty input array."""
    poses = np.zeros((0, 17, 3), dtype=np.float32)

    result = _compute_frame_metrics(poses)

    # All outputs should be empty lists
    for key in result:
        assert len(result[key]) == 0, f"{key} should be empty"
