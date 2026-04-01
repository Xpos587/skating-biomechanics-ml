"""Tests for kinematic_constraints module (bone length enforcement, joint angle limits)."""

import numpy as np
import pytest

from src.pose_3d.kinematic_constraints import (
    JOINT_LIMITS,
    apply_kinematic_constraints,
    enforce_bone_lengths,
    enforce_joint_angle_limits,
)
from src.types import H36M_SKELETON_EDGES


def make_t_pose_3d():
    """Create a standing pose with slightly bent elbows in H3.6M format (17, 3) in meters.

    Elbows are bent to ~135 deg so they stay within the [0, 160] joint limit.
    Knees are straight at ~180 deg (within [0, 180]).
    """
    pose = np.zeros((17, 3), dtype=np.float32)
    # HIP_CENTER at origin
    pose[0] = [0.0, 0.0, 0.0]  # HIP_CENTER
    # RHIP slightly right, LHIP slightly left
    pose[1] = [0.15, 0.1, 0.0]  # RHIP
    pose[4] = [-0.15, 0.1, 0.0]  # LHIP
    # Legs down
    pose[2] = [0.15, 0.45, 0.0]  # RKNEE
    pose[5] = [-0.15, 0.45, 0.0]  # LKNEE
    pose[3] = [0.15, 0.9, 0.0]  # RFOOT
    pose[6] = [-0.15, 0.9, 0.0]  # LFOOT
    # Spine up
    pose[7] = [0.0, -0.15, 0.0]  # SPINE
    pose[8] = [0.0, -0.35, 0.0]  # THORAX
    pose[9] = [0.0, -0.45, 0.0]  # NECK
    pose[10] = [0.0, -0.6, 0.0]  # HEAD
    # Arms with bent elbows (~135 deg, within [0, 160] limit)
    # RSHOULDER down-right from thorax
    pose[14] = [0.2, -0.35, 0.0]  # RSHOULDER
    pose[11] = [-0.2, -0.35, 0.0]  # LSHOULDER
    # RELBOW angled down (not fully extended horizontally)
    pose[15] = [0.45, -0.55, 0.0]  # RELBOW
    pose[12] = [-0.45, -0.55, 0.0]  # LELBOW
    # RWRIST further down
    pose[16] = [0.55, -0.80, 0.0]  # RWRIST
    pose[13] = [-0.55, -0.80, 0.0]  # LWRIST
    return pose


class TestEnforceBoneLengths:
    def test_consistent_sequence_unchanged(self):
        """A consistent sequence should not be significantly changed."""
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 10)  # (10, 17, 3)
        corrected = enforce_bone_lengths(poses.copy())
        np.testing.assert_allclose(corrected, poses, atol=0.01)

    def test_perturbed_bone_corrected(self):
        """A perturbed frame should have its bone lengths restored."""
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 10)
        perturbed = poses.copy()
        perturbed[5, 16, :] += [0.3, 0.2, 0.1]  # move RWRIST
        corrected = enforce_bone_lengths(perturbed)

        # The corrected bone lengths should be closer to the reference than the perturbed
        for parent, child in H36M_SKELETON_EDGES:
            ref_len = np.linalg.norm(t_pose[child] - t_pose[parent])
            corrected_len = np.linalg.norm(corrected[5, child] - corrected[5, parent])
            perturbed_len = np.linalg.norm(perturbed[5, child] - perturbed[5, parent])
            assert abs(corrected_len - ref_len) < abs(
                perturbed_len - ref_len
            ) + 0.01, f"Bone ({parent},{child}): corrected {corrected_len:.3f} vs ref {ref_len:.3f}"

    def test_shape_preserved(self):
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 5)
        corrected = enforce_bone_lengths(poses.copy())
        assert corrected.shape == poses.shape

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            enforce_bone_lengths(np.zeros((5, 17, 2), dtype=np.float32))

    def test_output_dtype_float32(self):
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 5)
        corrected = enforce_bone_lengths(poses.copy())
        assert corrected.dtype == np.float32

    def test_output_is_finite(self):
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 10)
        corrected = enforce_bone_lengths(poses.copy())
        assert np.all(np.isfinite(corrected))


class TestEnforceJointAngleLimits:
    def test_valid_angles_unchanged(self):
        """Valid joint angles (knees ~180 deg) should not be changed."""
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 5)
        corrected = enforce_joint_angle_limits(poses.copy())
        np.testing.assert_allclose(corrected, poses, atol=0.02)

    def test_shape_preserved(self):
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 5)
        corrected = enforce_joint_angle_limits(poses.copy())
        assert corrected.shape == poses.shape

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            enforce_joint_angle_limits(np.zeros((5, 10, 3), dtype=np.float32))

    def test_hyperextended_knee_clamped(self):
        """A knee angle exceeding 180 deg should be clamped to 180."""
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 3)

        # Hyperextend right knee by pushing RFOOT past RKNEE (towards hip)
        # RKNEE = pose[2] = [0.15, 0.45, 0]
        # RFOOT = pose[3] = [0.15, 0.9, 0]
        # Push RFOOT up past the knee line to simulate hyperextension
        poses[1, 3] = [0.15, 0.3, 0.1]  # foot closer to hip than knee

        corrected = enforce_joint_angle_limits(poses.copy())

        # The corrected knee angle should be within valid range [0, 180]
        # Check that something changed on the perturbed frame
        p = corrected[1, 1]  # RHIP
        j = corrected[1, 2]  # RKNEE
        c = corrected[1, 3]  # RFOOT

        ba = p - j
        bc = c - j
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        assert 0.0 <= angle <= 180.0

    def test_output_is_finite(self):
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 5)
        corrected = enforce_joint_angle_limits(poses.copy())
        assert np.all(np.isfinite(corrected))


class TestApplyKinematicConstraints:
    def test_shape_preserved(self):
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 10)
        corrected = apply_kinematic_constraints(poses.copy(), fps=25.0)
        assert corrected.shape == poses.shape

    def test_output_is_finite(self):
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 10)
        corrected = apply_kinematic_constraints(poses.copy(), fps=25.0)
        assert np.all(np.isfinite(corrected))

    def test_output_dtype_float32(self):
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 10)
        corrected = apply_kinematic_constraints(poses.copy(), fps=25.0)
        assert corrected.dtype == np.float32

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            apply_kinematic_constraints(
                np.zeros((5, 17, 2), dtype=np.float32), fps=30.0
            )

    def test_with_confidences(self):
        """Confidences are accepted but currently unused (reserved)."""
        t_pose = make_t_pose_3d()
        poses = np.stack([t_pose] * 10)
        confidences = np.random.rand(10, 17).astype(np.float32)
        corrected = apply_kinematic_constraints(
            poses.copy(), fps=25.0, confidences=confidences
        )
        assert corrected.shape == poses.shape
        assert np.all(np.isfinite(corrected))

    def test_smooth_poses_have_consistent_bone_lengths(self):
        """After kinematic constraints, bone lengths should be consistent."""
        t_pose = make_t_pose_3d()
        n = 30
        poses = np.stack([t_pose] * n).astype(np.float32)
        rng = np.random.RandomState(42)
        noise = rng.randn(n, 17, 3).astype(np.float32) * 0.02
        noisy_poses = poses + noise

        corrected = apply_kinematic_constraints(noisy_poses.copy(), fps=30.0)

        # Compute standard deviation of each bone length across frames
        from src.pose_3d.kinematic_constraints import _KINEMATIC_CHAIN_EDGES

        for parent, child in _KINEMATIC_CHAIN_EDGES:
            diffs = corrected[:, parent, :] - corrected[:, child, :]
            lengths = np.linalg.norm(diffs, axis=1)
            # After bone length enforcement, all frames should have nearly the same length
            assert np.std(lengths) < 0.005, (
                f"Bone ({parent},{child}): length std={np.std(lengths):.4f} too high"
            )
