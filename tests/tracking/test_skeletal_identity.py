"""Tests for 3D skeletal identity extraction."""

import numpy as np

from src.tracking.skeletal_identity import (
    NUM_BONES,
    compute_2d_skeletal_ratios,
    compute_bone_lengths_3d,
    compute_identity_profile,
    identity_similarity,
)
from src.types import H36Key


def _make_3d_pose(bone_scale: float = 1.0) -> np.ndarray:
    """Create synthetic 3D H3.6M pose with known bone lengths."""
    pose = np.zeros((17, 3), dtype=np.float32)
    s = bone_scale
    pose[H36Key.HIP_CENTER] = [0, 0, 0]
    pose[H36Key.RHIP] = [-0.05, 0, 0]
    pose[H36Key.LHIP] = [0.05, 0, 0]
    pose[H36Key.RKNEE] = [-0.05, -0.20 * s, 0]
    pose[H36Key.LKNEE] = [0.05, -0.20 * s, 0]
    pose[H36Key.RFOOT] = [-0.05, -0.40 * s, 0]
    pose[H36Key.LFOOT] = [0.05, -0.40 * s, 0]
    pose[H36Key.SPINE] = [0, -0.15 * s, 0]
    pose[H36Key.THORAX] = [0, -0.25 * s, 0]
    pose[H36Key.NECK] = [0, -0.30 * s, 0]
    pose[H36Key.HEAD] = [0, -0.35 * s, 0]
    pose[H36Key.LSHOULDER] = [0.08, -0.25 * s, 0]
    pose[H36Key.RSHOULDER] = [-0.08, -0.25 * s, 0]
    pose[H36Key.LELBOW] = [0.12, -0.15 * s, 0]
    pose[H36Key.RELBOW] = [-0.12, -0.15 * s, 0]
    pose[H36Key.LWRIST] = [0.14, -0.05 * s, 0]
    pose[H36Key.RWRIST] = [-0.14, -0.05 * s, 0]
    return pose


class TestComputeBoneLengths3D:
    def test_returns_correct_shape(self):
        poses = np.array([_make_3d_pose()] * 10)
        bones = compute_bone_lengths_3d(poses)
        assert bones.shape == (10, NUM_BONES)

    def test_femur_length(self):
        poses = np.array([_make_3d_pose(bone_scale=1.0)])
        bones = compute_bone_lengths_3d(poses)
        assert abs(bones[0, 0] - 0.2) < 1e-4

    def test_different_scale(self):
        poses_a = np.array([_make_3d_pose(1.0)] * 5)
        poses_b = np.array([_make_3d_pose(1.3)] * 5)
        ba = compute_bone_lengths_3d(poses_a)
        bb = compute_bone_lengths_3d(poses_b)
        assert not np.allclose(ba[0], bb[0])


class TestIdentityProfile:
    def test_shape(self):
        poses = np.array([_make_3d_pose()] * 20)
        bones = compute_bone_lengths_3d(poses)
        profile = compute_identity_profile(bones)
        assert profile.shape == (NUM_BONES,)

    def test_deterministic(self):
        poses = np.array([_make_3d_pose()] * 20)
        bones = compute_bone_lengths_3d(poses)
        p1 = compute_identity_profile(bones)
        p2 = compute_identity_profile(bones)
        assert np.allclose(p1, p2)


class TestIdentitySimilarity:
    def test_identical(self):
        poses = np.array([_make_3d_pose()] * 20)
        bones = compute_bone_lengths_3d(poses)
        profile = compute_identity_profile(bones)
        assert abs(identity_similarity(profile, profile) - 1.0) < 1e-5

    def test_same_proportions_different_scale(self):
        poses_a = np.array([_make_3d_pose(1.0)] * 20)
        poses_b = np.array([_make_3d_pose(1.3)] * 20)
        ba = compute_bone_lengths_3d(poses_a)
        bb = compute_bone_lengths_3d(poses_b)
        pa = compute_identity_profile(ba)
        pb = compute_identity_profile(bb)
        assert identity_similarity(pa, pb) > 0.99

    def test_different_proportions(self):
        poses_a = np.array([_make_3d_pose(1.0)] * 20)
        poses_b = np.array([_make_3d_pose(1.0)] * 20)
        poses_b[:, H36Key.RKNEE, 1] *= 2.0
        poses_b[:, H36Key.LKNEE, 1] *= 2.0
        poses_b[:, H36Key.RFOOT, 1] *= 2.0
        poses_b[:, H36Key.LFOOT, 1] *= 2.0
        ba = compute_bone_lengths_3d(poses_a)
        bb = compute_bone_lengths_3d(poses_b)
        pa = compute_identity_profile(ba)
        pb = compute_identity_profile(bb)
        assert identity_similarity(pa, pb) < 0.96


class TestNaNHandling:
    def test_nan_keypoints(self):
        poses = np.array([_make_3d_pose()] * 10)
        poses[3, H36Key.RFOOT, :] = np.nan
        bones = compute_bone_lengths_3d(poses)
        assert np.isnan(bones[3, 1])

    def test_profile_ignores_nan_frames(self):
        poses = np.array([_make_3d_pose()] * 20)
        poses[5, :, :] = np.nan
        bones = compute_bone_lengths_3d(poses)
        profile = compute_identity_profile(bones)
        assert not np.any(np.isnan(profile))


class Test2dSkeletalRatios:
    def test_returns_five_ratios(self):
        pose = _make_3d_pose()[:, :2]
        ratios = compute_2d_skeletal_ratios(pose)
        assert ratios.shape == (5,)
