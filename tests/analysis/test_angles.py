"""Tests for comprehensive biomechanics angle computation."""

import numpy as np

from src.analysis.angles import (
    ANGLE_DEFS,
    SEGMENT_DEFS,
    compute_joint_angles,
    compute_segment_angles,
)


def _standing_pose():
    """Create a standing pose with known angles."""
    pose = np.zeros((17, 2), dtype=np.float32)
    pose[0] = [320, 80]  # HIP_CENTER
    pose[1] = [310, 200]  # RHIP
    pose[2] = [310, 280]  # RKNEE
    pose[3] = [310, 360]  # RFOOT
    pose[4] = [330, 200]  # LHIP
    pose[5] = [330, 280]  # LKNEE
    pose[6] = [330, 360]  # LFOOT
    pose[7] = [320, 160]  # SPINE
    pose[8] = [320, 120]  # THORAX
    pose[9] = [320, 100]  # NECK
    pose[10] = [320, 80]  # HEAD
    pose[11] = [300, 120]  # LSHOULDER
    pose[12] = [280, 170]  # LELBOW
    pose[13] = [260, 220]  # LWRIST
    pose[14] = [340, 120]  # RSHOULDER
    pose[15] = [360, 170]  # RELBOW
    pose[16] = [380, 220]  # RWRIST
    return pose


class TestAngleDefs:
    def test_has_expected_joint_angles(self):
        """Should define at least 12 joint angles."""
        joint_names = [d["name"] for d in ANGLE_DEFS]
        assert len(joint_names) >= 12
        assert "R Knee" in joint_names
        assert "L Hip" in joint_names

    def test_has_expected_segment_angles(self):
        """Should define at least 14 segment angles."""
        seg_names = [d["name"] for d in SEGMENT_DEFS]
        assert len(seg_names) >= 14
        assert "R Foot" in seg_names
        assert "Trunk" in seg_names


class TestComputeJointAngles:
    def test_knee_angle_standing(self):
        """Standing pose should give ~180° knee angle."""
        pose = _standing_pose()
        angles = compute_joint_angles(pose)
        assert 170 < angles["R Knee"] <= 180

    def test_all_angles_in_range(self):
        """All angles should be in [0, 180] degrees."""
        pose = _standing_pose()
        angles = compute_joint_angles(pose)
        for name, val in angles.items():
            if not np.isnan(val):
                assert 0 <= val <= 180, f"{name} = {val} out of range"

    def test_bent_knee(self):
        """Deeply bent knee should give small angle."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[4] = [330, 200]  # LHIP
        pose[5] = [330, 250]  # LKNEE
        pose[6] = [270, 350]  # LFOOT (foot behind knee)
        angles = compute_joint_angles(pose)
        assert angles["L Knee"] < 170


class TestComputeSegmentAngles:
    def test_trunk_vertical(self):
        """Vertical trunk should give ~90° angle."""
        pose = _standing_pose()
        angles = compute_segment_angles(pose)
        assert 80 < angles["Trunk"] < 100

    def test_all_segment_angles_defined(self):
        """All defined segments should have computed angles."""
        pose = _standing_pose()
        angles = compute_segment_angles(pose)
        for d in SEGMENT_DEFS:
            assert d["name"] in angles
