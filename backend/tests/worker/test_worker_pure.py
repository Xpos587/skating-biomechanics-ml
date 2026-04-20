"""Tests for worker pure functions (_sample_poses, _compute_frame_metrics)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock aiobotocore before importing app.worker (which imports app.storage)
_mock_aiobotocore = MagicMock()
_mock_aiobotocore_session = MagicMock()
sys.modules["aiobotocore"] = _mock_aiobotocore
sys.modules["aiobotocore.session"] = _mock_aiobotocore_session

from app.worker import _compute_frame_metrics, _sample_poses  # noqa: E402

from src.types import H36Key  # noqa: E402

# ---------------------------------------------------------------------------
# _sample_poses
# ---------------------------------------------------------------------------


class TestSamplePoses:
    def test_sample_poses_basic(self):
        """50 frames, sample_rate=10 -> 5 sampled frames with correct indices."""
        poses = np.random.rand(50, 17, 3).astype(np.float32)
        result = _sample_poses(poses, sample_rate=10)

        assert result["frames"] == [0, 10, 20, 30, 40]
        assert len(result["poses"]) == 5
        assert len(result["poses"][0]) == 17  # 17 keypoints
        assert len(result["poses"][0][0]) == 3  # xyz

    def test_sample_poses_single_frame(self):
        """1 frame -> 1 sampled frame."""
        poses = np.random.rand(1, 17, 3).astype(np.float32)
        result = _sample_poses(poses, sample_rate=10)

        assert result["frames"] == [0]
        assert len(result["poses"]) == 1

    def test_sample_poses_custom_rate(self):
        """sample_rate=5 with 20 frames -> 4 sampled."""
        poses = np.random.rand(20, 17, 3).astype(np.float32)
        result = _sample_poses(poses, sample_rate=5)

        assert result["frames"] == [0, 5, 10, 15]
        assert len(result["poses"]) == 4

    def test_sample_poses_preserves_values(self):
        """Actual pose values at sampled indices must match original."""
        poses = np.arange(17 * 3, dtype=np.float32).reshape(1, 17, 3)
        # All frames identical for simplicity
        poses = np.broadcast_to(poses, (30, 17, 3)).copy()
        # Make each frame unique by adding frame index
        for i in range(30):
            poses[i] += i

        result = _sample_poses(poses, sample_rate=10)

        # Frame 0 should be unchanged (i=0 added, so original values)
        np.testing.assert_array_almost_equal(
            np.array(result["poses"][0]),
            poses[0],
        )
        # Frame 10 should match original frame 10
        np.testing.assert_array_almost_equal(
            np.array(result["poses"][1]),
            poses[10],
        )
        # Frame 20 should match original frame 20
        np.testing.assert_array_almost_equal(
            np.array(result["poses"][2]),
            poses[20],
        )


# ---------------------------------------------------------------------------
# _compute_frame_metrics
# ---------------------------------------------------------------------------


class TestComputeFrameMetrics:
    def test_compute_frame_metrics_returns_all_keys(self):
        """All 6 expected metric keys must be present."""
        poses = np.random.rand(10, 17, 3).astype(np.float32)
        result = _compute_frame_metrics(poses)

        expected_keys = {
            "knee_angles_r",
            "knee_angles_l",
            "hip_angles_r",
            "hip_angles_l",
            "trunk_lean",
            "com_height",
        }
        assert set(result.keys()) == expected_keys

    def test_compute_frame_metrics_knee_angles(self):
        """Construct a pose with known knee angles using vector math.

        compute_angles_batch computes angle at point b between vectors (a->b) and (b->c).
        For a straight leg (collinear), the vectors point in the same direction: angle = 0.
        For a 90-degree bend, cos = 0: angle = 90.
        """
        poses = np.zeros((1, 17, 3), dtype=np.float32)

        # Right leg: 90-degree bend at knee
        #   Hip at (0, 0, 0), Knee at (0, -1, 0), Foot at (1, -1, 0)
        #   vec1 (hip->knee) = (0, -1, 0), vec2 (knee->foot) = (1, 0, 0)
        #   cos = 0 / (1 * 1) = 0 -> angle = 90 degrees
        poses[0, H36Key.RHIP] = [0.0, 0.0, 0.0]
        poses[0, H36Key.RKNEE] = [0.0, -1.0, 0.0]
        poses[0, H36Key.RFOOT] = [1.0, -1.0, 0.0]

        # Left leg: straight line (collinear, same direction)
        #   Hip at (0, 0, 0), Knee at (0, -1, 0), Foot at (0, -2, 0)
        #   vec1 = (0, -1, 0), vec2 = (0, -1, 0)
        #   cos = 1 -> angle = 0 degrees (straight)
        poses[0, H36Key.LHIP] = [0.0, 0.0, 0.0]
        poses[0, H36Key.LKNEE] = [0.0, -1.0, 0.0]
        poses[0, H36Key.LFOOT] = [0.0, -2.0, 0.0]

        result = _compute_frame_metrics(poses)

        # Right knee: 90-degree bend
        assert result["knee_angles_r"][0] == pytest.approx(90.0, abs=1.0)
        # Left knee: straight leg -> 0 degrees (vectors are parallel)
        assert result["knee_angles_l"][0] == pytest.approx(0.0, abs=1.0)

    def test_compute_frame_metrics_nan_handling(self):
        """Poses with NaN in key joints produce None in output."""
        poses = np.random.rand(2, 17, 3).astype(np.float32)
        # Frame 0: set right knee to NaN
        poses[0, H36Key.RKNEE] = np.nan
        # Frame 1: set hip center to NaN
        poses[1, H36Key.HIP_CENTER] = np.nan

        result = _compute_frame_metrics(poses)

        # Right knee angle for frame 0 should be None
        assert result["knee_angles_r"][0] is None
        # com_height for frame 1 should be None
        assert result["com_height"][1] is None
        # Other frames should have valid floats
        assert result["knee_angles_r"][1] is not None
        assert result["com_height"][0] is not None

    def test_compute_frame_metrics_trunk_lean(self):
        """Trunk lean: arctan2(spine_vec_x, spine_vec_z) with non-zero z.

        The code projects spine->neck onto the horizontal plane (zeros y).
        trunk_lean = arctan2(x, z), but when z==0 the code guards to 0.0.
        Use z=1 so arctan2(1, 1) = 45 degrees.
        """
        poses = np.zeros((1, 17, 3), dtype=np.float32)

        poses[0, H36Key.SPINE] = [0.0, 0.0, 0.0]
        poses[0, H36Key.NECK] = [1.0, 0.5, 1.0]  # y is zeroed in projection

        result = _compute_frame_metrics(poses)

        assert result["trunk_lean"][0] == pytest.approx(45.0, abs=1.0)

    def test_compute_frame_metrics_trunk_lean_zero_z(self):
        """When z==0 after projection, trunk_lean is forced to 0.0."""
        poses = np.zeros((1, 17, 3), dtype=np.float32)

        poses[0, H36Key.SPINE] = [0.0, 0.0, 0.0]
        poses[0, H36Key.NECK] = [1.0, 0.5, 0.0]  # z=0 triggers zero guard

        result = _compute_frame_metrics(poses)

        assert result["trunk_lean"][0] == pytest.approx(0.0, abs=1e-5)

    def test_compute_frame_metrics_trunk_lean_forward(self):
        """Forward lean: neck offset in +z direction -> arctan2(0, 1) = 0 degrees."""
        poses = np.zeros((1, 17, 3), dtype=np.float32)

        poses[0, H36Key.SPINE] = [0.0, 0.0, 0.0]
        poses[0, H36Key.NECK] = [0.0, 0.5, 1.0]

        result = _compute_frame_metrics(poses)

        assert result["trunk_lean"][0] == pytest.approx(0.0, abs=1.0)

    def test_compute_frame_metrics_com_height(self):
        """com_height matches hip_center y-coordinate."""
        poses = np.random.rand(3, 17, 3).astype(np.float32)
        # Set specific hip_center y values
        poses[0, H36Key.HIP_CENTER, 1] = 0.75
        poses[1, H36Key.HIP_CENTER, 1] = 0.50
        poses[2, H36Key.HIP_CENTER, 1] = 0.25

        result = _compute_frame_metrics(poses)

        assert result["com_height"][0] == pytest.approx(0.75, abs=1e-5)
        assert result["com_height"][1] == pytest.approx(0.50, abs=1e-5)
        assert result["com_height"][2] == pytest.approx(0.25, abs=1e-5)

    def test_compute_frame_metrics_single_frame(self):
        """Single frame -> all metric lists have length 1."""
        poses = np.random.rand(1, 17, 3).astype(np.float32)
        result = _compute_frame_metrics(poses)

        for key in result:
            assert len(result[key]) == 1
            assert isinstance(result[key][0], float) or result[key][0] is None
