"""Tests for foot keypoint 3D→2D projection."""

import json
from pathlib import Path

import numpy as np
import pytest

DATA_ROOT = "data/datasets/athletepose3d"


@pytest.fixture
def cam_params():
    with Path(f"{DATA_ROOT}/cam_param.json").open() as f:
        return json.load(f)


@pytest.fixture
def sample_3d():
    return np.load(f"{DATA_ROOT}/videos/train_set/S1/Axel_10_cam_1.npy")


@pytest.fixture
def sample_coco():
    return np.load(f"{DATA_ROOT}/videos/train_set/S1/Axel_10_cam_1_coco.npy")


class TestFootProjection:
    def test_project_foot_point_in_frame(self, cam_params, sample_3d):
        """Projected LHEL should be within video bounds (1920x1088)."""
        from src.datasets.projector import project_point

        cam = cam_params["fs_camera_1"]
        p_world = sample_3d[50, 49]  # LHEL at frame 50 (visible from cam_1)
        x, y = project_point(p_world, cam)

        assert 0 <= x <= 1920, f"x={x} out of bounds"
        assert 0 <= y <= 1088, f"y={y} out of bounds"

    def test_foot_points_near_ankle(self, cam_params, sample_3d, sample_coco):
        """LHEL should be within 50px of COCO LANK."""
        from src.datasets.projector import project_point

        cam = cam_params["fs_camera_1"]
        # Use frame where keypoints are visible (frame 50)
        lank_x, lank_y = sample_coco[50, 15]
        lhel_x, lhel_y = project_point(sample_3d[50, 49], cam)

        dist = np.sqrt((lhel_x - lank_x) ** 2 + (lhel_y - lank_y) ** 2)
        assert dist < 50, f"LHEL to LANK: {dist:.0f}px > 50px"

    def test_projection_matches_coco_body(self, cam_params, sample_3d, sample_coco):
        """Our projection should match _coco.npy within 20px for LANK."""
        from src.datasets.projector import project_point

        cam = cam_params["fs_camera_1"]
        our_x, our_y = project_point(sample_3d[50, 33], cam)  # LANK = AP3D idx 33
        coco_x, coco_y = sample_coco[50, 15]  # COCO kp15 = left_ankle

        dist = np.sqrt((our_x - coco_x) ** 2 + (our_y - coco_y) ** 2)
        assert dist < 20, f"LANK projection dist={dist:.0f}px > 20px"

    def test_project_foot_frame_returns_6x2(self, cam_params, sample_3d):
        """project_foot_frame returns (6, 2) for foot keypoints."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        result = project_foot_frame(sample_3d[50], cam)

        assert result.shape == (6, 2), f"Expected (6, 2), got {result.shape}"

    def test_both_cameras_produce_valid_coords(self, cam_params, sample_3d):
        """Camera 1 and Camera 6 both produce in-frame foot coords."""
        from src.datasets.projector import project_foot_frame

        for cam_name in ["fs_camera_1", "fs_camera_6"]:
            cam = cam_params[cam_name]
            pts = project_foot_frame(sample_3d[50], cam)
            # Small toe slots (indices 2, 5) are always NaN
            projected_indices = [0, 1, 3, 4]
            non_nan = [i for i in projected_indices if not np.isnan(pts[i]).any()]
            assert len(non_nan) > 0, f"All projected foot points are NaN in {cam_name}"
            # At least some foot points should be in-frame
            valid = (pts[:, 0] >= 0) & (pts[:, 0] <= 1920) & (pts[:, 1] >= 0) & (pts[:, 1] <= 1088)
            assert valid.any(), f"No valid foot points for {cam_name}"

    def test_nan_3d_returns_nan(self, cam_params):
        """NaN input should return NaN output."""
        from src.datasets.projector import project_point

        cam = cam_params["fs_camera_1"]
        x, y = project_point(np.array([np.nan, 0.0, 0.0]), cam)
        assert np.isnan(x)
        assert np.isnan(y)


class TestWeakPerspectiveProjection:
    """Tests for localized weak-perspective foot projection."""

    def test_ankle_ap3d_indices_defined(self):
        """ANKLE_AP3D_INDICES constant should exist with L_ankle=33, R_ankle=95."""
        from src.datasets.projector import ANKLE_AP3D_INDICES

        assert list(ANKLE_AP3D_INDICES) == [33, 95]

    def test_small_toes_always_nan(self, cam_params, sample_3d):
        """Small toe slots (indices 2, 5) should always be NaN."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        foot_2d = project_foot_frame(sample_3d[50], cam)

        assert np.isnan(foot_2d[2]).all(), "L_small_toe should always be NaN"
        assert np.isnan(foot_2d[5]).all(), "R_small_toe should always be NaN"

    def test_weak_perspective_preserves_foot_geometry(self, cam_params, sample_3d):
        """Foot points should maintain correct relative positions (heel below ankle)."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        foot_2d = project_foot_frame(sample_3d[50], cam)

        coco_kps = np.load(f"{DATA_ROOT}/videos/train_set/S1/Axel_10_cam_1_coco.npy")
        l_ankle_y = coco_kps[50, 15, 1]  # left_ankle Y
        r_ankle_y = coco_kps[50, 16, 1]  # right_ankle Y

        # Left heel (index 0) should not be above left ankle
        if not np.isnan(foot_2d[0, 1]):
            assert foot_2d[0, 1] >= l_ankle_y - 5, (
                f"L_heel y={foot_2d[0, 1]:.0f} above L_ankle y={l_ankle_y:.0f}"
            )

        # Right heel (index 3) should not be above right ankle
        if not np.isnan(foot_2d[3, 1]):
            assert foot_2d[3, 1] >= r_ankle_y - 5, (
                f"R_heel y={foot_2d[3, 1]:.0f} above R_ankle y={r_ankle_y:.0f}"
            )

    def test_weak_perspective_foot_near_ankle(self, cam_params, sample_3d, sample_coco):
        """Weak-perspective foot points should be within reasonable distance of ankle."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        foot_2d = project_foot_frame(sample_3d[50], cam)

        l_ankle = sample_coco[50, 15]  # L_ankle from _coco.npy

        # Check only projected points (indices 0, 1 — skip index 2 which is always NaN)
        for i in [0, 1]:
            if not np.isnan(foot_2d[i, 0]):
                dist = np.linalg.norm(foot_2d[i] - l_ankle)
                assert dist < 100, f"Left foot index {i}: {dist:.0f}px from ankle"

    def test_nan_ankle_invalidates_heel(self, cam_params):
        """NaN processed ankle (AP3D 33) should invalidate L_heel (uses weak-perspective)."""
        from src.datasets.projector import project_foot_frame

        cam = cam_params["fs_camera_1"]
        kp3d = np.zeros((142, 3), dtype=np.float64)

        # Set valid positions for everything EXCEPT left processed ankle (index 33)
        kp3d[:] = [1000.0, 500.0, 2000.0]
        kp3d[33] = [np.nan, np.nan, np.nan]  # L_ankle (processed) is NaN

        # L_big_toe uses raw marker (26) — should still project independently
        kp3d[26] = [1100.0, 300.0, 2000.0]  # L_Toe (raw)

        # Right side is fully valid
        kp3d[95] = [2000.0, 500.0, 2000.0]  # R_ankle (processed)
        kp3d[112] = [2000.0, 300.0, 2000.0]  # RHEL (processed)
        kp3d[93] = [2100.0, 300.0, 2000.0]  # R_Toe (raw)

        foot_2d = project_foot_frame(kp3d, cam)

        # L_heel (index 0) uses weak-perspective from L_ankle → NaN
        assert np.isnan(foot_2d[0]).all(), "L_heel should be NaN when ankle is NaN"

        # L_big_toe (index 1) uses full-perspective from raw marker → may be valid
        # (depends on whether the synthetic 3D position projects in-frame)

        # Small toes always NaN
        assert np.isnan(foot_2d[2]).all(), "L_small_toe should always be NaN"
        assert np.isnan(foot_2d[5]).all(), "R_small_toe should always be NaN"

        # Right heel and big toe should be valid (right ankle is not NaN)
        assert not np.isnan(foot_2d[3]).any(), "R_heel should be valid"
        assert not np.isnan(foot_2d[4]).any(), "R_bigtoe should be valid"


class TestValidateFootProjection:
    """Tests for validate_foot_projection in-place validation."""

    def test_valid_heel_kept(self):
        """Heel within 60px and below ankle is kept."""
        from src.datasets.projector import validate_foot_projection

        l_ankle = np.array([500.0, 800.0])
        r_ankle = np.array([700.0, 800.0])
        coco_2d = np.zeros((17, 2))
        coco_2d[15] = l_ankle
        coco_2d[16] = r_ankle

        foot_2d = np.array(
            [
                [500.0, 820.0],  # L heel: 20px below ankle
                [530.0, 870.0],  # L big toe
                [490.0, 860.0],  # L small toe
                [700.0, 820.0],  # R heel: 20px below ankle
                [730.0, 870.0],  # R big toe
                [690.0, 860.0],  # R small toe
            ],
            dtype=np.float32,
        )

        validate_foot_projection(foot_2d, coco_2d)

        assert not np.isnan(foot_2d[0]).any(), "Valid L heel should be kept"
        assert not np.isnan(foot_2d[3]).any(), "Valid R heel should be kept"

    def test_invalid_heel_above_ankle(self):
        """Heel above ankle by more than 30px is rejected."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array(
            [
                [500.0, 750.0],  # L heel: 50px above ankle (> 30px tolerance)
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
                [700.0, 820.0],  # R heel: valid
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
            ],
            dtype=np.float32,
        )

        validate_foot_projection(foot_2d, coco_2d)

        assert np.isnan(foot_2d[0]).all(), "L heel 50px above ankle should be NaN"
        assert not np.isnan(foot_2d[3]).any(), "R heel within range should be kept"

    def test_invalid_heel_too_far(self):
        """Heel more than 60px from ankle is rejected."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array(
            [
                [500.0, 900.0],  # L heel: 100px below ankle (> 60px max)
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
            ],
            dtype=np.float32,
        )

        validate_foot_projection(foot_2d, coco_2d)

        assert np.isnan(foot_2d[0]).all(), "L heel 100px from ankle should be NaN"

    def test_valid_toe_kept(self):
        """Toe within 80px of ankle is kept."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array(
            [
                [0.0, 0.0],  # placeholder (heel)
                [530.0, 870.0],  # L big toe: ~74px from ankle
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
                [730.0, 870.0],  # R big toe: ~74px from ankle
                [0.0, 0.0],  # placeholder
            ],
            dtype=np.float32,
        )

        validate_foot_projection(foot_2d, coco_2d)

        assert not np.isnan(foot_2d[1]).any(), "L big toe within 80px should be kept"
        assert not np.isnan(foot_2d[4]).any(), "R big toe within 80px should be kept"

    def test_invalid_toe_too_far(self):
        """Toe more than 80px from ankle is rejected."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array(
            [
                [0.0, 0.0],  # placeholder
                [540.0, 880.0],  # L big toe: ~89px from ankle (> 80px max)
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
            ],
            dtype=np.float32,
        )

        validate_foot_projection(foot_2d, coco_2d)

        assert np.isnan(foot_2d[1]).all(), "L big toe 89px from ankle should be NaN"

    def test_mixed_valid_and_invalid(self):
        """Only invalid points become NaN, valid ones are preserved."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array(
            [
                [500.0, 820.0],  # L heel: 20px, valid
                [540.0, 880.0],  # L big toe: 89px, INVALID
                [490.0, 850.0],  # L small toe: ~51px, valid
                [700.0, 750.0],  # R heel: 50px above ankle, INVALID
                [730.0, 870.0],  # R big toe: 74px, valid
                [700.0, 900.0],  # R small toe: 100px, INVALID
            ],
            dtype=np.float32,
        )

        validate_foot_projection(foot_2d, coco_2d)

        # Valid
        assert not np.isnan(foot_2d[0]).any(), "L heel should be kept"
        assert not np.isnan(foot_2d[2]).any(), "L small toe should be kept"
        assert not np.isnan(foot_2d[4]).any(), "R big toe should be kept"
        # Invalid
        assert np.isnan(foot_2d[1]).all(), "L big toe too far should be NaN"
        assert np.isnan(foot_2d[3]).all(), "R heel above ankle should be NaN"
        assert np.isnan(foot_2d[5]).all(), "R small toe too far should be NaN"

    def test_already_nan_preserved(self):
        """Already-NaN points remain NaN (no crash)."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]
        coco_2d[16] = [700.0, 800.0]

        foot_2d = np.array(
            [
                [np.nan, np.nan],
                [530.0, 870.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [730.0, 870.0],
                [np.nan, np.nan],
            ],
            dtype=np.float32,
        )

        validate_foot_projection(foot_2d, coco_2d)

        assert np.isnan(foot_2d[0]).all()
        assert np.isnan(foot_2d[2]).all()
        assert np.isnan(foot_2d[3]).all()
        assert np.isnan(foot_2d[5]).all()
        assert not np.isnan(foot_2d[1]).any()
        assert not np.isnan(foot_2d[4]).any()

    def test_nan_ankle_invalidates_foot(self):
        """NaN ankle should invalidate its foot keypoints."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [np.nan, np.nan]  # L ankle is NaN
        coco_2d[16] = [700.0, 800.0]  # R ankle is valid

        foot_2d = np.array(
            [
                [500.0, 820.0],  # L heel: would be valid but ankle is NaN
                [530.0, 870.0],  # L big toe: same
                [490.0, 860.0],  # L small toe: same
                [700.0, 820.0],  # R heel: valid
                [730.0, 870.0],  # R big toe: valid
                [690.0, 860.0],  # R small toe: valid
            ],
            dtype=np.float32,
        )

        validate_foot_projection(foot_2d, coco_2d)

        # All left foot points should be NaN (ankle missing)
        for i in range(3):
            assert np.isnan(foot_2d[i]).all(), f"Left foot index {i} should be NaN"
        # Right foot should be kept
        for i in range(3, 6):
            assert not np.isnan(foot_2d[i]).any(), f"Right foot index {i} should be kept"

    def test_invalid_toe_above_ankle(self):
        """Toe above ankle is rejected even if distance is small."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle

        foot_2d = np.array(
            [
                [0.0, 0.0],  # placeholder
                [495.0, 760.0],  # L big toe: 40px above ankle (800-30=770 threshold)
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
                [0.0, 0.0],  # placeholder
            ],
            dtype=np.float32,
        )

        validate_foot_projection(foot_2d, coco_2d)

        assert np.isnan(foot_2d[1]).all(), "Toe above ankle should be NaN"
