"""Tests for foot keypoint 3D→2D projection."""
import json
import numpy as np
import pytest

DATA_ROOT = "data/datasets/athletepose3d"


@pytest.fixture
def cam_params():
    with open(f"{DATA_ROOT}/cam_param.json") as f:
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
            assert not np.any(np.isnan(pts)), f"NaN in {cam_name}"
            # At least some foot points should be in-frame
            valid = (
                (pts[:, 0] >= 0)
                & (pts[:, 0] <= 1920)
                & (pts[:, 1] >= 0)
                & (pts[:, 1] <= 1088)
            )
            assert valid.any(), f"No valid foot points for {cam_name}"

    def test_nan_3d_returns_nan(self, cam_params):
        """NaN input should return NaN output."""
        from src.datasets.projector import project_point

        cam = cam_params["fs_camera_1"]
        x, y = project_point(np.array([np.nan, 0.0, 0.0]), cam)
        assert np.isnan(x)
        assert np.isnan(y)


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

        foot_2d = np.array([
            [500.0, 820.0],   # L heel: 20px below ankle
            [530.0, 870.0],   # L big toe
            [490.0, 860.0],   # L small toe
            [700.0, 820.0],   # R heel: 20px below ankle
            [730.0, 870.0],   # R big toe
            [690.0, 860.0],   # R small toe
        ], dtype=np.float32)

        validate_foot_projection(foot_2d, coco_2d)

        assert not np.isnan(foot_2d[0]).any(), "Valid L heel should be kept"
        assert not np.isnan(foot_2d[3]).any(), "Valid R heel should be kept"

    def test_invalid_heel_above_ankle(self):
        """Heel above ankle by more than 30px is rejected."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array([
            [500.0, 750.0],   # L heel: 50px above ankle (> 30px tolerance)
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
            [700.0, 820.0],   # R heel: valid
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
        ], dtype=np.float32)

        validate_foot_projection(foot_2d, coco_2d)

        assert np.isnan(foot_2d[0]).all(), "L heel 50px above ankle should be NaN"
        assert not np.isnan(foot_2d[3]).any(), "R heel within range should be kept"

    def test_invalid_heel_too_far(self):
        """Heel more than 60px from ankle is rejected."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array([
            [500.0, 900.0],   # L heel: 100px below ankle (> 60px max)
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
        ], dtype=np.float32)

        validate_foot_projection(foot_2d, coco_2d)

        assert np.isnan(foot_2d[0]).all(), "L heel 100px from ankle should be NaN"

    def test_valid_toe_kept(self):
        """Toe within 80px of ankle is kept."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array([
            [0.0, 0.0],       # placeholder (heel)
            [530.0, 870.0],   # L big toe: ~74px from ankle
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
            [730.0, 870.0],   # R big toe: ~74px from ankle
            [0.0, 0.0],       # placeholder
        ], dtype=np.float32)

        validate_foot_projection(foot_2d, coco_2d)

        assert not np.isnan(foot_2d[1]).any(), "L big toe within 80px should be kept"
        assert not np.isnan(foot_2d[4]).any(), "R big toe within 80px should be kept"

    def test_invalid_toe_too_far(self):
        """Toe more than 80px from ankle is rejected."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array([
            [0.0, 0.0],       # placeholder
            [540.0, 880.0],   # L big toe: ~89px from ankle (> 80px max)
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
            [0.0, 0.0],       # placeholder
        ], dtype=np.float32)

        validate_foot_projection(foot_2d, coco_2d)

        assert np.isnan(foot_2d[1]).all(), "L big toe 89px from ankle should be NaN"

    def test_mixed_valid_and_invalid(self):
        """Only invalid points become NaN, valid ones are preserved."""
        from src.datasets.projector import validate_foot_projection

        coco_2d = np.zeros((17, 2))
        coco_2d[15] = [500.0, 800.0]  # L ankle
        coco_2d[16] = [700.0, 800.0]  # R ankle

        foot_2d = np.array([
            [500.0, 820.0],   # L heel: 20px, valid
            [540.0, 880.0],   # L big toe: 89px, INVALID
            [490.0, 850.0],   # L small toe: ~51px, valid
            [700.0, 750.0],   # R heel: 50px above ankle, INVALID
            [730.0, 870.0],   # R big toe: 74px, valid
            [700.0, 900.0],   # R small toe: 100px, INVALID
        ], dtype=np.float32)

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

        foot_2d = np.array([
            [np.nan, np.nan],
            [530.0, 870.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [730.0, 870.0],
            [np.nan, np.nan],
        ], dtype=np.float32)

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
        coco_2d[16] = [700.0, 800.0]    # R ankle is valid

        foot_2d = np.array([
            [500.0, 820.0],   # L heel: would be valid but ankle is NaN
            [530.0, 870.0],   # L big toe: same
            [490.0, 860.0],   # L small toe: same
            [700.0, 820.0],   # R heel: valid
            [730.0, 870.0],   # R big toe: valid
            [690.0, 860.0],   # R small toe: valid
        ], dtype=np.float32)

        validate_foot_projection(foot_2d, coco_2d)

        # All left foot points should be NaN (ankle missing)
        for i in range(3):
            assert np.isnan(foot_2d[i]).all(), f"Left foot index {i} should be NaN"
        # Right foot should be kept
        for i in range(3, 6):
            assert not np.isnan(foot_2d[i]).any(), f"Right foot index {i} should be kept"
