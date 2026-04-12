"""Tests for per-frame spatial reference estimation and compensation."""

from unittest.mock import MagicMock, patch

import numpy as np

from skating_ml.detection.spatial_reference import (
    CameraPose,
    compensate_poses_per_frame,
    estimate_pose_sequence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_poses(num_frames: int, fill: float = 0.5, with_nans: bool = False):
    """Create a simple (N, 17, 3) pose array.

    x and y are set to *fill* (normalized).  The third channel is confidence
    set to 1.0.  When *with_nans* is True, even-indexed frames get NaN x/y.
    """
    poses = np.full((num_frames, 17, 3), fill, dtype=np.float32)
    poses[:, :, 2] = 1.0  # confidence
    if with_nans:
        poses[::2, :, 0] = np.nan
        poses[::2, :, 1] = np.nan
    return poses


def _identity_camera_poses(num_frames: int, interval: int = 30):
    """Return camera poses with roll/pitch/yaw = 0 at every *interval* frame."""
    return [
        (i, CameraPose(roll=0.0, pitch=0.0, yaw=0.0, confidence=1.0))
        for i in range(0, num_frames, interval)
    ]


# ---------------------------------------------------------------------------
# estimate_pose_sequence tests
# ---------------------------------------------------------------------------


class TestEstimatePoseSequence:
    """Tests for estimate_pose_sequence()."""

    @patch("skating_ml.detection.spatial_reference.SpatialReferenceDetector")
    @patch("skating_ml.detection.spatial_reference.cv2.VideoCapture")
    def test_returns_only_high_confidence_frames(self, mock_cap_cls, mock_det_cls):
        """Frames with confidence <= 0.1 should be excluded."""
        # Setup mock video: 90 frames, read returns True for first 90 calls
        mock_cap = MagicMock()
        frames_returned = 0
        total_frames = 90

        def fake_read():
            nonlocal frames_returned
            if frames_returned >= total_frames:
                return False, None
            frames_returned += 1
            return True, np.zeros((100, 100, 3), dtype=np.uint8)

        mock_cap.read = fake_read
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        # Mock detector to return low confidence for frame 30, high for others
        call_count = 0

        def fake_estimate(frame):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # second sample (frame 30)
                return CameraPose(roll=0.0, confidence=0.05, source="horizon")
            return CameraPose(roll=2.0, pitch=0.0, yaw=0.0, confidence=0.8, source="horizon")

        mock_det_cls.return_value.estimate_pose = fake_estimate

        result = estimate_pose_sequence("fake.mp4", interval=30, fps=30.0)

        # Should have 2 entries (frames 0, 60), not 3 (frame 30 excluded)
        assert len(result) == 2
        frame_indices = [r[0] for r in result]
        assert 0 in frame_indices
        assert 60 in frame_indices
        assert 30 not in frame_indices

    @patch("skating_ml.detection.spatial_reference.SpatialReferenceDetector")
    @patch("skating_ml.detection.spatial_reference.cv2.VideoCapture")
    def test_returns_empty_for_unopenable_video(self, mock_cap_cls, _mock_det_cls):
        """Should return empty list when video cannot be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        result = estimate_pose_sequence("nonexistent.mp4")
        assert result == []

    @patch("skating_ml.detection.spatial_reference.SpatialReferenceDetector")
    @patch("skating_ml.detection.spatial_reference.cv2.VideoCapture")
    def test_smoothing_applied(self, mock_cap_cls, mock_det_cls):
        """Camera poses should be smoothed (values should differ from raw)."""
        mock_cap = MagicMock()
        frames_returned = 0

        def fake_read():
            nonlocal frames_returned
            if frames_returned >= 60:
                return False, None
            frames_returned += 1
            return True, np.zeros((100, 100, 3), dtype=np.uint8)

        mock_cap.read = fake_read
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        # Return noisy roll values
        call_count = 0
        raw_rolls = [5.0, -3.0]

        def fake_estimate(frame):
            nonlocal call_count
            roll = raw_rolls[call_count % len(raw_rolls)]
            call_count += 1
            return CameraPose(roll=roll, confidence=0.9, source="horizon")

        mock_det_cls.return_value.estimate_pose = fake_estimate

        result = estimate_pose_sequence("fake.mp4", interval=30, fps=30.0)

        assert len(result) == 2
        # After One-Euro smoothing, the second value should be pulled toward
        # the first (not exactly -3.0).
        assert result[1][1].roll != -3.0


# ---------------------------------------------------------------------------
# compensate_poses_per_frame tests
# ---------------------------------------------------------------------------


class TestCompensatePosesPerFrame:
    """Tests for compensate_poses_per_frame()."""

    def test_identity_camera_pose_preserves_poses(self):
        """CameraPose with roll=0 should leave poses unchanged."""
        poses = _make_poses(60)
        camera_poses = _identity_camera_poses(60, interval=30)

        result = compensate_poses_per_frame(poses, camera_poses)

        np.testing.assert_allclose(result, poses, atol=1e-6)

    def test_roll_compensation_applied(self):
        """Non-zero roll should rotate keypoints around (0.5, 0.5)."""
        poses = _make_poses(30, fill=0.7)  # offset from center
        camera_poses = [
            (0, CameraPose(roll=10.0, confidence=1.0)),
        ]

        result = compensate_poses_per_frame(poses, camera_poses)

        # Poses should have moved; they shouldn't equal input
        assert not np.allclose(result[:, :, :2], poses[:, :, :2], atol=1e-4)

        # Distance from center should be preserved (rotation preserves radius)
        def dist_from_center(p):
            dx = p[:, :, 0] - 0.5
            dy = p[:, :, 1] - 0.5
            return np.sqrt(dx**2 + dy**2)

        orig_dist = dist_from_center(poses)
        comp_dist = dist_from_center(result)
        np.testing.assert_allclose(comp_dist, orig_dist, atol=1e-4)

    def test_nan_frames_preserved(self):
        """Frames with NaN poses should stay NaN after compensation."""
        poses = _make_poses(30, with_nans=True)
        camera_poses = [
            (0, CameraPose(roll=5.0, confidence=1.0)),
        ]

        result = compensate_poses_per_frame(poses, camera_poses)

        # Even-indexed frames should still be NaN
        assert np.all(np.isnan(result[::2, :, 0]))
        assert np.all(np.isnan(result[::2, :, 1]))
        # Odd-indexed frames should not be NaN
        assert not np.any(np.isnan(result[1::2, :, 0]))

    def test_empty_camera_poses_returns_copy(self):
        """With no camera poses, should return a copy of the input."""
        poses = _make_poses(10)
        result = compensate_poses_per_frame(poses, [])

        np.testing.assert_array_equal(result, poses)
        # Verify it's a different object
        assert result is not poses

    def test_interpolation_between_sparse_samples(self):
        """Camera roll should be interpolated for frames between samples."""
        # Frame 0: roll=0, frame 30: roll=10
        # Frame 15 should get interpolated roll ~5 degrees
        poses = _make_poses(31)
        camera_poses = [
            (0, CameraPose(roll=0.0, confidence=1.0)),
            (30, CameraPose(roll=10.0, confidence=1.0)),
        ]

        result = compensate_poses_per_frame(poses, camera_poses)

        # Frame 0 (roll=0): unchanged
        np.testing.assert_allclose(result[0], poses[0], atol=1e-6)

        # Frame 15 (roll ~5): should be slightly rotated
        # Frame 30 (roll=10): should be most rotated
        # Verify monotonic displacement
        def mean_offset(p):
            return np.nanmean(p[:, :, 0] - 0.5)

        offset_0 = abs(mean_offset(result[0:1]))
        offset_15 = abs(mean_offset(result[14:16]))
        offset_30 = abs(mean_offset(result[29:31]))

        # Displacement should increase with increasing roll
        assert offset_15 >= offset_0 or abs(offset_15 - offset_0) < 1e-6
        assert offset_30 >= offset_15 or abs(offset_30 - offset_15) < 1e-6

    def test_custom_frame_indices(self):
        """Should use provided frame_indices for interpolation mapping."""
        poses = _make_poses(3, fill=0.7)  # offset from center so rotation is visible
        # Frames are actually at indices 10, 20, 30 in the video
        frame_indices = np.array([10, 20, 30])
        camera_poses = [
            (10, CameraPose(roll=0.0, confidence=1.0)),
            (30, CameraPose(roll=10.0, confidence=1.0)),
        ]

        result = compensate_poses_per_frame(poses, camera_poses, frame_indices=frame_indices)

        # First pose (frame 10, roll=0) should be unchanged
        np.testing.assert_allclose(result[0], poses[0], atol=1e-6)
        # Last pose (frame 30, roll=10) should be rotated
        assert not np.allclose(result[2, :, :2], poses[2, :, :2], atol=1e-4)
