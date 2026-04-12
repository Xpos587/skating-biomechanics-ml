"""Tests for multi-person pose tracker."""

import numpy as np
import pytest

from skating_ml.detection.pose_tracker import PoseTracker
from skating_ml.types import H36Key


@pytest.fixture
def tracker():
    """Create a default tracker instance."""
    return PoseTracker(max_disappeared=5, min_hits=2, fps=30.0)


@pytest.fixture
def sample_poses():
    """Create sample poses for testing (H3.6M 17kp format)."""
    # Two different people with different proportions
    poses = np.zeros((2, 17, 2), dtype=np.float32)

    # Person 1: Taller, broader shoulders
    poses[0, H36Key.LSHOULDER] = [0.3, 0.2]
    poses[0, H36Key.RSHOULDER] = [0.7, 0.2]
    poses[0, H36Key.LWRIST] = [0.25, 0.4]
    poses[0, H36Key.RWRIST] = [0.75, 0.4]
    poses[0, H36Key.LHIP] = [0.35, 0.5]
    poses[0, H36Key.RHIP] = [0.65, 0.5]
    poses[0, H36Key.LKNEE] = [0.35, 0.7]
    poses[0, H36Key.LFOOT] = [0.35, 0.9]

    # Person 2: Shorter, narrower shoulders
    poses[1, H36Key.LSHOULDER] = [0.4, 0.25]
    poses[1, H36Key.RSHOULDER] = [0.6, 0.25]
    poses[1, H36Key.LWRIST] = [0.38, 0.45]
    poses[1, H36Key.RWRIST] = [0.62, 0.45]
    poses[1, H36Key.LHIP] = [0.42, 0.5]
    poses[1, H36Key.RHIP] = [0.58, 0.5]
    poses[1, H36Key.LKNEE] = [0.42, 0.65]
    poses[1, H36Key.LFOOT] = [0.42, 0.8]

    return poses


class TestPoseTracker:
    """Test PoseTracker functionality."""

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.max_disappeared == 5
        assert tracker.min_hits == 2
        assert tracker.fps == 30.0
        assert tracker.dt == pytest.approx(1.0 / 30.0)
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 0

    def test_empty_update(self, tracker):
        """Test update with no detections."""
        track_ids = tracker.update(np.array([]))
        assert track_ids == []
        assert len(tracker.tracks) == 0

    def test_single_detection_creates_track(self, tracker, sample_poses):
        """Test that a single detection creates a track."""
        single_pose = sample_poses[0:1]
        track_ids = tracker.update(single_pose)

        assert len(track_ids) == 1
        assert track_ids[0] >= 0

    def test_multiple_detections_create_multiple_tracks(self, tracker, sample_poses):
        """Test that multiple detections create multiple tracks."""
        track_ids = tracker.update(sample_poses)

        assert len(track_ids) == 2
        assert track_ids[0] != track_ids[1]

    def test_track_persistence_across_frames(self, tracker, sample_poses):
        """Test that tracks persist across frames."""
        track_ids_1 = tracker.update(sample_poses)

        # Move poses slightly
        poses_moved = sample_poses.copy()
        poses_moved[:, :, 0] += 0.01

        track_ids_2 = tracker.update(poses_moved)

        # Same tracks should be returned
        assert set(track_ids_1) == set(track_ids_2)

    def test_biometric_extraction(self, tracker, sample_poses):
        """Test biometric extraction for re-identification."""
        biometrics = tracker._extract_biometrics(sample_poses[0])

        # Check that all expected ratios are present
        expected_keys = [
            "shoulder_width/torso",
            "femur/tibia",
            "arm_span/height",
            "torso/height",
            "shoulder_width/height",
        ]
        for key in expected_keys:
            assert key in biometrics
            assert biometrics[key] > 0

    def test_biometric_distance_same_person(self, tracker, sample_poses):
        """Test biometric distance for same person (different poses)."""
        bio1 = tracker._extract_biometrics(sample_poses[0])
        bio2 = tracker._extract_biometrics(sample_poses[0])  # Same pose

        distance = tracker._biometric_distance(bio1, bio2)
        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_biometric_distance_different_people(self, tracker, sample_poses):
        """Test biometric distance for different people."""
        bio1 = tracker._extract_biometrics(sample_poses[0])
        bio2 = tracker._extract_biometrics(sample_poses[1])

        distance = tracker._biometric_distance(bio1, bio2)
        # Different proportions should give non-zero distance
        assert distance > 0

    def test_confirmed_tracks_filter(self, tracker, sample_poses):
        """Test filtering of confirmed tracks."""
        tracker.update(sample_poses)
        tracker.update(sample_poses)

        confirmed = tracker.get_confirmed_tracks()
        assert len(confirmed) >= 1  # At least one track should be confirmed

    def test_mid_hip_calculation(self, tracker, sample_poses):
        """Test mid-hip calculation from poses."""
        mid_hips = tracker._get_mid_hips(sample_poses)

        assert len(mid_hips) == 2
        for i, mid_hip in enumerate(mid_hips):
            # Mid-hip should be average of left and right hips
            expected = (sample_poses[i, H36Key.LHIP] + sample_poses[i, H36Key.RHIP]) / 2
            np.testing.assert_array_almost_equal(mid_hip, expected)

    def test_track_state_initialization(self, tracker, sample_poses):
        """Test track state initialization."""
        tracker.update(sample_poses)

        # First track should have state
        assert len(tracker.tracks) > 0
        track = tracker.tracks[0]
        assert track.hits >= 1
