"""Tests for tracklet building and merging."""

from unittest.mock import MagicMock

import numpy as np

from src.tracking.skeletal_identity import SkeletalIdentityExtractor
from src.tracking.tracklet_merger import (
    Tracklet,
    TrackletMerger,
    build_tracklets,
)


def _make_pose(cx: float, cy: float) -> np.ndarray:
    pose = np.zeros((17, 3), dtype=np.float32)
    pose[:, 0] = cx
    pose[:, 1] = cy
    pose[:, 2] = 0.8
    return pose


def _make_foot() -> np.ndarray:
    return np.zeros((6, 3), dtype=np.float32)


class TestTracklet:
    def test_properties(self):
        t = Tracklet(track_id=0, frames=[10, 11, 12])
        assert t.start_frame == 10
        assert t.end_frame == 12
        assert t.length == 3

    def test_empty(self):
        t = Tracklet(track_id=0)
        assert t.start_frame == -1
        assert t.end_frame == -1
        assert t.length == 0

    def test_centroid(self):
        pose = _make_pose(0.5, 0.3)
        t = Tracklet(track_id=0, frames=[5], poses={5: pose})
        cx, _cy = t.centroid_at(5)
        assert abs(cx - 0.5) < 1e-5

    def test_missing_centroid(self):
        t = Tracklet(track_id=0, frames=[5])
        cx, _cy = t.centroid_at(5)
        assert np.isnan(cx)


class TestBuildTracklets:
    def test_single_track(self):
        fd = {
            0: {0: (_make_pose(0.3, 0.5), _make_foot())},
            1: {0: (_make_pose(0.31, 0.51), _make_foot())},
        }
        ts = build_tracklets(fd)
        assert len(ts) == 1
        assert ts[0].track_id == 0
        assert ts[0].length == 2

    def test_multiple_tracks(self):
        fd = {
            0: {0: (_make_pose(0.3, 0.5), _make_foot()), 1: (_make_pose(0.7, 0.5), _make_foot())},
            1: {
                0: (_make_pose(0.31, 0.51), _make_foot()),
                1: (_make_pose(0.69, 0.49), _make_foot()),
            },
        }
        ts = build_tracklets(fd)
        assert len(ts) == 2

    def test_empty(self):
        assert build_tracklets({}) == []


class TestTrackletMerger:
    def test_spatial_only_near_wins(self):
        merger = TrackletMerger(identity_extractor=None, similarity_threshold=0.5)
        target = Tracklet(
            track_id=0,
            frames=list(range(10)),
            poses={i: _make_pose(0.3, 0.5) for i in range(10)},
        )
        near = Tracklet(
            track_id=1,
            frames=list(range(15, 25)),
            poses={15 + i: _make_pose(0.32, 0.52) for i in range(10)},
        )
        far = Tracklet(
            track_id=2,
            frames=list(range(15, 25)),
            poses={15 + i: _make_pose(0.8, 0.8) for i in range(10)},
        )
        match = merger.find_best_match(target, [near, far])
        assert match is not None
        assert match.track_id == 1

    def test_spatial_only_too_far(self):
        merger = TrackletMerger(identity_extractor=None, similarity_threshold=0.9)
        target = Tracklet(
            track_id=0,
            frames=list(range(10)),
            poses={i: _make_pose(0.3, 0.5) for i in range(10)},
        )
        far = Tracklet(
            track_id=1,
            frames=list(range(900, 910)),
            poses={900 + i: _make_pose(0.8, 0.8) for i in range(10)},
        )
        assert merger.find_best_match(target, [far]) is None

    def test_bone_similarity_drives_matching(self):
        extractor = MagicMock(spec=SkeletalIdentityExtractor)
        # Target has similar profile to far_correct (pa), different from near_wrong (pb)
        pa = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        pb = np.array([0.5, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5])
        # Pre-normalized profiles
        pa_norm = pa / np.linalg.norm(pa)
        pb_norm = pb / np.linalg.norm(pb)
        extractor.extract_profile.side_effect = [
            pa_norm,
            pb_norm,
            pa_norm,
        ]  # target, near_wrong, far_correct

        merger = TrackletMerger(
            identity_extractor=extractor,
            similarity_threshold=0.5,
            bone_weight=1.0,
            spatial_weight=0.0,
            temporal_weight=0.0,
        )
        target = Tracklet(
            track_id=0,
            frames=list(range(10)),
            poses={i: _make_pose(0.3, 0.5) for i in range(10)},
        )
        near_wrong = Tracklet(
            track_id=1,
            frames=list(range(15, 25)),
            poses={15 + i: _make_pose(0.31, 0.51) for i in range(10)},
        )
        far_correct = Tracklet(
            track_id=2,
            frames=list(range(20, 30)),
            poses={20 + i: _make_pose(0.6, 0.4) for i in range(10)},
        )
        match = merger.find_best_match(target, [near_wrong, far_correct])
        assert match is not None
        assert match.track_id == 2

    def test_merge_combines_frames(self):
        t1 = Tracklet(
            track_id=0,
            frames=[1, 2],
            poses={1: _make_pose(0.3, 0.5), 2: _make_pose(0.31, 0.51)},
            foot_keypoints={1: _make_foot(), 2: _make_foot()},
        )
        t2 = Tracklet(
            track_id=1,
            frames=[5, 6],
            poses={5: _make_pose(0.32, 0.52), 6: _make_pose(0.33, 0.53)},
            foot_keypoints={5: _make_foot(), 6: _make_foot()},
        )
        merged = TrackletMerger().merge(t1, t2)
        assert merged.track_id == 0
        assert merged.frames == [1, 2, 5, 6]
        assert len(merged.poses) == 4

    def test_no_match_short_target(self):
        merger = TrackletMerger(identity_extractor=None)
        target = Tracklet(track_id=0, frames=[5])
        assert merger.find_best_match(target, []) is None
