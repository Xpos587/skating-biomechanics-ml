"""Tests for matplotlib person selector logic."""

import numpy as np

from src.pose_estimation.person_selector import compute_bboxes_from_poses, point_in_bbox


class TestComputeBboxes:
    def test_single_person(self):
        """Should compute bbox for single person."""
        poses = np.zeros((1, 17, 2), dtype=np.float32)
        poses[0, 0] = [100, 100]  # head
        poses[0, 8] = [200, 300]  # foot
        bboxes = compute_bboxes_from_poses(poses)
        assert len(bboxes) == 1
        x1, y1, x2, y2 = bboxes[0]
        assert x1 <= 100 <= x2
        assert y1 <= 100 <= y2

    def test_multiple_persons(self):
        """Should compute bbox for each person."""
        poses = np.zeros((2, 17, 2), dtype=np.float32)
        poses[0, 0] = [100, 100]
        poses[1, 0] = [400, 200]
        bboxes = compute_bboxes_from_poses(poses)
        assert len(bboxes) == 2


class TestPointInBbox:
    def test_inside(self):
        assert point_in_bbox(150, 200, (100, 150, 200, 250))

    def test_outside(self):
        assert not point_in_bbox(50, 50, (100, 150, 200, 250))

    def test_edge(self):
        assert point_in_bbox(100, 150, (100, 150, 200, 250))
