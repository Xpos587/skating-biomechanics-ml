"""Tests for skeleton drawer Sports2D-style updates."""

import numpy as np
import pytest

from src.visualization.skeleton.drawer import draw_skeleton


def _frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestSports2DBoneColors:
    def test_left_bone_is_green(self):
        """Left-side bones should use Sports2D green color."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[4] = [0.5, 0.4]  # LHIP (normalized)
        pose[5] = [0.5, 0.6]  # LKNEE (normalized)
        frame = _frame()
        draw_skeleton(frame, pose, 480, 640, confidence_threshold=0.0, line_width=10)
        # Center of the line - should be green (0, 255, 0) in BGR
        center = frame[240, 320]
        assert center[1] > 200, f"Expected green, got BGR={center}"  # Green channel high

    def test_right_bone_is_orange(self):
        """Right-side bones should use Sports2D orange color."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[1] = [0.55, 0.4]  # RHIP (normalized)
        pose[2] = [0.55, 0.6]  # RKNEE (normalized)
        frame = _frame()
        draw_skeleton(frame, pose, 480, 640, confidence_threshold=0.0, line_width=10)
        # Center of the line - should be orange (0, 128, 255) in BGR
        center = frame[240, 352]
        # Orange has high red channel (255 in BGR[2])
        assert center[2] > 200, f"Expected orange (high red), got BGR={center}"


class TestFootKeypoints:
    def test_draws_foot_keypoints(self):
        """Should draw foot keypoints when provided."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[0] = [320, 80]   # HEAD
        pose[5] = [310, 200]  # LHIP
        pose[6] = [310, 280]  # LKNEE
        pose[7] = [310, 360]  # LFOOT

        foot_kp = np.array([
            [290, 365, 0.9],  # L_Heel
            [330, 370, 0.9],  # L_BigToe
            [310, 372, 0.5],  # L_SmallToe (skipped)
            [340, 365, 0.9],  # R_Heel
            [380, 370, 0.9],  # R_BigToe
            [360, 372, 0.5],  # R_SmallToe (skipped)
        ], dtype=np.float32)

        frame_before = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_after = frame_before.copy()
        draw_skeleton(frame_after, pose, 480, 640, foot_keypoints=foot_kp)

        assert not np.array_equal(frame_before, frame_after)

    def test_no_foot_keypoints_no_error(self):
        """Should work fine without foot keypoints."""
        pose = np.zeros((17, 2), dtype=np.float32)
        frame = _frame()
        result = draw_skeleton(frame, pose, 480, 640)
        assert result is frame

    def test_draws_heel_toe_line(self):
        """Should draw line connecting heel to big toe."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[7] = [310, 360]  # LFOOT
        foot_kp = np.array([
            [290, 365, 0.9],  # L_Heel
            [330, 370, 0.9],  # L_BigToe
            [310, 372, 0.0],  # L_SmallToe
            [0, 0, 0.0],      # R_Heel
            [0, 0, 0.0],      # R_BigToe
            [0, 0, 0.0],      # R_SmallToe
        ], dtype=np.float32)

        frame = _frame()
        draw_skeleton(frame, pose, 480, 640, foot_keypoints=foot_kp)

        foot_region = frame[360:375, 285:335]
        assert np.any(foot_region > 0)
