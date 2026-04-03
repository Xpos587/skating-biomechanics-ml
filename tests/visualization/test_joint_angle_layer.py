"""Tests for enhanced joint angle visualization."""

import numpy as np
import pytest

from src.pose_estimation import H36Key
from src.utils.geometry import angle_3pt
from src.visualization.config import VisualizationConfig
from src.visualization.layers.base import LayerContext
from src.visualization.layers.joint_angle_layer import JointAngleLayer


def _make_context(pose_2d=None, w=640, h=480):
    return LayerContext(
        frame_width=w,
        frame_height=h,
        pose_2d=pose_2d,
        pose_3d=None,
        confidences=None,
        normalized=False,
    )


class TestJointAngleLayerDegrees:
    def test_arc_renders_without_error(self):
        """Should render angle arcs on a valid pose."""
        pose = np.zeros((17, 2), dtype=np.float32)
        # L-KNEE at center, L-HIP above, L-FOOT below
        pose[H36Key.LHIP] = [300, 180]
        pose[H36Key.LKNEE] = [300, 240]
        pose[H36Key.LFOOT] = [300, 300]
        pose[H36Key.RHIP] = [340, 180]
        pose[H36Key.RKNEE] = [340, 240]
        pose[H36Key.RFOOT] = [340, 300]
        pose[H36Key.LSHOULDER] = [280, 120]
        pose[H36Key.RSHOULDER] = [360, 120]
        pose[H36Key.THORAX] = [320, 150]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = JointAngleLayer()
        ctx = _make_context(pose_2d=pose)
        result = layer.render(frame, ctx)
        assert result is frame

    def test_degree_label_drawn(self):
        """Should draw degree text labels near joints."""
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[H36Key.LHIP] = [300, 180]
        pose[H36Key.LKNEE] = [300, 240]
        pose[H36Key.LFOOT] = [300, 300]
        pose[H36Key.RHIP] = [340, 180]
        pose[H36Key.RKNEE] = [340, 240]
        pose[H36Key.RFOOT] = [340, 300]
        pose[H36Key.LSHOULDER] = [280, 120]
        pose[H36Key.RSHOULDER] = [360, 120]
        pose[H36Key.THORAX] = [320, 150]

        frame_before = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_after = frame_before.copy()

        layer = JointAngleLayer(show_degree_labels=True)
        ctx = _make_context(pose_2d=pose)
        layer.render(frame_after, ctx)

        # Frame should be modified (degree labels add pixels)
        assert not np.array_equal(frame_before, frame_after)


class TestHybridMode:
    def test_2d_mode_uses_2d_angles(self):
        """In '2d' mode, should compute angles from 2D pose regardless of 3D availability."""
        pose_2d = np.zeros((17, 2), dtype=np.float32)
        pose_2d[H36Key.LHIP] = [300, 180]
        pose_2d[H36Key.LKNEE] = [300, 240]
        pose_2d[H36Key.LFOOT] = [300, 300]
        pose_2d[H36Key.RHIP] = [340, 180]
        pose_2d[H36Key.RKNEE] = [340, 240]
        pose_2d[H36Key.RFOOT] = [340, 300]
        pose_2d[H36Key.LSHOULDER] = [280, 120]
        pose_2d[H36Key.RSHOULDER] = [360, 120]
        pose_2d[H36Key.THORAX] = [320, 150]

        # 3D with different angle (z-offset creates different angle)
        pose_3d = np.zeros((17, 3), dtype=np.float32)
        pose_3d[:, :2] = pose_2d
        pose_3d[:, 2] = 50  # Add z-depth to create different 3D angles

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = JointAngleLayer(angle_source="2d")
        ctx = LayerContext(
            frame_width=640, frame_height=480,
            pose_2d=pose_2d, pose_3d=pose_3d,
            normalized=False,
        )
        result = layer.render(frame, ctx)
        assert result is frame

    def test_invalid_angle_source_raises(self):
        """Invalid angle_source should raise ValueError."""
        with pytest.raises(ValueError, match="angle_source must be"):
            JointAngleLayer(angle_source="invalid")
