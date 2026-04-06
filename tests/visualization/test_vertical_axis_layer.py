"""Tests for redesigned VerticalAxisLayer."""

import numpy as np

from src.pose_estimation import H36Key
from src.visualization.layers.base import LayerContext
from src.visualization.layers.vertical_axis_layer import (
    TiltQuality,
    VerticalAxisLayer,
    classify_tilt,
)


def _make_context(pose_2d=None, pose_3d=None, w=640, h=480, normalized=False, frame_idx=0):
    return LayerContext(
        frame_width=w,
        frame_height=h,
        pose_2d=pose_2d,
        pose_3d=pose_3d,
        normalized=normalized,
        frame_idx=frame_idx,
    )


def _upright_pose(w=640, h=480):
    """Pose with spine perfectly vertical (shoulder directly above hip)."""
    pose = np.zeros((17, 2), dtype=np.float32)
    cx = w // 2
    pose[H36Key.LHIP] = [cx - 20, 300]
    pose[H36Key.RHIP] = [cx + 20, 300]
    pose[H36Key.LSHOULDER] = [cx - 15, 180]
    pose[H36Key.RSHOULDER] = [cx + 15, 180]
    pose[H36Key.HEAD] = [cx, 100]
    return pose


def _leaning_forward_pose(w=640, h=480, lean_px=40):
    """Pose with shoulders shifted right (forward lean in image coords)."""
    pose = _upright_pose(w, h)
    pose[H36Key.LSHOULDER][0] += lean_px
    pose[H36Key.RSHOULDER][0] += lean_px
    pose[H36Key.HEAD][0] += lean_px
    return pose


class TestClassifyTilt:
    """Test the tilt quality classification function."""

    def test_zero_tilt_is_good(self):
        assert classify_tilt(0.0) == TiltQuality.GOOD

    def test_small_tilt_is_good(self):
        assert classify_tilt(4.0) == TiltQuality.GOOD

    def test_boundary_good_to_warn(self):
        assert classify_tilt(5.0) == TiltQuality.WARN

    def test_medium_tilt_is_warn(self):
        assert classify_tilt(8.0) == TiltQuality.WARN

    def test_boundary_warn_to_bad(self):
        assert classify_tilt(10.0) == TiltQuality.BAD

    def test_large_tilt_is_bad(self):
        assert classify_tilt(15.0) == TiltQuality.BAD

    def test_negative_tilt(self):
        assert classify_tilt(-4.0) == TiltQuality.GOOD
        assert classify_tilt(-8.0) == TiltQuality.WARN
        assert classify_tilt(-15.0) == TiltQuality.BAD


class TestVerticalAxisLayerRender:
    """Test rendering behavior of VerticalAxisLayer."""

    def test_no_pose_returns_frame_unchanged(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=None)
        result = layer.render(frame, ctx)
        assert result is frame
        assert np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_upright_pose_modifies_frame(self):
        pose = _upright_pose()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame
        # Frame should be modified (gravity line drawn)
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_leaning_pose_modifies_frame(self):
        pose = _leaning_forward_pose()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_normalized_coords_render(self):
        pose = np.zeros((17, 2), dtype=np.float32)
        pose[H36Key.LHIP] = [0.47, 0.6]
        pose[H36Key.RHIP] = [0.53, 0.6]
        pose[H36Key.LSHOULDER] = [0.48, 0.35]
        pose[H36Key.RSHOULDER] = [0.52, 0.35]
        pose[H36Key.HEAD] = [0.5, 0.2]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=True)
        result = layer.render(frame, ctx)
        assert result is frame
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_show_degree_label_false(self):
        """When show_degree_label=False, still renders gravity line but no text."""
        pose = _leaning_forward_pose()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer(show_degree_label=False)
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame
        # Should still draw gravity line and spine
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))


class TestVerticalAxisLayerEdgeCases:
    """Edge cases and robustness tests."""

    def test_nan_pose_returns_unchanged(self):
        """All-NaN pose should not crash."""
        pose = np.full((17, 2), np.nan, dtype=np.float32)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame

    def test_coincident_hip_shoulder(self):
        """Hip and shoulder at same position should not crash."""
        pose = np.zeros((17, 2), dtype=np.float32)
        cx, cy = 320, 240
        pose[H36Key.LHIP] = [cx - 10, cy]
        pose[H36Key.RHIP] = [cx + 10, cy]
        pose[H36Key.LSHOULDER] = [cx - 10, cy]
        pose[H36Key.RSHOULDER] = [cx + 10, cy]
        pose[H36Key.HEAD] = [cx, cy]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer()
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame

    def test_custom_thresholds(self):
        """Custom thresholds should change classification."""
        assert classify_tilt(7.0, 10.0, 20.0) == TiltQuality.GOOD
        assert classify_tilt(15.0, 10.0, 20.0) == TiltQuality.WARN
        assert classify_tilt(25.0, 10.0, 20.0) == TiltQuality.BAD

    def test_backward_compatible_constructor(self):
        """Old-style constructor with config and viz_config still works."""
        from src.visualization.config import LayerConfig, VisualizationConfig

        layer = VerticalAxisLayer(
            config=LayerConfig(enabled=True, z_index=5),
            viz_config=VisualizationConfig(),
        )
        assert layer.enabled
        assert layer.z_index == 5

    def test_head_alignment_with_offset(self):
        """Pose with head offset should draw head alignment indicator."""
        pose = _upright_pose()
        # Shift head to the right
        pose[H36Key.HEAD][0] += 50
        pose[H36Key.HEAD][1] = 80

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        layer = VerticalAxisLayer(show_head_alignment=True)
        ctx = _make_context(pose_2d=pose, normalized=False)
        result = layer.render(frame, ctx)
        assert result is frame
        # Should have drawn something
        assert not np.array_equal(result, np.zeros((480, 640, 3), dtype=np.uint8))

    def test_head_alignment_disabled(self):
        """show_head_alignment=False should not draw head offset line."""
        pose = _upright_pose()
        pose[H36Key.HEAD][0] += 50

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

        layer_on = VerticalAxisLayer(show_head_alignment=True)
        layer_off = VerticalAxisLayer(show_head_alignment=False)
        ctx = _make_context(pose_2d=pose, normalized=False)

        layer_on.render(frame1, ctx)
        layer_off.render(frame2, ctx)

        # Both should be modified, but frame with head alignment should differ
        assert not np.array_equal(frame1, frame2)


class TestTiltDirectionLabel:
    """Test direction label helper."""

    def test_zero_tilt_no_direction(self):
        from src.visualization.layers.vertical_axis_layer import _tilt_direction_label

        assert _tilt_direction_label(0.0) == ""
        assert _tilt_direction_label(0.5) == ""

    def test_positive_tilt_right(self):
        from src.visualization.layers.vertical_axis_layer import _tilt_direction_label

        assert _tilt_direction_label(5.0) == "R"
        assert _tilt_direction_label(1.5) == "R"

    def test_negative_tilt_left(self):
        from src.visualization.layers.vertical_axis_layer import _tilt_direction_label

        assert _tilt_direction_label(-5.0) == "L"
        assert _tilt_direction_label(-1.5) == "L"
