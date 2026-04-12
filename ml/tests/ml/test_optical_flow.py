"""Tests for NeuFlowV2 optical flow wrapper."""

from unittest import mock

import numpy as np
import pytest


class TestOpticalFlowEstimator:
    """Tests for dense optical flow estimation."""

    def test_estimate_returns_flow_field(self):
        """estimate() returns (H, W, 2) float32 flow field."""
        from skating_ml.extras.optical_flow import OpticalFlowEstimator

        mock_session = mock.MagicMock()
        # NeuFlowV2 outputs (1, 2, 432, 768)
        mock_session.run.return_value = [np.random.rand(1, 2, 432, 768).astype(np.float32)]

        est = OpticalFlowEstimator.__new__(OpticalFlowEstimator)
        est._session = mock_session
        est._input_names = ["input1", "input2"]
        est._prev_frame = None

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        flow = est.estimate(frame1, frame2)

        assert flow.shape == (480, 640, 2)
        assert flow.dtype == np.float32

    def test_estimate_frame_size_mismatch_raises(self):
        """estimate() raises ValueError if frames have different sizes."""
        from skating_ml.extras.optical_flow import OpticalFlowEstimator

        est = OpticalFlowEstimator.__new__(OpticalFlowEstimator)

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="same size"):
            est.estimate(frame1, frame2)

    def test_estimate_from_previous(self):
        """estimate_from_previous() caches previous frame."""
        from skating_ml.extras.optical_flow import OpticalFlowEstimator

        mock_session = mock.MagicMock()
        mock_session.run.return_value = [np.zeros((1, 2, 432, 768), dtype=np.float32)]

        est = OpticalFlowEstimator.__new__(OpticalFlowEstimator)
        est._session = mock_session
        est._input_names = ["input1", "input2"]
        est._prev_frame = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        flow = est.estimate_from_previous(frame)

        assert flow is None  # No previous frame yet
        assert est._prev_frame is not None  # Frame cached

        flow = est.estimate_from_previous(frame)
        assert flow.shape == (480, 640, 2)

    def test_reset(self):
        """reset() clears cached previous frame."""
        from skating_ml.extras.optical_flow import OpticalFlowEstimator

        est = OpticalFlowEstimator.__new__(OpticalFlowEstimator)
        est._prev_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        est.reset()
        assert est._prev_frame is None


class TestOpticalFlowLayer:
    """Tests for optical flow visualization layer."""

    def test_render_adds_flow_overlay(self):
        """OpticalFlowLayer renders HSV color wheel flow."""
        from skating_ml.visualization.layers.base import LayerContext
        from skating_ml.visualization.layers.optical_flow_layer import OpticalFlowLayer

        layer = OpticalFlowLayer(opacity=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        flow = np.random.rand(480, 640, 2).astype(np.float32) * 10 - 5

        ctx = LayerContext(frame_width=640, frame_height=480)
        ctx.custom_data["flow_field"] = flow

        result = layer.render(frame, ctx)
        assert result.shape == (480, 640, 3)

    def test_render_no_flow_returns_unchanged(self):
        """Layer is a no-op when no flow_field in context."""
        from skating_ml.visualization.layers.base import LayerContext
        from skating_ml.visualization.layers.optical_flow_layer import OpticalFlowLayer

        layer = OpticalFlowLayer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        ctx = LayerContext(frame_width=640, frame_height=480)

        result = layer.render(frame, ctx)
        assert np.array_equal(result, frame)
