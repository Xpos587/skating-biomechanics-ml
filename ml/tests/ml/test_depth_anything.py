"""Tests for Depth Anything V2 wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import numpy as np

if TYPE_CHECKING:
    from skating_ml.extras.depth_anything import DepthEstimator


class TestDepthEstimator:
    """Tests for monocular depth estimation."""

    @staticmethod
    def _make_estimator(mock_session, input_name: str = "image") -> DepthEstimator:
        """Create a DepthEstimator bypassing __init__ with required attributes set."""
        from skating_ml.extras.depth_anything import DepthEstimator

        est = DepthEstimator.__new__(DepthEstimator)
        est._session = mock_session
        est._input_size = 518
        est._input_name = input_name
        return est

    def test_estimate_returns_depth_map(self):
        """estimate() returns (H, W) float32 depth map."""

        mock_session = mock.MagicMock()
        mock_session.get_input_details.return_value = [
            {"name": "image", "shape": [1, 3, 518, 518], "type": "float32"}
        ]
        mock_session.run.return_value = [np.random.rand(1, 518, 518).astype(np.float32)]

        est = self._make_estimator(mock_session)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = est.estimate(frame)

        assert depth.shape == (480, 640)
        assert depth.dtype == np.float32

    def test_estimate_normalizes_to_0_1(self):
        """Depth map values are normalized to [0, 1]."""

        mock_session = mock.MagicMock()
        mock_session.get_input_details.return_value = [
            {"name": "image", "shape": [1, 3, 518, 518], "type": "float32"}
        ]
        raw_depth = np.random.rand(1, 518, 518).astype(np.float32) * 10 + 5
        mock_session.run.return_value = [raw_depth]

        est = self._make_estimator(mock_session)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = est.estimate(frame)

        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_estimate_prepares_input_correctly(self):
        """Input is resized to model size and transposed to NCHW."""

        mock_session = mock.MagicMock()
        mock_session.get_input_details.return_value = [
            {"name": "image", "shape": [1, 3, 518, 518], "type": "float32"}
        ]
        mock_session.run.return_value = [np.zeros((1, 518, 518), dtype=np.float32)]

        est = self._make_estimator(mock_session)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        est.estimate(frame)

        # Verify session.run was called
        call_args = mock_session.run.call_args
        assert call_args is not None
        # Input should be NCHW with model input size.
        # session.run(None, feed_dict) passes feed_dict as positional arg.
        input_feed = call_args[0][1]
        assert input_feed is not None
        assert input_feed["image"].shape == (1, 3, 518, 518)

    def test_init_from_registry(self):
        """DepthEstimator loads from ModelRegistry."""
        from skating_ml.extras.depth_anything import DepthEstimator
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry(device="cpu")
        reg.register("depth_anything", vram_mb=200, path="/tmp/depth.onnx")

        mock_session = mock.MagicMock()
        mock_session.get_input_details.return_value = [
            {"name": "image", "shape": [1, 3, 518, 518], "type": "float32"}
        ]

        with mock.patch("skating_ml.extras.model_registry.ort.InferenceSession", return_value=mock_session):
            est = DepthEstimator(reg)
            assert est._session is mock_session


class TestDepthMapLayer:
    """Tests for depth map visualization layer."""

    def test_render_adds_depth_overlay(self):
        """DepthMapLayer renders color-mapped depth onto frame."""
        from skating_ml.visualization.layers.base import LayerContext
        from skating_ml.visualization.layers.depth_layer import DepthMapLayer

        layer = DepthMapLayer(opacity=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32)

        ctx = LayerContext(frame_width=640, frame_height=480)
        ctx.custom_data["depth_map"] = depth

        result = layer.render(frame, ctx)
        assert result.shape == (480, 640, 3)
        # Frame should no longer be all-black (depth overlay applied)
        assert not np.all(result == 0)

    def test_render_no_depth_returns_unchanged(self):
        """Layer is a no-op when no depth_map in context."""
        from skating_ml.visualization.layers.base import LayerContext
        from skating_ml.visualization.layers.depth_layer import DepthMapLayer

        layer = DepthMapLayer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        ctx = LayerContext(frame_width=640, frame_height=480)

        result = layer.render(frame, ctx)
        assert np.array_equal(result, frame)
