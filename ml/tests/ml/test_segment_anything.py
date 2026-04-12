"""Tests for SAM2 segmentation wrapper."""

from unittest import mock

import numpy as np


class TestSegmentAnything:
    """Tests for SAM2 image segmentation."""

    @staticmethod
    def _make_estimator(mock_ve_session, mock_pd_session):
        """Create SegmentAnything bypassing __init__."""
        from skating_ml.extras.segment_anything import SegmentAnything

        est = SegmentAnything.__new__(SegmentAnything)
        est._ve_session = mock_ve_session
        est._pd_session = mock_pd_session
        est._input_size = 1024
        est._ve_input_name = "pixel_values"
        est._ve_output_names = ["image_embeddings.0", "image_embeddings.1", "image_embeddings.2"]
        est._pd_output_names = ["iou_scores", "pred_masks", "object_score_logits"]
        return est

    def test_segment_returns_mask(self):
        """segment() returns (H, W) bool mask."""
        mock_ve = mock.MagicMock()
        mock_pd = mock.MagicMock()

        # Vision encoder outputs
        mock_ve.run.return_value = [
            np.random.rand(1, 32, 256, 256).astype(np.float32),
            np.random.rand(1, 64, 128, 128).astype(np.float32),
            np.random.rand(1, 256, 64, 64).astype(np.float32),
        ]

        # Prompt decoder outputs: pred_masks shape [batch, prompts, masks, H, W]
        mock_pd.run.return_value = [
            np.array([[[0.95]]]),  # iou_scores
            np.ones((1, 1, 1, 256, 256), dtype=np.float32),  # pred_masks
            np.array([[[0.9]]]),  # object_score_logits
        ]

        est = self._make_estimator(mock_ve, mock_pd)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = est.segment(frame, point=(320, 240))

        assert mask.shape == (480, 640)
        assert mask.dtype == bool

    def test_segment_with_no_point_returns_empty(self):
        """segment() with point=None returns None (no prompt)."""
        from skating_ml.extras.segment_anything import SegmentAnything

        est = SegmentAnything.__new__(SegmentAnything)
        result = est.segment(np.zeros((480, 640, 3), dtype=np.uint8), point=None)
        assert result is None

    def test_segment_resize_back_to_original(self):
        """Mask is resized to original frame size."""
        mock_ve = mock.MagicMock()
        mock_pd = mock.MagicMock()

        mock_ve.run.return_value = [
            np.random.rand(1, 32, 256, 256).astype(np.float32),
            np.random.rand(1, 64, 128, 128).astype(np.float32),
            np.random.rand(1, 256, 64, 64).astype(np.float32),
        ]

        mock_pd.run.return_value = [
            np.array([[[0.95]]]),
            np.ones((1, 1, 1, 256, 256), dtype=np.float32),
            np.array([[[0.9]]]),
        ]

        est = self._make_estimator(mock_ve, mock_pd)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mask = est.segment(frame, point=(640, 360))

        assert mask.shape == (720, 1280)


class TestSegmentationLayer:
    """Tests for segmentation mask visualization."""

    def test_render_adds_mask_overlay(self):
        """SegmentationMaskLayer renders semi-transparent mask."""
        from skating_ml.visualization.layers.base import LayerContext
        from skating_ml.visualization.layers.segmentation_layer import SegmentationMaskLayer

        layer = SegmentationMaskLayer(opacity=0.3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:400, 200:500] = True

        ctx = LayerContext(frame_width=640, frame_height=480)
        ctx.custom_data["seg_mask"] = mask

        result = layer.render(frame, ctx)
        assert result.shape == (480, 640, 3)
        # Masked region should have color
        assert not np.all(result == 0)

    def test_render_no_mask_returns_unchanged(self):
        """No-op when no seg_mask in context."""
        from skating_ml.visualization.layers.base import LayerContext
        from skating_ml.visualization.layers.segmentation_layer import SegmentationMaskLayer

        layer = SegmentationMaskLayer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        ctx = LayerContext(frame_width=640, frame_height=480)

        result = layer.render(frame, ctx)
        assert np.array_equal(result, frame)
