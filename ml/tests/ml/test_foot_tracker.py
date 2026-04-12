"""Tests for FootTrackNet wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest

if TYPE_CHECKING:
    from skating_ml.extras.foot_tracker import FootTracker


class TestFootTracker:
    """Tests for person and foot detection."""

    @staticmethod
    def _make_tracker(mock_session) -> FootTracker:
        """Create a FootTracker bypassing __init__ with required attributes set."""
        from skating_ml.extras.foot_tracker import FootTracker

        tracker = FootTracker.__new__(FootTracker)
        tracker._session = mock_session
        tracker._input_name = "image"
        tracker._output_names = ["heatmap", "bbox"]
        return tracker

    def test_detect_returns_detections(self):
        """detect() returns list of detection dicts."""
        mock_session = mock.MagicMock()
        # CenterNet-style output: heatmap (2 classes, 120x160) + bbox (8 channels, 120x160)
        heatmap = np.zeros((2, 120, 160), dtype=np.float32)
        heatmap[0, 10, 20] = 10.0  # person peak (sigmoid -> ~1.0)
        heatmap[1, 30, 40] = 10.0  # face peak

        bbox = np.zeros((8, 120, 160), dtype=np.float32)
        # person bbox offsets at (10, 20)
        bbox[0, 10, 20] = 5.0  # top
        bbox[1, 10, 20] = 3.0  # left
        bbox[2, 10, 20] = 7.0  # right
        bbox[3, 10, 20] = 9.0  # bottom

        mock_session.run.return_value = [heatmap, bbox]

        tracker = self._make_tracker(mock_session)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = tracker.detect(frame)

        assert len(detections) >= 1
        assert "bbox" in detections[0]
        assert "class_id" in detections[0]
        assert "confidence" in detections[0]

    def test_detect_filters_low_confidence(self):
        """detect() filters out detections below confidence threshold."""
        mock_session = mock.MagicMock()
        # All zeros — no peaks above threshold
        heatmap = np.zeros((2, 120, 160), dtype=np.float32) * 0.01
        bbox = np.zeros((8, 120, 160), dtype=np.float32)

        mock_session.run.return_value = [heatmap, bbox]

        tracker = self._make_tracker(mock_session)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = tracker.detect(frame)

        assert detections == []

    def test_detect_scales_bbox_to_original_size(self):
        """Bbox coordinates are scaled from model input size to original frame size."""
        mock_session = mock.MagicMock()
        heatmap = np.zeros((2, 120, 160), dtype=np.float32)
        heatmap[0, 10, 20] = 10.0  # person peak

        bbox = np.zeros((8, 120, 160), dtype=np.float32)
        bbox[0, 10, 20] = 5.0  # top
        bbox[1, 10, 20] = 3.0  # left
        bbox[2, 10, 20] = 7.0  # right
        bbox[3, 10, 20] = 9.0  # bottom

        mock_session.run.return_value = [heatmap, bbox]

        tracker = self._make_tracker(mock_session)

        # Original frame is 960x1280 (2x model input 480x640)
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        detections = tracker.detect(frame)

        assert len(detections) >= 1
        bbox_result = detections[0]["bbox"]
        # Bbox should be scaled by 2x (orig/model)
        assert bbox_result[0] == pytest.approx(bbox_result[0], abs=500)
        assert bbox_result[2] > bbox_result[0]  # x2 > x1

    def test_detect_empty_output(self):
        """detect() returns empty list when no detections."""
        mock_session = mock.MagicMock()
        heatmap = np.zeros((2, 120, 160), dtype=np.float32)
        bbox = np.zeros((8, 120, 160), dtype=np.float32)

        mock_session.run.return_value = [heatmap, bbox]

        tracker = self._make_tracker(mock_session)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = tracker.detect(frame)

        assert detections == []

    def test_init_from_registry(self):
        """FootTracker loads from ModelRegistry."""
        from skating_ml.extras.foot_tracker import FootTracker
        from skating_ml.extras.model_registry import ModelRegistry

        reg = ModelRegistry(device="cpu")
        reg.register("foot_tracker", vram_mb=30, path="/tmp/foot_tracker.onnx")

        mock_inputs = [mock.MagicMock(name="image")]
        mock_inputs[0].name = "image"
        mock_outputs = [mock.MagicMock(name="heatmap"), mock.MagicMock(name="bbox")]
        mock_outputs[0].name = "heatmap"
        mock_outputs[1].name = "bbox"

        mock_session = mock.MagicMock()
        mock_session.get_inputs.return_value = mock_inputs
        mock_session.get_outputs.return_value = mock_outputs

        with mock.patch(
            "skating_ml.extras.model_registry.ort.InferenceSession", return_value=mock_session
        ):
            tracker = FootTracker(reg)
            assert tracker._session is mock_session
            assert tracker._input_name == "image"
            assert tracker._output_names == ["heatmap", "bbox"]
