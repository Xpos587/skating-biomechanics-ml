"""Tests for batch RTMO inference module."""

import numpy as np
import pytest

from src.pose_estimation.rtmo_batch import (
    RTMO_INPUT_SIZE,
    _nms,
    postprocess_batch,
    preprocess_batch,
)


class TestPreprocessBatch:
    def test_output_shape(self):
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
        tensor, ratios, _pad_offsets = preprocess_batch(frames)
        assert tensor.shape == (4, 3, RTMO_INPUT_SIZE, RTMO_INPUT_SIZE)
        assert tensor.dtype == np.float32
        assert len(ratios) == 4

    def test_ratio_for_square_frame(self):
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _tensor, ratios, _pad_offsets = preprocess_batch([frame])
        assert ratios[0] == 1.0

    def test_ratio_for_tall_frame(self):
        frame = np.random.randint(0, 255, (1280, 640, 3), dtype=np.uint8)
        _tensor, ratios, _pad_offsets = preprocess_batch([frame])
        assert ratios[0] == pytest.approx(640 / 1280)

    def test_padding_is_gray(self):
        frame = np.zeros((1280, 640, 3), dtype=np.uint8)
        tensor, _ratios, _pad_offsets = preprocess_batch([frame])
        # 1280x640 -> ratio = 0.5, new_h = 640, new_w = 320
        # Pad left=160, right=160 (center the 320px content in 640px)
        # Check right padding (width 480:640)
        right_padding = tensor[0, :, :, 480:640]  # All channels, all height, width 480-640
        assert np.allclose(right_padding, 114.0, atol=1.0)
        # Also check left padding
        left_padding = tensor[0, :, :, 0:160]  # All channels, all height, width 0-160
        assert np.allclose(left_padding, 114.0, atol=1.0)

    def test_single_frame(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _ratios, _pad_offsets = preprocess_batch([frame])
        assert tensor.shape == (1, 3, RTMO_INPUT_SIZE, RTMO_INPUT_SIZE)

    def test_no_normalization(self):
        """RTMO does NOT normalize — values should be raw pixel values as float32."""
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        tensor, _ratios, _pad_offsets = preprocess_batch([frame])
        # Check that values are raw pixel values, not normalized to [0, 1]
        assert tensor[0, 0, 0, 0] == pytest.approx(200.0)


class TestPostprocessBatch:
    def test_no_detections(self):
        dets = np.zeros((2, 0, 5), dtype=np.float32)
        keypoints = np.zeros((2, 0, 17, 3), dtype=np.float32)
        results = postprocess_batch(dets, keypoints, [1.0, 1.0], [(0, 0), (0, 0)])
        assert len(results) == 2
        for kp, sc in results:
            assert kp.shape == (0, 17, 2)
            assert sc.shape == (0, 17)

    def test_single_detection_passes_nms(self):
        dets = np.zeros((1, 1, 5), dtype=np.float32)
        dets[0, 0] = [100, 100, 200, 200, 0.9]  # High score bbox
        keypoints = np.random.rand(1, 1, 17, 3).astype(np.float32)
        keypoints[0, 0, :, 2] = 0.8  # High kp confidence
        results = postprocess_batch(dets, keypoints, [1.0], [(0, 0)], score_thr=0.3)
        assert len(results) == 1
        kp, sc = results[0]
        assert kp.shape[0] == 1
        assert sc.shape[0] == 1

    def test_low_score_filtered(self):
        dets = np.zeros((1, 1, 5), dtype=np.float32)
        dets[0, 0] = [100, 100, 200, 200, 0.1]  # Low score
        keypoints = np.random.rand(1, 1, 17, 3).astype(np.float32)
        results = postprocess_batch(dets, keypoints, [1.0], [(0, 0)], score_thr=0.3)
        kp, _sc = results[0]
        assert kp.shape[0] == 0

    def test_ratio_rescales_keypoints(self):
        dets = np.zeros((1, 1, 5), dtype=np.float32)
        dets[0, 0] = [320, 320, 640, 640, 0.9]
        keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
        keypoints[0, 0, :, 0] = 320.0  # x in 640-space
        keypoints[0, 0, :, 1] = 240.0  # y in 640-space
        keypoints[0, 0, :, 2] = 0.9
        ratio = 0.5  # image was 1280x1280, so ratio = 640/1280 = 0.5
        results = postprocess_batch(dets, keypoints, [ratio], [(0, 0)], score_thr=0.3)
        kp, _ = results[0]
        # 320 / 0.5 = 640 in original coords
        assert kp[0, 0, 0] == pytest.approx(640.0)


class TestNMS:
    def test_no_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, nms_thr=0.45)
        assert len(keep) == 2

    def test_full_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, nms_thr=0.45)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_empty(self):
        keep = _nms(np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32), 0.45)
        assert len(keep) == 0
