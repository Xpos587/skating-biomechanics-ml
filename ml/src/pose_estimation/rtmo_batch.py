"""Batch RTMO inference — bypasses rtmlib for direct ONNX batch calls."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

RTMO_INPUT_SIZE = 640
RTMO_MODELS = {
    "lightweight": "data/models/rtmo/rtmo-s.onnx",
    "balanced": "data/models/rtmo/rtmo-m.onnx",
    "performance": "data/models/rtmo/rtmo-l.onnx",
}


def preprocess_batch(
    frames: list[np.ndarray],
    input_size: int = RTMO_INPUT_SIZE,
) -> tuple[np.ndarray, list[float]]:
    """Preprocess frames for RTMO batch inference.

    Args:
        frames: List of BGR frames (H, W, 3) uint8.
        input_size: Model input size (640).

    Returns:
        (batch_tensor, ratios) where batch_tensor is (B, 3, input_size, input_size) float32
        and ratios is list of per-frame scale factors.
    """
    batch_size = len(frames)
    batch_tensor = np.zeros((batch_size, 3, input_size, input_size), dtype=np.float32)
    ratios = []

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]

        # Letterbox resize: ratio = min(640/h, 640/w)
        ratio = min(input_size / h, input_size / w)
        ratios.append(ratio)

        # Resize with INTER_LINEAR
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to (input_size, input_size) with gray fill value 114
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_top = (input_size - new_h) // 2
        pad_left = (input_size - new_w) // 2
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        # Transpose HWC -> CHW
        transposed = padded.transpose(2, 0, 1)

        # Cast to float32 (NO normalization - raw pixel values)
        batch_tensor[i] = np.ascontiguousarray(transposed, dtype=np.float32)

    return batch_tensor, ratios


def postprocess_batch(
    dets: np.ndarray,
    keypoints: np.ndarray,
    ratios: list[float],
    score_thr: float = 0.3,
    nms_thr: float = 0.45,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decode RTMO batch output.

    Args:
        dets: (B, N, 5) detection outputs.
        keypoints: (B, N, 17, 3) keypoint outputs.
        ratios: Per-frame scale factors from preprocessing.
        score_thr: Minimum detection score.
        nms_thr: NMS IoU threshold.

    Returns:
        List of (keypoints, scores) tuples per frame.
        keypoints: (M, 17, 2) pixel coords, scores: (M, 17).
        M = number of detected persons after NMS (varies per frame).
    """
    batch_size = dets.shape[0]
    results = []

    for b in range(batch_size):
        ratio = ratios[b]

        # Extract boxes and scores from dets
        boxes = dets[b, :, :4]  # (N, 4)
        det_scores = dets[b, :, 4]  # (N,)

        # Extract keypoints and kp_scores
        kp_coords = keypoints[b, :, :, :2]  # (N, 17, 2)
        kp_scores = keypoints[b, :, :, 2]  # (N, 17)

        # Rescale to original image coordinates
        boxes = boxes / ratio
        kp_coords = kp_coords / ratio

        # Filter by score threshold
        score_mask = det_scores >= score_thr
        if not np.any(score_mask):
            results.append(
                (np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32))
            )
            continue

        boxes = boxes[score_mask]
        det_scores = det_scores[score_mask]
        kp_coords = kp_coords[score_mask]
        kp_scores = kp_scores[score_mask]

        # Apply NMS
        keep_indices = _nms(boxes, det_scores, nms_thr)

        if len(keep_indices) == 0:
            results.append(
                (np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32))
            )
            continue

        # Keep NMS-passing detections
        final_kp_coords = kp_coords[keep_indices]  # (M, 17, 2)
        final_kp_scores = kp_scores[keep_indices]  # (M, 17)

        results.append((final_kp_coords, final_kp_scores))

    return results


def _nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> list[int]:
    """Single-class NMS.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].
        scores: (N,) array of detection scores.
        nms_thr: IoU threshold for suppression.

    Returns:
        List of indices to keep.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


class BatchRTMO:
    """Direct ONNX batch RTMO inference.

    Args:
        mode: Model preset.
        device: "cpu" or "cuda".
        score_thr: Detection score threshold.
        nms_thr: NMS IoU threshold.
    """

    def __init__(
        self,
        mode: str = "balanced",
        device: str = "auto",
        score_thr: float = 0.3,
        nms_thr: float = 0.45,
    ) -> None:
        """Initialize BatchRTMO.

        Args:
            mode: Model preset — "lightweight", "balanced", "performance".
            device: Device to use — "auto", "cpu", "cuda".
            score_thr: Detection score threshold.
            nms_thr: NMS IoU threshold.
        """
        self._mode = mode
        self._device = device
        self._score_thr = score_thr
        self._nms_thr = nms_thr

        # Resolve device
        if device == "auto":
            from ..device import DeviceConfig

            self._device = DeviceConfig(device="auto").device
        else:
            self._device = device

        # Load model (prefer FP16 variant when available)
        model_path = Path(RTMO_MODELS[mode])
        fp16_model_path = Path(str(model_path).replace(".onnx", "-fp16.onnx"))
        if fp16_model_path.exists():
            model_path = fp16_model_path
            logger.info(f"Using FP16 model: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(
                f"RTMO model not found: {model_path}\n"
                f"Download models with: uv run python ml/scripts/download_ml_models.py"
            )

        # Create ONNX session with optimized SessionOptions
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse = True
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=providers,
        )

        # Get input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        # Warm-up inference (eliminates first-call CUDA compilation latency)
        dummy = np.zeros((1, 3, RTMO_INPUT_SIZE, RTMO_INPUT_SIZE), dtype=np.float32)
        self._session.run(self._output_names, {self._input_name: dummy})

        logger.info(f"BatchRTMO initialized: mode={mode}, device={self._device}")

    def infer_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run batch inference on frames.

        Args:
            frames: List of BGR frames (H, W, 3).

        Returns:
            List of (keypoints, scores) per frame.
        """
        if not frames:
            return []

        # Preprocess
        batch_tensor, ratios = preprocess_batch(frames)

        # Run inference
        outputs = self._session.run(
            self._output_names,
            {self._input_name: batch_tensor},
        )

        dets = outputs[0]  # (B, N, 5)
        keypoints = outputs[1]  # (B, N, 17, 3)

        # Postprocess
        results = postprocess_batch(
            dets,
            keypoints,
            ratios,
            score_thr=self._score_thr,
            nms_thr=self._nms_thr,
        )

        return results

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "_session"):
            del self._session
