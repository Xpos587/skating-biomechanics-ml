# src/pose_3d/onnx_extractor.py
"""ONNX Runtime-based 3D pose estimation.

Drop-in replacement for AthletePose3DExtractor / TCPFormerExtractor
that uses ONNX Runtime instead of PyTorch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ONNXPoseExtractor:
    """3D pose estimation using ONNX Runtime.

    Loads a converted MotionAGFormer or TCPFormer ONNX model and
    runs inference via onnxruntime — no PyTorch dependency needed.

    Args:
        model_path: Path to .onnx model file.
        device: ``"cpu"`` or ``"cuda"`` (falls back to CPU).
        temporal_window: Number of frames per inference window (default 81).
    """

    TEMPORAL_WINDOW = 81

    def __init__(
        self,
        model_path: Path | str,
        device: str = "cpu",
        temporal_window: int = 81,
    ) -> None:
        import onnxruntime as ort

        self.temporal_window = temporal_window
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        active = self.session.get_providers()[0]
        logger.info(f"ONNX 3D pose: {model_path.name}, provider={active}")

    def estimate_3d(self, poses_2d: NDArray[np.float32]) -> NDArray[np.float32]:
        """Estimate 3D poses from 2D input.

        Args:
            poses_2d: (N, 17, 2) normalized coordinates [0, 1].

        Returns:
            (N, 17, 3) with estimated z-coordinates.
        """
        n_frames = len(poses_2d)
        w = self.temporal_window

        if n_frames <= w:
            return self._infer_window(poses_2d)[:n_frames]

        # Sliding window with stride = w // 2
        stride = w // 2
        results = np.zeros((n_frames, 17, 3), dtype=np.float32)
        counts = np.zeros(n_frames, dtype=np.float32)

        start = 0
        while start < n_frames:
            end = min(start + w, n_frames)
            window = poses_2d[start:end]
            out = self._infer_window(window)
            results[start:end] += out[: end - start]
            counts[start:end] += 1
            if end == n_frames:
                break
            start += stride

        # Average overlapping regions
        counts = np.maximum(counts, 1)[:, np.newaxis, np.newaxis]
        results /= counts
        return results

    def _infer_window(self, poses_2d: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run inference on a single window (pad if needed). (N, 17, 2) → (N, 17, 3)."""
        n = len(poses_2d)
        w = self.temporal_window

        # Pad to window size if needed
        if n < w:
            pad_count = w - n
            # Repeat last frame for padding
            padded = np.concatenate([poses_2d, np.tile(poses_2d[-1:], (pad_count, 1, 1))], axis=0)
        else:
            padded = poses_2d

        # Add confidence channel (=1.0) and batch dim: (1, w, 17, 3)
        conf = np.ones((w, 17, 1), dtype=np.float32)
        inp = np.concatenate([padded, conf], axis=2)[np.newaxis]

        result = self.session.run(None, {self.input_name: inp})[0]
        # result: (1, w, 17, 3) → (w, 17, 3)
        return result[0]  # type: ignore[index]
