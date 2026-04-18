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
        device: str = "auto",
        temporal_window: int = 81,
    ) -> None:
        import onnxruntime as ort

        from ..device import DeviceConfig

        self.temporal_window = temporal_window
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        cfg = DeviceConfig(device=device)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=cfg.onnx_providers
        )
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

        # Collect all windows for batched inference
        windows = []
        window_starts = []

        start = 0
        while start < n_frames:
            end = min(start + w, n_frames)
            window_starts.append((start, end))
            window = poses_2d[start:end]
            windows.append(window)
            if end == n_frames:
                break
            start += stride

        # Batch inference (process all windows at once)
        batch_results = self._infer_batch(windows)

        # Scatter results back to frame array
        results = np.zeros((n_frames, 17, 3), dtype=np.float32)
        counts = np.zeros(n_frames, dtype=np.float32)

        for (start, end), out in zip(window_starts, batch_results, strict=True):
            frame_count = end - start
            results[start:end] += out[:frame_count]
            counts[start:end] += 1

        # Average overlapping regions
        counts = np.maximum(counts, 1)[:, np.newaxis, np.newaxis]
        results /= counts
        return results

    def _infer_batch(self, windows: list[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        """Run batched inference on multiple windows.

        Args:
            windows: List of (N_i, 17, 2) arrays, where N_i <= temporal_window

        Returns:
            List of (N_i, 17, 3) result arrays
        """
        w = self.temporal_window
        batch_size = len(windows)

        # Pad all windows to temporal_window size
        padded_windows = []
        for window in windows:
            n = len(window)
            if n < w:
                # Repeat last frame for padding
                pad_count = w - n
                padded = np.concatenate([window, np.tile(window[-1:], (pad_count, 1, 1))], axis=0)
            else:
                padded = window
            padded_windows.append(padded)

        # Stack into batch: (batch_size, w, 17, 2)
        batch_input = np.stack(padded_windows, axis=0)

        # Add confidence channel: (batch_size, w, 17, 3)
        conf = np.ones((batch_size, w, 17, 1), dtype=np.float32)
        batch_input = np.concatenate([batch_input, conf], axis=3)

        # Run batched inference
        result = self.session.run(None, {self.input_name: batch_input})[0]

        # Extract results, truncating to original window lengths
        results = []
        for i, window in enumerate(windows):
            n = len(window)
            results.append(result[i][:n])

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
