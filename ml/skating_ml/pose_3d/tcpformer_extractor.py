"""TCPFormer 3D pose lifter wrapper (ONNX Runtime).

Memory-Induced Transformer for monocular 3D human pose estimation.
Uses 81-frame temporal window with ONNX Runtime — no PyTorch dependency.

Reference: https://github.com/AsukaCamellia/TCPFormer
"""

from pathlib import Path

import numpy as np

from .onnx_extractor import ONNXPoseExtractor


class TCPFormerExtractor:
    """3D pose lifting using TCPFormer (ONNX Runtime).

    High-accuracy 3D pose estimation with 422MB model.
    Uses 81-frame temporal window for smooth 3D trajectories.
    """

    TEMPORAL_WINDOW = 81

    def __init__(
        self,
        model_path: Path | str = "data/models/TCPFormer_ap3d_81.onnx",
        device: str = "auto",
    ) -> None:
        """Initialize TCPFormer 3D pose lifter.

        Args:
            model_path: Path to TCPFormer .onnx model file.
            device: "cuda", "cpu", or "auto" (default).
        """
        self.model_path = Path(model_path)
        self._onnx = ONNXPoseExtractor(self.model_path, device=device)

    def extract_sequence(
        self,
        poses_2d: np.ndarray,
    ) -> np.ndarray:
        """Extract 3D poses from 2D pose sequence.

        Args:
            poses_2d: (N, 17, 2) or (N, 17, 3) array in H3.6M format

        Returns:
            poses_3d: (N, 17, 3) array with x, y, z coordinates
        """
        return self._onnx.estimate_3d(poses_2d[:, :, :2])

    def reset(self) -> None:
        """Reset internal state (ONNX extractor is stateless — no-op)."""
