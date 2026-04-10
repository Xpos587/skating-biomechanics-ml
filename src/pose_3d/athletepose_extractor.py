"""AthletePose3D 3D pose estimator (ONNX Runtime).

Monocular 3D pose estimation using fine-tuned AthletePose3D models
via ONNX Runtime — no PyTorch dependency.

Models:
- MotionAgFormer-S: 59MB, fast, suitable for RTX 3050 Ti
- TCPFormer: 422MB, more accurate

Reference: AthletePose3D: A Large-Scale 3D Sports Pose Dataset
"""

from pathlib import Path

import numpy as np

from .onnx_extractor import ONNXPoseExtractor


class AthletePose3DExtractor:
    """Monocular 3D pose estimation using AthletePose3D (ONNX Runtime).

    Processes 2D poses (H3.6M 17-keypoint format) and outputs 3D poses.
    Requires a .onnx model file (PyTorch .pth.tr is no longer supported).

    Model Types:
        - motionagformer-s: Small, fast (59MB)
        - motionagformer-b: Base model (not tested)
        - tcpformer: High accuracy (422MB)
    """

    TEMPORAL_WINDOW = 81

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str = "auto",
        model_type: str = "motionagformer-s",
    ) -> None:
        """Initialize the 3D pose estimator.

        Args:
            model_path: Path to .onnx model file.
            device: "cuda", "cpu", or "auto" (default).
            model_type: Model architecture type (informational only — ONNX is architecture-agnostic).
        """
        if model_path is None:
            raise FileNotFoundError(
                "AthletePose3DExtractor requires a model_path. Pass a .onnx model file."
            )

        self.model_path = Path(model_path)
        self.model_type = model_type.lower()
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


# Standalone function for quick inference
def extract_3d_poses(
    poses_2d: np.ndarray,
    model_path: Path | str,
    model_type: str = "motionagformer-s",
    device: str = "auto",
) -> np.ndarray:
    """Extract 3D poses from 2D pose sequence.

    Convenience function that creates extractor and runs inference.

    Args:
        poses_2d: (N, 17, 2) array in H3.6M format
        model_path: Path to .onnx model file
        model_type: Model architecture type (informational only).
        device: "cuda", "cpu", or "auto"

    Returns:
        poses_3d: (N, 17, 3) array with x, y, z coordinates

    Raises:
        ValueError: If poses_2d is not in H3.6M 17-keypoint format
    """
    if poses_2d.shape[1] != 17:
        raise ValueError(
            f"poses_2d must have 17 keypoints in H3.6M format, got {poses_2d.shape[1]}. "
            f"Use RTMPoseExtractor for new pose extraction."
        )

    extractor = AthletePose3DExtractor(model_path, device, model_type)
    return extractor.extract_sequence(poses_2d)
