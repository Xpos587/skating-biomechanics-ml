"""AthletePose3D 3D pose estimator.

Monocular 3D pose estimation using fine-tuned AthletePose3D models.

Models:
- MotionAgFormer-S: 59MB, fast, suitable for RTX 3050 Ti
- TCPFormer: 422MB, more accurate

Reference: AthletePose3D: A Large-Scale 3D Sports Pose Dataset
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .biomechanics_estimator import Biomechanics3DEstimator


class AthletePose3DExtractor:
    """Monocular 3D pose estimation using AthletePose3D.

    Processes 2D poses (H3.6M 17-keypoint format) and outputs 3D poses.
    Supports MotionAGFormer and TCPFormer architectures.

    Model Types:
        - motionagformer-s: Small, fast (59MB)
        - motionagformer-b: Base model (not tested)
        - tcpformer: High accuracy (422MB)
    """

    # Temporal window size (frames)
    TEMPORAL_WINDOW = 81

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str = "auto",
        model_type: str = "motionagformer-s",
        use_simple: bool = False,
    ):
        """Initialize the 3D pose estimator.

        Args:
            model_path: Path to model checkpoint (.pth.tr file), or None for simple mode
            device: "cuda", "cpu", or "auto" (default)
            model_type: Model architecture type ("motionagformer-s", "motionagformer-b", "tcpformer")
            use_simple: If True, use biomechanics estimator instead of ML model
        """
        self.model_path = Path(model_path) if model_path else None
        self.model_type = model_type.lower()
        self.use_simple = use_simple or (model_path is None)

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Temporal buffer for 81-frame window
        self.temporal_buffer: deque[np.ndarray] = deque(maxlen=self.TEMPORAL_WINDOW)

        # Load model (lazy loading on first use)
        self.model: torch.nn.Module | None = None
        self._model_loaded = False

        # Simple biomechanics estimator (fallback)
        self._simple_estimator = Biomechanics3DEstimator() if self.use_simple else None

    def _load_model(self) -> torch.nn.Module:
        """Load the 3D pose model (MotionAGFormer or TCPFormer)."""
        if self._model_loaded:
            return self.model  # type: ignore

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Import models from our models directory
        import sys
        from pathlib import Path

        # Add models directory to path
        models_dir = Path(__file__).parent.parent / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        # Choose model class based on type
        if self.model_type == "tcpformer":
            from tcpformer import MemoryInducedTransformer
            ModelClass = MemoryInducedTransformer
            # TCPFormer uses 3 input channels (x, y, confidence)
            dim_in = 3
            # TCPFormer checkpoint has specific configuration
            n_layers = 16
            dim_feat = 128
        else:
            # Default to MotionAGFormer
            from motionagformer import MotionAGFormer
            ModelClass = MotionAGFormer
            # MotionAGFormer also uses 3 input channels (x, y, confidence)
            dim_in = 3

            # Determine model configuration based on type
            if self.model_type == "motionagformer-s":
                n_layers = 4
                dim_feat = 64
            elif self.model_type == "motionagformer-b":
                n_layers = 8
                dim_feat = 128
            else:
                # Default to small config
                n_layers = 4
                dim_feat = 64

        # Create model
        self.model = ModelClass(
            n_layers=n_layers,
            dim_in=dim_in,
            dim_feat=dim_feat,
            dim_rep=512,
            dim_out=3,  # 3D output (x, y, z)
            num_heads=4,
            num_joints=17,
            n_frames=self.TEMPORAL_WINDOW,
        )

        # Extract state dict from checkpoint
        if "model" in checkpoint:
            # TCPFormer format
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix (from DataParallel wrapper)
        # AND move tensors to target device to avoid device mismatch
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            # Move tensor to target device
            if isinstance(v, torch.Tensor):
                new_state_dict[new_key] = v.to(self.device)
            else:
                new_state_dict[new_key] = v

        # Load weights (handle different architectures)
        if self.model_type == "tcpformer":
            # TCPFormer uses strict loading (architecture matches)
            self.model.load_state_dict(new_state_dict, strict=True)
        else:
            # MotionAGFormer - use non-strict (checkpoint has only att branches)
            self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True

        return self.model

    def extract_frame(
        self,
        pose_2d: np.ndarray,
    ) -> np.ndarray | None:
        """Extract 3D pose from single 2D pose (with temporal context).

        Args:
            pose_2d: (17, 2) or (17, 3) array in H3.6M format
                - If (17, 2): x, y coordinates
                - If (17, 3): x, y, confidence

        Returns:
            pose_3d: (17, 3) array with x, y, z coordinates
                Returns None if temporal buffer not full yet
        """
        # Add to temporal buffer
        self.temporal_buffer.append(pose_2d)

        # Need full window for inference
        if len(self.temporal_buffer) < self.TEMPORAL_WINDOW:
            return None

        # Stack into temporal window
        window = np.stack(list(self.temporal_buffer))  # (81, 17, 2 or 3)

        return self._extract_window(window)

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
        # Use simple estimator if enabled or no model
        if self.use_simple or self._simple_estimator is not None:
            return self._simple_estimator.estimate_3d(poses_2d)

        n_frames = poses_2d.shape[0]

        # Ensure correct format (add confidence if needed)
        if poses_2d.shape[2] == 2:
            poses_with_conf = np.zeros((n_frames, 17, 3), dtype=np.float32)
            poses_with_conf[:, :, :2] = poses_2d
            poses_with_conf[:, :, 2] = 1.0
            poses_2d = poses_with_conf

        # Pad sequence to multiple of window size
        pad_size = (self.TEMPORAL_WINDOW - (n_frames % self.TEMPORAL_WINDOW)) % self.TEMPORAL_WINDOW
        if pad_size > 0:
            padding = np.tile(poses_2d[-1:], (pad_size, 1, 1))
            poses_2d_padded = np.vstack([poses_2d, padding])
        else:
            poses_2d_padded = poses_2d

        # Process in windows with stride for overlap
        stride = self.TEMPORAL_WINDOW // 4  # 25% overlap for smoother output
        poses_3d_accum = np.zeros((len(poses_2d_padded), 17, 3), dtype=np.float32)
        poses_3d_count = np.zeros(len(poses_2d_padded), dtype=np.int32)

        for i in range(0, len(poses_2d_padded) - self.TEMPORAL_WINDOW + 1, stride):
            window = poses_2d_padded[i : i + self.TEMPORAL_WINDOW]
            poses_3d_window = self._extract_window(window)

            # Accumulate poses from this window
            end_idx = i + self.TEMPORAL_WINDOW
            poses_3d_accum[i:end_idx] += poses_3d_window
            poses_3d_count[i:end_idx] += 1

        # Average overlapping regions
        poses_3d = poses_3d_accum[:n_frames] / poses_3d_count[:n_frames, np.newaxis, np.newaxis]

        return poses_3d

    def _extract_window(self, window: np.ndarray) -> np.ndarray:
        """Extract 3D pose from temporal window.

        Args:
            window: (81, 17, 2) or (81, 17, 3) array

        Returns:
            pose_3d: (81, 17, 3) array - all frames in window
        """
        # Convert to tensor
        tensor = torch.from_numpy(window).float().to(self.device)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # (1, 81, 17, 2 or 3)

        # Run inference
        with torch.no_grad():
            model = self._load_model()
            output = model(tensor)  # (1, 81, 17, 3)

        # Extract all frames from output
        pose_3d = output[0].cpu().numpy()  # (81, 17, 3)

        return pose_3d

    def reset(self):
        """Reset temporal buffer."""
        self.temporal_buffer.clear()


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
        model_path: Path to model checkpoint
        model_type: Model architecture type ("motionagformer-s", "tcpformer")
        device: "cuda", "cpu", or "auto"

    Returns:
        poses_3d: (N, 17, 3) array with x, y, z coordinates

    Raises:
        ValueError: If poses_2d is not in H3.6M 17-keypoint format
    """
    # Validate input format - must be H3.6M 17 keypoints
    if poses_2d.shape[1] != 17:
        raise ValueError(
            f"poses_2d must have 17 keypoints in H3.6M format, got {poses_2d.shape[1]}. "
            f"Use H36MExtractor for new pose extraction."
        )

    # Create extractor and run
    extractor = AthletePose3DExtractor(model_path, device, model_type)
    return extractor.extract_sequence(poses_2d)
