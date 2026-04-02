"""TCPFormer 3D pose lifter wrapper.

Memory-Induced Transformer for monocular 3D human pose estimation.
Uses 81-frame temporal window with adaptive fusion of attention and graph branches.
"""

from pathlib import Path

import numpy as np
import torch

from .biomechanics_estimator import Biomechanics3DEstimator


class TCPFormerExtractor:
    """3D pose lifting using TCPFormer (Memory-Induced Transformer).

    High-accuracy 3D pose estimation with 422MB model.
    Uses 81-frame temporal window for smooth 3D trajectories.

    Reference: https://github.com/AsukaCamellia/TCPFormer
    """

    # Temporal window size (frames)
    TEMPORAL_WINDOW = 81

    def __init__(
        self,
        model_path: Path | str = "data/models/TCPFormer_ap3d_81.pth.tr",
        device: str = "auto",
        use_simple: bool = False,
    ):
        """Initialize TCPFormer 3D pose lifter.

        Args:
            model_path: Path to TCPFormer checkpoint (.pth.tr file).
            device: "cuda", "cpu", or "auto" (default).
            use_simple: If True, use biomechanics estimator instead of ML model.
        """
        self.model_path = Path(model_path)
        self.use_simple = use_simple or (not self.model_path.exists())

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model (lazy loading on first use)
        self.model: torch.nn.Module | None = None
        self._model_loaded = False

        # Simple biomechanics estimator (fallback)
        self._simple_estimator = Biomechanics3DEstimator() if self.use_simple else None

    def _load_model(self) -> torch.nn.Module:
        """Load TCPFormer model from checkpoint."""
        if self._model_loaded:
            return self.model  # type: ignore

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Import TCPFormer from our models directory
        import sys
        from pathlib import Path

        # Add models directory to path
        models_dir = Path(__file__).parent.parent / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))

        from tcpformer import MemoryInducedTransformer

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        # Extract state dict (checkpoint has 'model' key with 'module.' prefix)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix (from DataParallel wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith("module.") else k] = v

        # Create TCPFormer model with correct configuration
        # From checkpoint analysis: 16 layers, 128 feat dim, 3 input dim
        self.model = MemoryInducedTransformer(
            n_layers=16,
            dim_in=3,  # x, y, confidence
            dim_feat=128,
            dim_rep=512,
            dim_out=3,  # x, y, z output
            num_heads=4,
            num_joints=17,
            n_frames=self.TEMPORAL_WINDOW,
        )

        # Load weights
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True

        return self.model

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

        # Ensure (N, 17, 3) format with confidence
        if poses_2d.shape[2] == 2:
            # Add confidence channel (set to 1.0)
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

        # Process in windows with overlap for smoother output
        stride = self.TEMPORAL_WINDOW // 4  # 25% overlap
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
            window: (81, 17, 3) array with x, y, confidence

        Returns:
            pose_3d: (81, 17, 3) array - all frames in window
        """
        # Convert to tensor
        tensor = torch.from_numpy(window).float().to(self.device)

        # Add batch dimension: (1, 81, 17, 3)
        tensor = tensor.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            model = self._load_model()
            output = model(tensor)  # (1, 81, 17, 3)

        # Extract all frames from output
        pose_3d = output[0].cpu().numpy()  # (81, 17, 3)

        return pose_3d

    def reset(self):
        """Reset any internal state (TCPFormer has no temporal state)."""
        pass
