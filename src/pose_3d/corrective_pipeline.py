"""3D-based corrective lens for 2D skeleton overlays.

Pipeline: 2D poses -> 3D lift -> kinematic constraints -> anchor projection -> blend

Uses intermediate 3D estimation to correct common 2D pose errors (occluded
joints, left/right swaps, anatomically impossible poses) and projects the
result back to 2D for visualization.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class CorrectiveLens:
    """3D-based corrective lens for 2D skeleton overlays.

    Pipeline: 2D poses -> 3D lift -> kinematic constraints -> anchor projection -> blend
    """

    def __init__(self, model_path: Path | str | None = None, device: str = "auto"):
        """Initialize with optional MotionAGFormer model path.

        Falls back to Biomechanics3DEstimator if no model provided.
        """
        from .athletepose_extractor import AthletePose3DExtractor
        from .biomechanics_estimator import Biomechanics3DEstimator

        if model_path is not None and Path(model_path).exists():
            self.extractor: AthletePose3DExtractor | Biomechanics3DEstimator = (
                AthletePose3DExtractor(model_path=model_path, device=device)
            )
            self._uses_ml = True
        else:
            self.extractor = Biomechanics3DEstimator()
            self._uses_ml = False

    def correct_sequence(
        self,
        poses_2d_norm: NDArray[np.float32],  # (N, 17, 2) normalized [0,1]
        fps: float,
        width: int,
        height: int,
        confidences: NDArray[np.float32] | None = None,  # (N, 17) per-joint confidence
        blend_threshold: float = 0.5,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Apply 3D corrective pipeline to 2D pose sequence.

        Returns:
            poses_2d_corrected: (N, 17, 2) normalized [0,1] corrected poses
            poses_3d: (N, 17, 3) intermediate 3D poses in meters
        """
        # Step 1: Lift to 3D
        if self._uses_ml:
            poses_3d = self.extractor.extract_sequence(poses_2d_norm)  # type: ignore[union-attr]
        else:
            poses_3d = self.extractor.estimate_3d(poses_2d_norm)  # type: ignore[union-attr]

        # Step 2: Apply kinematic constraints
        from .kinematic_constraints import apply_kinematic_constraints

        poses_3d = apply_kinematic_constraints(poses_3d, fps, confidences)

        # Step 3: Anchor-based projection back to 2D
        from .anchor_projection import anchor_project, blend_by_confidence

        poses_2d_corrected = anchor_project(poses_3d, poses_2d_norm, width, height)

        # Step 4: Blend with original by confidence
        if confidences is not None:
            poses_2d_corrected = blend_by_confidence(
                poses_2d_norm, poses_2d_corrected, confidences, threshold=blend_threshold
            )

        return poses_2d_corrected, poses_3d
