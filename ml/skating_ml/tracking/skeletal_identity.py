"""3D skeletal identity extraction for person re-identification.

Uses 3D bone lengths (invariant to clothing, lighting, viewing angle)
as biometric identity signatures. Lifts 2D H3.6M poses to 3D via
MotionAGFormer-S, then computes Euclidean distances between connected
joints to form a bone length vector.

Reference:
    - Hawk-Eye SkeleTRACK: 29-keypoint skeletal tracking
    - AFLink: Appearance-Free Link for tracklet association (StrongSORT)
"""

import logging
from pathlib import Path

import numpy as np

from skating_ml.types import H36Key

logger = logging.getLogger(__name__)

# 12 discriminative bone pairs for identity (indices into H3.6M 17-keypoint format)
BONE_PAIRS: list[tuple[int, int]] = [
    (H36Key.RHIP, H36Key.RKNEE),  # 0: femur R
    (H36Key.RKNEE, H36Key.RFOOT),  # 1: tibia R
    (H36Key.LHIP, H36Key.LKNEE),  # 2: femur L
    (H36Key.LKNEE, H36Key.LFOOT),  # 3: tibia L
    (H36Key.RSHOULDER, H36Key.RELBOW),  # 4: humerus R
    (H36Key.RELBOW, H36Key.RWRIST),  # 5: ulna R
    (H36Key.LSHOULDER, H36Key.LELBOW),  # 6: humerus L
    (H36Key.LELBOW, H36Key.LWRIST),  # 7: ulna L
    (H36Key.HIP_CENTER, H36Key.THORAX),  # 8: spine lower
    (H36Key.THORAX, H36Key.NECK),  # 9: spine upper
    (H36Key.LSHOULDER, H36Key.RSHOULDER),  # 10: shoulder width
    (H36Key.LHIP, H36Key.RHIP),  # 11: pelvis width
]

NUM_BONES = len(BONE_PAIRS)  # 12

# Indices of spine bones for scale normalization
SPINE_INDICES = [8, 9]


def compute_bone_lengths_3d(poses_3d: np.ndarray) -> np.ndarray:
    """Compute 3D Euclidean bone lengths from lifted poses.

    Args:
        poses_3d: (N, 17, 3) poses in meters.

    Returns:
        (N, NUM_BONES) bone lengths. NaN where keypoints are missing.
    """
    n = len(poses_3d)
    bones = np.zeros((n, NUM_BONES), dtype=np.float32)
    for i, (j1, j2) in enumerate(BONE_PAIRS):
        diff = poses_3d[:, j1, :] - poses_3d[:, j2, :]
        bones[:, i] = np.linalg.norm(diff, axis=1)
    return bones


def compute_identity_profile(bones: np.ndarray) -> np.ndarray:
    """Compute a stable, scale-invariant identity profile.

    Applies temporal median filtering and normalizes by spine length.

    Args:
        bones: (N, NUM_BONES) bone lengths.

    Returns:
        (NUM_BONES,) normalized identity vector.
    """
    median_bones = np.nanmedian(bones, axis=0)
    spine = median_bones[SPINE_INDICES[0]] + median_bones[SPINE_INDICES[1]]
    if spine > 1e-6:
        median_bones = median_bones / spine
    return median_bones.astype(np.float32)


def identity_similarity(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Cosine similarity between two identity profiles.

    Returns:
        1.0 for identical, ~0.0 for orthogonal, -1.0 for opposite.
    """
    norm_a = np.linalg.norm(profile_a)
    norm_b = np.linalg.norm(profile_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(profile_a, profile_b) / (norm_a * norm_b))


class SkeletalIdentityExtractor:
    """Extract 3D skeletal identity profiles from 2D pose sequences.

    Uses MotionAGFormer-S for 2D→3D lifting.

    Args:
        model_path: Path to MotionAGFormer-S checkpoint (.pth.tr or .onnx).
        device: Device for inference ("auto", "cuda", "cpu").
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str = "auto",
    ) -> None:
        self._extractor = None
        self._use_ml = False
        if model_path is not None:
            try:
                from skating_ml.pose_3d.athletepose_extractor import (
                    AthletePose3DExtractor,
                )

                self._extractor = AthletePose3DExtractor(
                    model_path=model_path,
                    device=device,
                )
                self._use_ml = True
                logger.info("SkeletalIdentity: using MotionAGFormer-S")
            except Exception as e:
                logger.warning("MotionAGFormer unavailable: %s", e)

    def extract_profile(
        self,
        poses_2d: np.ndarray,
    ) -> np.ndarray | None:
        """Extract identity profile from 2D poses.

        Args:
            poses_2d: (N, 17, 2 or 3) H3.6M poses (normalized [0,1]).

        Returns:
            (NUM_BONES,) identity profile, or None if insufficient data.
        """
        if len(poses_2d) < 10:
            return None

        poses_xy = poses_2d[:, :, :2]
        poses_3d = self._lift_to_3d(poses_xy)
        if poses_3d is None:
            return None

        bones = compute_bone_lengths_3d(poses_3d)
        valid = np.any(~np.isnan(bones), axis=1)
        if valid.sum() < 5:
            return None
        return compute_identity_profile(bones[valid])

    def _lift_to_3d(self, poses_2d: np.ndarray) -> np.ndarray | None:
        """Lift 2D poses to 3D."""
        if self._use_ml and self._extractor is not None:
            try:
                return self._extractor.extract_sequence(poses_2d)
            except Exception as e:
                logger.warning("3D lifting failed: %s", e)
        return None


def compute_2d_skeletal_ratios(pose: np.ndarray) -> np.ndarray:
    """Compute scale-invariant 2D skeletal ratios for anomaly detection.

    Used in the online anti-steal check. These ratios are
    perspective-distorted but stable within the same viewing angle.
    A sudden discontinuity indicates a different person.

    Args:
        pose: (17, 2 or 3) H3.6M pose.

    Returns:
        (5,) ratio vector.
    """
    xy = pose[:, :2]
    sw = np.linalg.norm(xy[H36Key.LSHOULDER] - xy[H36Key.RSHOULDER])
    pw = np.linalg.norm(xy[H36Key.LHIP] - xy[H36Key.RHIP])
    th = np.linalg.norm(xy[H36Key.HIP_CENTER] - xy[H36Key.NECK])
    fl = np.linalg.norm(xy[H36Key.LHIP] - xy[H36Key.LKNEE])
    fr = np.linalg.norm(xy[H36Key.RHIP] - xy[H36Key.RKNEE])
    denom = max(pw, th, 1e-6)
    return np.array([sw / denom, pw / denom, th / denom, fl / denom, fr / denom])
