"""Anchor-based projection of corrected 3D poses back to 2D.

Projects root-relative 3D poses to 2D image coordinates by anchoring at the
detected 2D hip position and computing a per-frame scale from torso length
ratio between 2D and 3D. Uses an orthographic approximation which is sufficient
because body depth range (~0.5m) vs camera distance (~3-5m) produces small
perspective distortion.
"""

import numpy as np

from ..types import H36Key


def _median_fill_nan(values: np.ndarray, radius: int = 5) -> np.ndarray:
    """Fill NaN values with temporal median of nearby valid frames.

    Args:
        values: Array of shape (N, D) that may contain NaN rows.
        radius: Number of nearby frames on each side to consider.

    Returns:
        Array with NaN rows replaced by temporal median.
    """
    filled = values.copy()
    nan_mask = np.isnan(filled).any(axis=1)
    if not nan_mask.any():
        return filled

    n = len(filled)
    for i in np.where(nan_mask)[0]:
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        window = filled[lo:hi]
        valid = window[~np.isnan(window).any(axis=1)]
        if len(valid) > 0:
            filled[i] = np.median(valid, axis=0)

    return filled


def anchor_project(
    poses_3d: np.ndarray,
    poses_2d_norm: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Project root-relative 3D poses to 2D anchored at detected 2D hip.

    For each frame, computes a per-frame scale by matching the 2D torso length
    (hip-center to mid-shoulder in pixels) against the 3D torso length (in
    meters).  Only the x, y components of 3D offsets are used (z is ignored),
    which is the orthographic approximation.

    Args:
        poses_3d: Root-relative 3D poses (N, 17, 3) float32, in meters.
        poses_2d_norm: 2D normalised poses (N, 17, 2) float32, coords in [0, 1].
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Corrected normalised poses (N, 17, 2) float32, coords in [0, 1].
    """
    n = poses_3d.shape[0]
    projected_norm = np.zeros((n, 17, 2), dtype=np.float32)

    # Pre-compute 2D reference points (pixels) with NaN fallback
    lhip_2d = poses_2d_norm[:, H36Key.LHIP, :]  # (N, 2)
    rhip_2d = poses_2d_norm[:, H36Key.RHIP, :]  # (N, 2)
    lshoulder_2d = poses_2d_norm[:, H36Key.LSHOULDER, :]  # (N, 2)
    rshoulder_2d = poses_2d_norm[:, H36Key.RSHOULDER, :]  # (N, 2)

    hip_2d_norm = (lhip_2d + rhip_2d) / 2.0  # (N, 2)
    shoulder_2d_norm = (lshoulder_2d + rshoulder_2d) / 2.0  # (N, 2)

    # Fill NaN in reference points with temporal median
    hip_2d_norm = _median_fill_nan(hip_2d_norm)
    shoulder_2d_norm = _median_fill_nan(shoulder_2d_norm)

    # Convert to pixels
    hip_2d_px = hip_2d_norm * np.array([width, height], dtype=np.float32)
    shoulder_2d_px = shoulder_2d_norm * np.array([width, height], dtype=np.float32)

    # 2D torso length per frame (pixels)
    torso_diff_2d = shoulder_2d_px - hip_2d_px  # (N, 2)
    torso_2d_px = np.linalg.norm(torso_diff_2d, axis=1)  # (N,)

    # 3D torso length per frame (meters)
    hip_center_3d = poses_3d[:, H36Key.HIP_CENTER, :]  # (N, 3)
    mid_shoulder_3d = (poses_3d[:, H36Key.LSHOULDER, :] + poses_3d[:, H36Key.RSHOULDER, :]) / 2.0
    torso_diff_3d = mid_shoulder_3d - hip_center_3d  # (N, 3)
    torso_3d_m = np.linalg.norm(torso_diff_3d, axis=1)  # (N,)

    # Per-frame scale: pixels per meter
    # Fallback for degenerate 3D (torso < 0.01m): use global scale
    global_scale = min(width, height) / 1.7  # assume ~1.7m person fills smaller dim
    scale = np.where(
        torso_3d_m > 0.01,
        torso_2d_px / (torso_3d_m + 1e-6),
        global_scale,
    ).astype(np.float32)

    for i in range(n):
        offsets = poses_3d[i] - hip_center_3d[i]  # (17, 3)
        # Orthographic: only x, y components
        offsets_xy = offsets[:, :2]  # (17, 2)

        projected_px = hip_2d_px[i] + offsets_xy * scale[i]  # (17, 2)

        # Normalise to [0, 1]
        projected_norm[i, :, 0] = projected_px[:, 0] / width
        projected_norm[i, :, 1] = projected_px[:, 1] / height

    # Clip to [0, 1]
    np.clip(projected_norm, 0.0, 1.0, out=projected_norm)

    return projected_norm


def blend_by_confidence(
    poses_raw: np.ndarray,
    poses_corrected: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Blend raw 2D and corrected 2D poses by per-joint confidence.

    High-confidence joints (> threshold + 0.2) keep the raw 2D estimate.
    Low-confidence joints (< threshold - 0.2) use the corrected 2D estimate.
    A soft linear transition is applied in between.

    Args:
        poses_raw: Raw 2D poses (N, 17, 2) float32.
        poses_corrected: Corrected 2D poses (N, 17, 2) float32.
        confidences: Per-joint confidence from 2D detector (N, 17) float32.
        threshold: Confidence midpoint for the blending sigmoid.

    Returns:
        Blended poses (N, 17, 2) float32.
    """
    # weight_3d = 1.0 when confidence is low (use corrected),
    # weight_3d = 0.0 when confidence is high (use raw).
    weight_3d = 1.0 - np.clip((confidences - threshold + 0.2) / 0.4, 0.0, 1.0)  # (N, 17)

    # Broadcast to (N, 17, 2)
    w = weight_3d[:, :, np.newaxis]

    blended: np.ndarray = ((1.0 - w) * poses_raw + w * poses_corrected).astype(np.float32)
    return blended
