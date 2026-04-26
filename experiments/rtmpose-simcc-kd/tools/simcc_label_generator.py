"""Generate Gaussian SimCC soft labels from teacher coordinates."""

from __future__ import annotations

import numpy as np


def generate_simcc_label(
    coords: np.ndarray,
    confidence: np.ndarray,
    num_x_bins: int = 192,
    num_y_bins: int = 256,
    base_sigma: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized coordinates [0,1] to Gaussian SimCC soft labels.

    Args:
        coords: (K, 2) float32 — (x, y) in [0, 1]. K=17 for COCO.
        confidence: (K,) float32 — teacher confidence per keypoint (0-1).
        num_x_bins: Number of x-axis bins (e.g., 192 for 256x192 input).
        num_y_bins: Number of y-axis bins (e.g., 256 for 256x192 input).
        base_sigma: Base Gaussian sigma in normalized [0,1] space.

    Returns:
        simcc_x: (K, num_x_bins) float32 — x-axis soft labels.
        simcc_y: (K, num_y_bins) float32 — y-axis soft labels.
    """
    K = coords.shape[0]

    x_bins = np.linspace(0, 1, num_x_bins, dtype=np.float32)
    y_bins = np.linspace(0, 1, num_y_bins, dtype=np.float32)

    sigma = np.clip(base_sigma * (1.0 - confidence), 0.005, base_sigma)

    simcc_x = np.zeros((K, num_x_bins), dtype=np.float32)
    simcc_y = np.zeros((K, num_y_bins), dtype=np.float32)

    for k in range(K):
        x, y = coords[k]
        s = sigma[k]

        gx = np.exp(-0.5 * ((x_bins - x) / s) ** 2)
        gy = np.exp(-0.5 * ((y_bins - y) / s) ** 2)

        simcc_x[k] = gx / gx.sum() if gx.sum() > 0 else gx
        simcc_y[k] = gy / gy.sum() if gy.sum() > 0 else gy

    return simcc_x, simcc_y


def batch_generate_simcc(
    coords: np.ndarray,
    confidence: np.ndarray,
    num_x_bins: int = 192,
    num_y_bins: int = 256,
    base_sigma: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized batch version.

    Args:
        coords: (B, K, 2) float32.
        confidence: (B, K) float32.

    Returns:
        simcc_x: (B, K, num_x_bins) float32.
        simcc_y: (B, K, num_y_bins) float32.
    """
    B, K = coords.shape[:2]
    x_bins = np.linspace(0, 1, num_x_bins, dtype=np.float32).reshape(1, 1, -1)
    y_bins = np.linspace(0, 1, num_y_bins, dtype=np.float32).reshape(1, 1, -1)

    sigma = np.clip(base_sigma * (1.0 - confidence), 0.005, base_sigma)[..., np.newaxis]

    xs = coords[..., 0:1]
    ys = coords[..., 1:2]

    gx = np.exp(-0.5 * ((x_bins - xs) / sigma) ** 2)
    gy = np.exp(-0.5 * ((y_bins - ys) / sigma) ** 2)

    gx_sum = gx.sum(axis=-1, keepdims=True)
    gy_sum = gy.sum(axis=-1, keepdims=True)
    gx = np.where(gx_sum > 0, gx / gx_sum, gx)
    gy = np.where(gy_sum > 0, gy / gy_sum, gy)

    return gx.astype(np.float32), gy.astype(np.float32)


def transform_teacher_coords(
    crop_coords: np.ndarray,
    crop_params: np.ndarray,
) -> np.ndarray:
    """Transform teacher coords from crop [0,1] to image [0,1].

    Args:
        crop_coords: (K, 2) — normalized crop space.
        crop_params: (6,) — (x1, y1, crop_w, crop_h, img_w, img_h).

    Returns:
        image_coords: (K, 2) — normalized image space.
    """
    x1, y1, cw, ch, img_w, img_h = crop_params

    crop_w_teacher = 384.0
    crop_h_teacher = 288.0

    px = crop_coords[:, 0] * crop_w_teacher
    py = crop_coords[:, 1] * crop_h_teacher

    sx = cw / crop_w_teacher
    sy = ch / crop_h_teacher

    global_x = x1 + px * sx
    global_y = y1 + py * sy

    return np.stack([global_x / img_w, global_y / img_h], axis=-1).astype(np.float32)
