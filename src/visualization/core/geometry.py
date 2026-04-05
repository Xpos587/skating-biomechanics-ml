"""Coordinate transformation utilities for visualization.

Provides functions for:
- Normalized to pixel coordinate conversion
- Pixel to normalized coordinate conversion
- 3D to 2D projection
- Spatial axis transformations
"""

import numpy as np
from numpy.typing import NDArray

from src.visualization.config import (
    camera_distance as default_camera_distance,
)
from src.visualization.config import (
    focal_length as default_focal_length,
)

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]  # OpenCV image (H, W, 3)
Position2D = tuple[float, float]  # (x, y) in normalized or pixel coords
Position3D = tuple[float, float, float]  # (x, y, z) in meters

# Normalized pose: (N, 17, 2) with coordinates in [0, 1]
NormalizedPose2D = NDArray[np.float32]

# Pixel pose: (N, 17, 2) with pixel coordinates
PixelPose2D = NDArray[np.float32]

# 3D pose: (N, 17, 3) with coordinates in meters
Pose3D = NDArray[np.float32]


# =============================================================================
# COORDINATE TRANSFORMATION
# =============================================================================


def normalized_to_pixel(
    pos_normalized: Position2D | NDArray[np.float32],
    width: int,
    height: int,
) -> Position2D | NDArray[np.int32]:
    """Convert normalized coordinates to pixel coordinates.

    Args:
        pos_normalized: (x, y) in [0, 1] or array of shape (..., 2).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        (x, y) in pixel coordinates or array of pixel coordinates.

    Example:
        >>> normalized_to_pixel((0.5, 0.5), 1920, 1080)
        (960, 540)  # Center of frame
    """
    if isinstance(pos_normalized, np.ndarray):
        # Vectorized conversion for arrays
        result = pos_normalized.copy()
        if result.shape[-1] >= 2:
            result[..., 0] = np.clip(result[..., 0] * width, 0, width - 1)
            result[..., 1] = np.clip(result[..., 1] * height, 0, height - 1)
            # Replace NaN (from undetected keypoints) with 0 before int cast
            nan_mask = np.isnan(result[..., 0]) | np.isnan(result[..., 1])
            if result.shape[-1] > 2:
                nan_mask = (
                    np.isnan(result[..., 0]) | np.isnan(result[..., 1]) | np.isnan(result[..., 2])
                )
            result[nan_mask] = 0
        return result.astype(np.int32)
    else:
        # Single position
        x, y = pos_normalized
        x_px = int(np.clip(x * width, 0, width - 1))
        y_px = int(np.clip(y * height, 0, height - 1))
        return (x_px, y_px)


def pixel_to_normalized(
    pos_pixel: Position2D | NDArray[np.float32],
    width: int,
    height: int,
) -> Position2D | NDArray[np.float32]:
    """Convert pixel coordinates to normalized coordinates.

    Args:
        pos_pixel: (x, y) in pixels or array of shape (..., 2).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        (x, y) in [0, 1] or array of normalized coordinates.

    Example:
        >>> pixel_to_normalized((960, 540), 1920, 1080)
        (0.5, 0.5)  # Center of frame
    """
    if isinstance(pos_pixel, np.ndarray):
        # Vectorized conversion for arrays
        result = pos_pixel.copy().astype(np.float32)
        if result.shape[-1] >= 2:
            result[..., 0] = result[..., 0] / width if width > 0 else 0.5
            result[..., 1] = result[..., 1] / height if height > 0 else 0.5
        return result
    else:
        # Single position
        x, y = pos_pixel
        x_norm = x / width if width > 0 else 0.5
        y_norm = y / height if height > 0 else 0.5
        return (x_norm, y_norm)


# =============================================================================
# 3D PROJECTION
# =============================================================================


def project_3d_to_2d(
    pos_3d: Position3D | NDArray[np.float32],
    width: int,
    height: int,
    focal_length: int = default_focal_length,
    camera_distance: float = default_camera_distance,
) -> Position2D | NDArray[np.int32]:
    """Project 3D coordinates to 2D using perspective projection.

    Args:
        pos_3d: (x, y, z) in meters or array of shape (..., 3).
            Origin is at hip center, +X right, +Y down, +Z forward.
        width: Frame width in pixels.
        height: Frame height in pixels.
        focal_length: Camera focal length in pixels.
        camera_distance: Camera distance from origin in meters.

    Returns:
        (x, y) in pixel coordinates or array of pixel coordinates.

    Note:
        This uses a simple pinhole camera model. For more accurate
        projection, use actual camera intrinsics from calibration.

    Example:
        >>> project_3d_to_2d((0.5, 0.3, 1.0), 1920, 1080)
        (1280, 540)  # Projected to pixel coordinates
    """
    is_array = isinstance(pos_3d, np.ndarray)

    if is_array:
        # Vectorized projection for arrays
        pos_3d = np.asarray(pos_3d)
        original_shape = pos_3d.shape

        # Flatten to (N, 3)
        if pos_3d.ndim == 1:
            pos_3d = pos_3d.reshape(1, 3)
        elif pos_3d.ndim == 2:
            pass  # Already (N, 3)
        else:
            # Flatten all but last dimension
            pos_3d = pos_3d.reshape(-1, 3)

        x = pos_3d[:, 0]
        y = pos_3d[:, 1]
        z = pos_3d[:, 2]

        # Perspective projection
        # Z-depth: camera_distance - z (positive = in front of camera)
        depth = camera_distance - z

        # Avoid division by zero
        depth = np.where(depth <= 0.1, 0.1, depth)

        # Project to 2D
        scale = focal_length / depth
        x_2d = width // 2 + x * scale
        y_2d = height // 2 + y * scale

        # Stack results
        result = np.nan_to_num(np.stack([x_2d, y_2d], axis=-1), nan=0).astype(np.int32)

        # Reshape to original shape (with 2 instead of 3)
        new_shape = (*original_shape[:-1], 2)
        result = result.reshape(new_shape)

        return result
    else:
        # Single position
        x, y, z = pos_3d

        # Perspective projection
        depth = camera_distance - z

        # Avoid division by zero
        depth = max(0.1, depth)

        scale = focal_length / depth
        x_2d = width // 2 + int(x * scale)
        y_2d = height // 2 + int(y * scale)

        return (x_2d, y_2d)


def project_3d_to_normalized(
    pos_3d: Position3D | NDArray[np.float32],
    focal_length: int = default_focal_length,
    camera_distance: float = default_camera_distance,
) -> Position2D | NDArray[np.float32]:
    """Project 3D coordinates to normalized 2D coordinates.

    Args:
        pos_3d: (x, y, z) in meters or array of shape (..., 3).
        focal_length: Camera focal length in pixels.
        camera_distance: Camera distance from origin in meters.

    Returns:
        (x, y) in [0, 1] or array of normalized coordinates.

    Example:
        >>> project_3d_to_normalized((0.5, 0.3, 1.0))
        (0.75, 0.55)  # Normalized coordinates
    """
    is_array = isinstance(pos_3d, np.ndarray)

    if is_array:
        # Project to pixel coordinates first
        # Assuming standard 1920x1080 for normalization
        pixel_coords = project_3d_to_2d(pos_3d, 1920, 1080, focal_length, camera_distance)
        # Convert to normalized
        return pixel_to_normalized(pixel_coords, 1920, 1080)  # type: ignore[arg-type]
    else:
        x, y = pos_3d[0], pos_3d[1]  # type: ignore[assignment]

        # Perspective projection
        depth = camera_distance - pos_3d[2]

        # Avoid division by zero
        depth = max(0.1, depth)

        scale = focal_length / depth

        x_norm = 0.5 + x * scale / 1920
        y_norm = 0.5 + y * scale / 1080

        return (x_norm, y_norm)


# =============================================================================
# SPATIAL AXIS TRANSFORMATIONS
# =============================================================================


def get_axis_endpoints(
    center: Position2D,
    length: int = 50,
    origin_3d: Position3D | None = None,
) -> dict[str, tuple[Position2D, Position2D]]:
    """Get start and end points for spatial axes.

    Args:
        center: (x, y) center position for axes.
        length: Length of axis arrows in pixels.
        origin_3d: Optional (x, y, z) 3D origin for perspective.

    Returns:
        Dict with 'x', 'y', 'z' keys, each containing (start, end) positions.

    Example:
        >>> endpoints = get_axis_endpoints((100, 100), 50)
        >>> endpoints['x']
        ((100, 100), (150, 100))  # X-axis arrow
    """
    x_center, y_center = center

    axes = {
        "x": ((x_center, y_center), (x_center + length, y_center)),
        "y": ((x_center, y_center), (x_center, y_center + length)),
        "z": ((x_center, y_center), (x_center, y_center - length)),
    }

    return axes


def get_axis_endpoints_3d(
    origin: Position3D,
    length: float = 0.5,
    camera_distance: float = default_camera_distance,
) -> dict[str, tuple[Position3D, Position3D]]:
    """Get 3D endpoints for spatial axes.

    Args:
        origin: (x, y, z) origin position in meters.
        length: Length of axis arrows in meters.
        camera_distance: Camera distance for perspective.

    Returns:
        Dict with 'x', 'y', 'z' keys, each containing (start, end) 3D positions.

    Example:
        >>> endpoints = get_axis_endpoints_3d((0, 0, 0), 0.5)
        >>> endpoints['x']
        ((0, 0, 0), (0.5, 0, 0))  # X-axis arrow
    """
    x_orig, y_orig, z_orig = origin

    axes = {
        "x": (origin, (x_orig + length, y_orig, z_orig)),
        "y": (origin, (x_orig, y_orig + length, z_orig)),
        "z": (origin, (x_orig, y_orig, z_orig + length)),
    }

    return axes


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def clip_to_frame(
    position: Position2D,
    width: int,
    height: int,
    margin: int = 0,
) -> Position2D:
    """Clip position to frame boundaries.

    Args:
        position: (x, y) position in pixels.
        width: Frame width.
        height: Frame height.
        margin: Margin from edge.

    Returns:
        Clipped (x, y) position.

    Example:
        >>> clip_to_frame((-10, 2000), 1920, 1080, 10)
        (10, 1070)  # Clipped to frame with margin
    """
    x, y = position
    x_clipped = max(margin, min(width - margin, x))
    y_clipped = max(margin, min(height - margin, y))
    return (x_clipped, y_clipped)


def calculate_bounding_box(
    points: NDArray[np.float32],
    width: int,
    height: int,
    padding: int = 10,
) -> tuple[int, int, int, int]:
    """Calculate bounding box for a set of points.

    Args:
        points: Array of shape (N, 2) with pixel coordinates.
        width: Frame width (for clipping).
        height: Frame height (for clipping).
        padding: Padding around bounding box.

    Returns:
        (x_min, y_min, x_max, y_max) bounding box.

    Example:
        >>> points = np.array([[100, 100], [200, 200], [150, 150]])
        >>> calculate_bounding_box(points, 1920, 1080, 10)
        (90, 90, 210, 210)  # Bounding box with padding
    """
    if len(points) == 0:
        return (padding, padding, width - padding, height - padding)

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min = max(0, int(np.min(x_coords)) - padding)
    y_min = max(0, int(np.min(y_coords)) - padding)
    x_max = min(width, int(np.max(x_coords)) + padding)
    y_max = min(height, int(np.max(y_coords)) + padding)

    return (x_min, y_min, x_max, y_max)


def calculate_center_of_bbox(
    bbox: tuple[int, int, int, int],
) -> Position2D:
    """Calculate center of bounding box.

    Args:
        bbox: (x_min, y_min, x_max, y_max) bounding box.

    Returns:
        (x_center, y_center) position.

    Example:
        >>> calculate_center_of_bbox((100, 100, 200, 200))
        (150, 150)  # Center of bounding box
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    return (x_center, y_center)


def normalize_to_aspect_ratio(
    pos_normalized: Position2D,
    target_aspect: float,
    current_aspect: float,
) -> Position2D:
    """Adjust normalized position for aspect ratio change.

    Args:
        pos_normalized: (x, y) in [0, 1].
        target_aspect: Target aspect ratio (width/height).
        current_aspect: Current aspect ratio (width/height).

    Returns:
        Adjusted (x, y) position.

    Example:
        >>> normalize_to_aspect_ratio((0.5, 0.5), 16/9, 4/3)
        (0.56, 0.5)  # Adjusted for aspect ratio
    """
    x, y = pos_normalized

    if current_aspect > target_aspect:
        # Crop width
        scale = target_aspect / current_aspect
        x = 0.5 + (x - 0.5) * scale
    else:
        # Crop height
        scale = current_aspect / target_aspect
        y = 0.5 + (y - 0.5) * scale

    return (x, y)
