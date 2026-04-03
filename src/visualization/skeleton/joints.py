"""Joint styling for skeleton visualization.

Provides functions for:
- Joint color assignment (left/right/center)
- Joint radius based on confidence
- Skeleton line colors
"""

from typing import Final

from src.types import H36Key
from src.visualization.config import (
    color_center,
    color_joint,
    color_left_side,
    color_right_side,
    confidence_threshold,
    joint_radius,
)

# =============================================================================
# JOINT GROUPS
# =============================================================================

# Left side joints (arm + leg)
LEFT_JOINTS: Final[frozenset[int]] = frozenset(
    [
        H36Key.LHIP,
        H36Key.LKNEE,
        H36Key.LFOOT,
        H36Key.LSHOULDER,
        H36Key.LELBOW,
        H36Key.LWRIST,
    ]
)

# Right side joints (arm + leg)
RIGHT_JOINTS: Final[frozenset[int]] = frozenset(
    [
        H36Key.RHIP,
        H36Key.RKNEE,
        H36Key.RFOOT,
        H36Key.RSHOULDER,
        H36Key.RELBOW,
        H36Key.RWRIST,
    ]
)

# Center joints (torso + head)
CENTER_JOINTS: Final[frozenset[int]] = frozenset(
    [
        H36Key.HIP_CENTER,
        H36Key.SPINE,
        H36Key.THORAX,
        H36Key.NECK,
        H36Key.HEAD,
    ]
)


# =============================================================================
# JOINT COLORS
# =============================================================================


def get_skeleton_color(
    joint_idx: int,
    left_color: tuple[int, int, int] = color_left_side,
    right_color: tuple[int, int, int] = color_right_side,
    center_color: tuple[int, int, int] = color_center,
) -> tuple[int, int, int]:
    """Get color for skeleton joint based on side.

    Args:
        joint_idx: H36Key joint index.
        left_color: Color for left side joints.
        right_color: Color for right side joints.
        center_color: Color for center joints.

    Returns:
        BGR color tuple.

    Example:
        >>> get_skeleton_color(H36Key.LSHOULDER)
        (255, 0, 0)  # Blue (default left color)
    """
    if joint_idx in LEFT_JOINTS:
        return left_color
    elif joint_idx in RIGHT_JOINTS:
        return right_color
    else:
        return center_color


def get_joint_color(
    joint_idx: int,
    joint_color: tuple[int, int, int] = color_joint,
    **kwargs,
) -> tuple[int, int, int]:
    """Get color for joint circle.

    Args:
        joint_idx: H36Key joint index.
        joint_color: Default joint color.
        **kwargs: Additional arguments (for compatibility with get_skeleton_color).

    Returns:
        BGR color tuple.

    Example:
        >>> get_joint_color(H36Key.LSHOULDER)
        (255, 255, 255)  # White (default joint color)
    """
    return joint_color


# =============================================================================
# JOINT SIZES
# =============================================================================


def get_joint_radius(
    confidence: float = 1.0,
    base_radius: int = joint_radius,
    min_radius: int = 2,
    max_radius: int = 8,
    threshold: float = confidence_threshold,
) -> int:
    """Calculate joint radius based on confidence.

    Args:
        confidence: Joint confidence value [0, 1].
        base_radius: Base radius at full confidence.
        min_radius: Minimum radius (for low confidence).
        max_radius: Maximum radius.
        threshold: Confidence threshold for visibility.

    Returns:
        Joint radius in pixels.

    Example:
        >>> get_joint_radius(0.8)
        4  # Full radius (confidence above threshold)
    """
    if confidence < threshold:
        return 0  # Don't draw low-confidence joints

    # Scale radius by confidence
    radius = int(base_radius * confidence)

    # Clamp to range
    return max(min_radius, min(max_radius, radius))


def get_joint_radius_3d(
    depth: float,
    base_radius: int = joint_radius,
    depth_min: float = 0.0,
    depth_max: float = 2.0,
) -> int:
    """Calculate joint radius based on depth (3D visualization).

    Args:
        depth: Depth value in meters.
        base_radius: Base radius at depth_min.
        depth_min: Minimum depth (closest).
        depth_max: Maximum depth (farthest).

    Returns:
        Joint radius in pixels.

    Example:
        >>> get_joint_radius_3d(1.0)
        4  # Medium depth, medium radius
    """
    # Normalize depth to [0, 1]
    t = (depth - depth_min) / (depth_max - depth_min) if depth_max > depth_min else 0.5
    t = max(0.0, min(1.0, t))

    # Scale radius: closer = larger
    scale = 1.0 - 0.5 * t  # 1.0 at depth_min, 0.5 at depth_max
    return max(2, int(base_radius * scale))


# =============================================================================
# CONFIDENCE VISUALIZATION
# =============================================================================


def get_confidence_color(
    confidence: float,
    low_color: tuple[int, int, int] = (0, 0, 128),
    high_color: tuple[int, int, int] = (0, 255, 0),
) -> tuple[int, int, int]:
    """Get color representing confidence level.

    Args:
        confidence: Confidence value [0, 1].
        low_color: Color for low confidence (dark blue).
        high_color: Color for high confidence (green).

    Returns:
        BGR color tuple interpolated based on confidence.

    Example:
        >>> get_confidence_color(0.5)
        (0, 128, 64)  # Medium confidence (blue-green)
    """
    t = max(0.0, min(1.0, confidence))

    r = int(low_color[0] * (1 - t) + high_color[0] * t)
    g = int(low_color[1] * (1 - t) + high_color[1] * t)
    b = int(low_color[2] * (1 - t) + high_color[2] * t)

    return (b, g, r)


def get_confidence_color_rdygn(
    confidence: float,
) -> tuple[int, int, int]:
    """Get color representing confidence level using RdYlGn colormap.

    Maps confidence [0, 1] to Red (low) -> Yellow (mid) -> Green (high).
    Uses manual interpolation (Sports2D approach).

    Args:
        confidence: Confidence value [0, 1].

    Returns:
        BGR color tuple.

    Example:
        >>> get_confidence_color_rdygn(0.9)
        (0, 230, 0)  # Green-ish
    """
    t = max(0.0, min(1.0, confidence))

    # RdYlGn: Red (0, 0, 255) -> Yellow (0, 255, 255) -> Green (0, 255, 0)
    # Interpolate based on t
    if t < 0.5:
        # Red to Yellow
        local_t = t * 2  # [0, 1]
        b = 0
        g = int(255 * local_t)
        r = 255
    else:
        # Yellow to Green
        local_t = (t - 0.5) * 2  # [0, 1]
        b = 0
        g = 255
        r = int(255 * (1 - local_t))

    return (b, g, r)


def get_confidence_radius(
    confidence: float,
    base_radius: int = joint_radius,
    min_radius: int = 2,
    max_radius: int = 10,
) -> int:
    """Get radius scaled by confidence.

    Args:
        confidence: Confidence value [0, 1].
        base_radius: Base radius at confidence=1.0.
        min_radius: Minimum radius.
        max_radius: Maximum radius.

    Returns:
        Joint radius in pixels.

    Example:
        >>> get_confidence_radius(0.7)
        5  # Slightly smaller than base_radius
    """
    radius = int(base_radius * confidence)
    return max(min_radius, min(max_radius, radius))


# =============================================================================
# BONE THICKNESS
# =============================================================================


def get_bone_thickness(
    joint_idx: int,
    base_thickness: int = 2,
    center_thickness: int = 3,
) -> int:
    """Get bone thickness based on joint type.

    Args:
        joint_idx: H36Key joint index.
        base_thickness: Default thickness.
        center_thickness: Thickness for center/torso bones.

    Returns:
        Bone thickness in pixels.

    Example:
        >>> get_bone_thickness(H36Key.SPINE)
        3  # Thicker for center bones
    """
    if joint_idx in CENTER_JOINTS:
        return center_thickness
    else:
        return base_thickness


def get_bone_thickness_3d(
    depth: float,
    base_thickness: int = 2,
    depth_min: float = 0.0,
    depth_max: float = 2.0,
) -> int:
    """Get bone thickness based on depth (3D visualization).

    Args:
        depth: Depth value in meters.
        base_thickness: Base thickness at depth_min.
        depth_min: Minimum depth (closest).
        depth_max: Maximum depth (farthest).

    Returns:
        Bone thickness in pixels.

    Example:
        >>> get_bone_thickness_3d(1.5)
        1  # Thinner for far bones
    """
    # Normalize depth to [0, 1]
    t = (depth - depth_min) / (depth_max - depth_min) if depth_max > depth_min else 0.5
    t = max(0.0, min(1.0, t))

    # Scale thickness: closer = thicker
    scale = 1.0 - 0.5 * t  # 1.0 at depth_min, 0.5 at depth_max
    return max(1, int(base_thickness * scale))
