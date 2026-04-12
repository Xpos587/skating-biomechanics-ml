"""Skeleton drawer for H3.6M 17-keypoint format.

Provides functions for drawing:
- 2D skeleton overlay (normalized or pixel coordinates)
- 3D skeleton with depth color coding
- 3D skeleton in picture-in-picture mode
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from skating_ml.types import H36M_SKELETON_EDGES
from skating_ml.visualization.config import (
    joint_radius,
    line_width,
)
from skating_ml.visualization.core.colors import get_depth_color
from skating_ml.visualization.core.geometry import (
    normalized_to_pixel,
    project_3d_to_2d,
)
from skating_ml.visualization.skeleton.joints import (
    get_bone_thickness_3d,
    get_joint_color,
    get_joint_radius,
    get_joint_radius_3d,
)

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]  # OpenCV image (H, W, 3)
Pose2D = NDArray[np.float32]  # (17, 2) normalized or pixel coords
Pose3D = NDArray[np.float32]  # (17, 3) in meters


# =============================================================================
# 2D SKELETON DRAWING
# =============================================================================


def draw_skeleton(
    frame: Frame,
    pose: Pose2D,
    height: int,
    width: int,
    confidence_threshold: float = 0.5,
    line_width: int = 2,
    joint_radius: int = 4,
    normalized: bool | None = None,
    confidences: np.ndarray | None = None,
    foot_keypoints: np.ndarray | None = None,
) -> Frame:
    """Draw 2D skeleton overlay on frame.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        pose: Pose array (17, 2) normalized [0,1] or (17, 3) pixel coordinates.
            If (17, 3), third column is confidence.
        height: Frame height for coordinate conversion.
        width: Frame width for coordinate conversion.
        confidence_threshold: Skip keypoints below this confidence (if 3rd dim exists).
        line_width: Width of skeleton connection lines.
        joint_radius: Radius of joint circles.

    Returns:
        Frame with skeleton drawn (modified in place).

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> pose = np.random.rand(17, 2).astype(np.float32) * 0.5 + 0.25
        >>> draw_skeleton(frame, pose, 480, 640)
    """
    # Handle backward compatibility: (17, 3) poses where column 2 is confidence
    confidences = None
    if pose.shape[1] == 3:
        confidences = pose[:, 2]
        pose = pose[:, :2]

    # Check if coordinates are normalized or pixel format
    # If all values are in [0, 1], assume normalized
    if pose.max() <= 1.0:
        # Convert normalized to pixel
        pose_px = normalized_to_pixel(pose, width, height)
    else:
        # Already in pixel coordinates
        pose_px = np.nan_to_num(pose.round(), nan=0).astype(np.int32)

    # Draw skeleton edges (bones)
    for joint_a, joint_b in H36M_SKELETON_EDGES:
        # Check if joints are valid
        if confidences is not None:
            if confidences[joint_a] < confidence_threshold:
                continue
            if confidences[joint_b] < confidence_threshold:
                continue

        pt_a = tuple(np.asarray(pose_px[joint_a]).round().astype(int))
        pt_b = tuple(np.asarray(pose_px[joint_b]).round().astype(int))

        # Check if points are within frame
        if not (_is_valid_point(pt_a, width, height) and _is_valid_point(pt_b, width, height)):
            continue

        # Clean single-color bones (light gray)
        color = (200, 200, 200)

        cv2.line(frame, pt_a, pt_b, color, line_width, cv2.LINE_AA)

    # Draw joints
    for joint_idx in range(len(pose_px)):
        # Check confidence
        if confidences is not None and confidences[joint_idx] < confidence_threshold:
            continue

        pt = tuple(np.asarray(pose_px[joint_idx]).round().astype(int))

        # Check if point is within frame
        if not _is_valid_point(pt, width, height):
            continue

        # Get joint color based on confidence
        conf = 1.0 if confidences is None else confidences[joint_idx]
        if confidences is not None and conf >= confidence_threshold:
            from skating_ml.visualization.skeleton.joints import get_confidence_color_rdygn

            color = get_confidence_color_rdygn(conf)
        else:
            color = get_joint_color(joint_idx)
        radius = get_joint_radius(
            conf,
            joint_radius,
            threshold=confidence_threshold,
        )

        if radius > 0:
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

    # Draw foot keypoints (HALPE26: heel + big toe + small toe)
    if foot_keypoints is not None:
        _draw_foot_keypoints(frame, foot_keypoints, width, height, confidence_threshold)

    return frame


def draw_skeleton_batch(
    frames: list[Frame],
    poses: NDArray[np.float32],
    width: int,
    height: int,
    normalized: bool = True,
    **kwargs,
) -> list[Frame]:
    """Draw skeleton on multiple frames (batch processing).

    Args:
        frames: List of OpenCV images.
        poses: Pose array (N, 17, 2) with N = len(frames).
        width: Frame width in pixels.
        height: Frame height in pixels.
        normalized: Whether pose coordinates are normalized [0, 1].
        **kwargs: Additional arguments for draw_skeleton().

    Returns:
        List of frames with skeleton drawn.

    Example:
        >>> frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
        >>> poses = np.random.rand(10, 17, 2).astype(np.float32) * 0.5 + 0.25
        >>> result = draw_skeleton_batch(frames, poses, 640, 480)
    """
    result = []

    for i, frame in enumerate(frames):
        frame_draw = draw_skeleton(
            frame.copy(),
            poses[i],
            width,
            height,
            normalized,
            **kwargs,
        )
        result.append(frame_draw)

    return result


# =============================================================================
# 3D SKELETON DRAWING
# =============================================================================


def draw_skeleton_3d(
    frame: Frame,
    pose_3d: Pose3D,
    width: int,
    height: int,
    line_width: int = line_width,
    joint_radius: int = joint_radius,
    depth_min: float = 0.0,
    depth_max: float = 2.0,
    camera_distance: float = 3.0,
    focal_length: int = 800,
) -> Frame:
    """Draw 3D skeleton with depth color coding.

    Projects 3D pose to 2D using perspective projection and colors
    joints/bones based on depth (Z-coordinate).

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        pose_3d: Pose array (17, 3) in meters (x, y, z).
        width: Frame width in pixels.
        height: Frame height in pixels.
        line_width: Base width of skeleton lines.
        joint_radius: Base radius of joint circles.
        depth_min: Minimum depth for color mapping (meters).
        depth_max: Maximum depth for color mapping (meters).
        camera_distance: Camera distance from origin (meters).
        focal_length: Camera focal length (pixels).

    Returns:
        Frame with 3D skeleton drawn (modified in place).

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> pose_3d = np.random.randn(17, 3).astype(np.float32) * 0.5
        >>> draw_skeleton_3d(frame, pose_3d, 640, 480)
    """
    # Project 3D to 2D
    pose_2d = project_3d_to_2d(pose_3d, width, height, focal_length, camera_distance)

    # Extract depths
    depths = pose_3d[:, 2]

    # Draw skeleton edges
    for joint_a, joint_b in H36M_SKELETON_EDGES:
        pt_a = tuple(np.asarray(pose_2d[joint_a]).round().astype(int))
        pt_b = tuple(np.asarray(pose_2d[joint_b]).round().astype(int))

        # Check if points are within frame
        if not (_is_valid_point(pt_a, width, height) and _is_valid_point(pt_b, width, height)):
            continue

        # Get average depth for color
        avg_depth = (depths[joint_a] + depths[joint_b]) / 2
        color = get_depth_color(avg_depth, depth_min, depth_max)

        # Scale thickness by depth
        thickness = get_bone_thickness_3d(avg_depth, line_width, depth_min, depth_max)

        cv2.line(frame, pt_a, pt_b, color, thickness, cv2.LINE_AA)

    # Draw joints
    for joint_idx in range(len(pose_2d)):
        pt = tuple(np.asarray(pose_2d[joint_idx]).round().astype(int))

        # Check if point is within frame
        if not _is_valid_point(pt, width, height):
            continue

        # Get color and radius based on depth
        depth = depths[joint_idx]
        color = get_depth_color(depth, depth_min, depth_max)
        radius = get_joint_radius_3d(depth, joint_radius, depth_min, depth_max)

        if radius > 0:
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

    return frame


def draw_skeleton_3d_pip(
    frame: Frame,
    pose_3d: Pose3D,
    width: int,
    height: int,
    pip_size: tuple[int, int] = (320, 240),
    pip_position: str = "bottom_right",
    bg_color: tuple[int, int, int] = (0, 0, 0),
    line_width: int = line_width,
    joint_radius: int = joint_radius,
    depth_min: float = 0.0,
    depth_max: float = 2.0,
) -> Frame:
    """Draw 3D skeleton in picture-in-picture mode.

    .. deprecated:: 0.5
        Replaced by 3D-corrected 2D overlay via ``CorrectiveLens``.
        The PIP window has been removed from the visualization pipeline.
        Kept for backward compatibility only.

    Creates a small inset view showing the 3D skeleton from a side angle,
    useful for visualizing depth that's hard to see in the main view.

    Args:
        frame: Main OpenCV image (H, W, 3) BGR format.
        pose_3d: Pose array (17, 3) in meters.
        width: Main frame width (for reference).
        height: Main frame height (for reference).
        pip_size: Size of PIP window (width, height).
        pip_position: Position of PIP window ("top_left", "top_right",
            "bottom_left", "bottom_right").
        bg_color: Background color for PIP window.
        line_width: Base width of skeleton lines.
        joint_radius: Base radius of joint circles.
        depth_min: Minimum depth for color mapping.
        depth_max: Maximum depth for color mapping.

    Returns:
        Frame with PIP skeleton overlay added.

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> pose_3d = np.random.randn(17, 3).astype(np.float32) * 0.5
        >>> draw_skeleton_3d_pip(frame, pose_3d, 640, 480)
    """
    pip_width, pip_height = pip_size

    # Calculate PIP position
    if pip_position == "top_left":
        x_offset, y_offset = 10, 10
    elif pip_position == "top_right":
        x_offset, y_offset = width - pip_width - 10, 10
    elif pip_position == "bottom_left":
        x_offset, y_offset = 10, height - pip_height - 10
    else:  # bottom_right
        x_offset, y_offset = width - pip_width - 10, height - pip_height - 10

    # Create PIP frame
    pip_frame = np.full((pip_height, pip_width, 3), bg_color, dtype=np.uint8)

    # Center the pose in PIP frame
    # Extract X and Z coordinates (side view)
    pose_xz = pose_3d[:, [0, 2]]  # (17, 2)

    # Normalize to fit in PIP frame
    x_min, x_max = pose_xz[:, 0].min(), pose_xz[:, 0].max()
    z_min, z_max = pose_xz[:, 1].min(), pose_xz[:, 1].max()

    x_range = max(x_max - x_min, 0.5)
    z_range = max(z_max - z_min, 0.5)

    pose_xz[:, 0] = (pose_xz[:, 0] - x_min) / x_range * 0.8 + 0.1
    pose_xz[:, 1] = (pose_xz[:, 1] - z_min) / z_range * 0.8 + 0.1

    # Convert to pixel coordinates
    pose_pip = normalized_to_pixel(pose_xz, pip_width, pip_height)

    # Draw skeleton edges in PIP
    for joint_a, joint_b in H36M_SKELETON_EDGES:
        pt_a = tuple(np.asarray(pose_pip[joint_a]).round().astype(int))
        pt_b = tuple(np.asarray(pose_pip[joint_b]).round().astype(int))

        # Get color based on depth (Z coordinate)
        avg_depth = (pose_3d[joint_a, 2] + pose_3d[joint_b, 2]) / 2
        color = get_depth_color(avg_depth, depth_min, depth_max)

        cv2.line(pip_frame, pt_a, pt_b, color, line_width, cv2.LINE_AA)

    # Draw joints in PIP
    for joint_idx in range(len(pose_pip)):
        pt = tuple(np.asarray(pose_pip[joint_idx]).round().astype(int))

        depth = pose_3d[joint_idx, 2]
        color = get_depth_color(depth, depth_min, depth_max)
        radius = max(2, joint_radius // 2)

        cv2.circle(pip_frame, pt, radius, color, -1, cv2.LINE_AA)

    # Add border to PIP
    cv2.rectangle(pip_frame, (0, 0), (pip_width - 1, pip_height - 1), (255, 255, 255), 2)

    # Composite PIP onto main frame
    y_end = y_offset + pip_height
    x_end = x_offset + pip_width

    # Create mask for smooth blending
    mask = np.ones((pip_height, pip_width), dtype=np.float32) * 0.9
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # Blend
    roi = frame[y_offset:y_end, x_offset:x_end]
    for c in range(3):
        roi[:, :, c] = (roi[:, :, c] * (1 - mask) + pip_frame[:, :, c] * mask).astype(np.uint8)

    frame[y_offset:y_end, x_offset:x_end] = roi

    # Draw border
    cv2.rectangle(frame, (x_offset, y_offset), (x_end - 1, y_end - 1), (255, 255, 255), 2)

    return frame


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Sports2D color constants (BGR)
_SPORTS2D_LEFT_COLOR: tuple[int, int, int] = (0, 255, 0)  # Green
_SPORTS2D_RIGHT_COLOR: tuple[int, int, int] = (0, 128, 255)  # Orange
_SPORTS2D_CENTER_COLOR: tuple[int, int, int] = (255, 153, 51)  # Blue

# Indices into the (6, 3) foot keypoint array
_FOOT_HEEL_L = 0
_FOOT_BIG_TOE_L = 1
_FOOT_SMALL_TOE_L = 2
_FOOT_HEEL_R = 3
_FOOT_BIG_TOE_R = 4
_FOOT_SMALL_TOE_R = 5


def _draw_foot_keypoints(
    frame: Frame,
    foot_keypoints: np.ndarray,
    width: int,
    height: int,
    confidence_threshold: float = 0.5,
    kp_radius: int = 2,
    line_thickness: int = 1,
) -> None:
    """Draw HALPE26 foot keypoints and segments.

    Draws heels and big toes as circles (RdYlGn confidence color).
    Draws heel-to-toe connecting lines.
    Skips small toes (too noisy on ice skates).
    """
    from skating_ml.visualization.skeleton.joints import get_confidence_color_rdygn

    fk = foot_keypoints.copy()
    if fk[:, 0].max() <= 1.0 and fk[:, 1].max() <= 1.0:
        fk[:, 0] *= width
        fk[:, 1] *= height

    foot_pairs = [
        (_FOOT_HEEL_L, _FOOT_BIG_TOE_L),
        (_FOOT_HEEL_R, _FOOT_BIG_TOE_R),
    ]

    for heel_idx, toe_idx in foot_pairs:
        heel_conf = fk[heel_idx, 2]
        toe_conf = fk[toe_idx, 2]

        # Skip NaN keypoints
        if np.isnan(heel_conf) or np.isnan(toe_conf):
            continue

        heel_pt = (int(fk[heel_idx, 0]), int(fk[heel_idx, 1]))
        toe_pt = (int(fk[toe_idx, 0]), int(fk[toe_idx, 1]))

        if not (_is_valid_point(heel_pt, width, height) and _is_valid_point(toe_pt, width, height)):
            continue

        # Draw heel-to-toe segment line
        if heel_conf >= confidence_threshold and toe_conf >= confidence_threshold:
            line_color = _SPORTS2D_LEFT_COLOR if heel_idx == _FOOT_HEEL_L else _SPORTS2D_RIGHT_COLOR
            cv2.line(frame, heel_pt, toe_pt, line_color, line_thickness, cv2.LINE_AA)

        # Draw heel circle
        if heel_conf >= confidence_threshold:
            color = get_confidence_color_rdygn(heel_conf)
            cv2.circle(frame, heel_pt, kp_radius, color, -1, cv2.LINE_AA)

        # Draw big toe circle
        if toe_conf >= confidence_threshold:
            color = get_confidence_color_rdygn(toe_conf)
            cv2.circle(frame, toe_pt, kp_radius, color, -1, cv2.LINE_AA)


def _get_sports2d_bone_color(joint_a: int, joint_b: int) -> tuple[int, int, int]:
    """Return bone color based on Sports2D side convention.

    Sports2D uses:
    - Green for left-side bones
    - Orange for right-side bones
    - Blue for center/torso bones

    Args:
        joint_a: First joint index (H36Key).
        joint_b: Second joint index (H36Key).

    Returns:
        BGR color tuple for the bone.

    Example:
        >>> _get_sports2d_bone_color(H36Key.LHIP, H36Key.LKNEE)
        (0, 255, 0)  # Green
    """
    from skating_ml.types import H36Key

    left_joints = {
        H36Key.LHIP,
        H36Key.LKNEE,
        H36Key.LFOOT,
        H36Key.LSHOULDER,
        H36Key.LELBOW,
        H36Key.LWRIST,
    }
    right_joints = {
        H36Key.RHIP,
        H36Key.RKNEE,
        H36Key.RFOOT,
        H36Key.RSHOULDER,
        H36Key.RELBOW,
        H36Key.RWRIST,
    }

    a_left = joint_a in left_joints
    a_right = joint_a in right_joints
    b_left = joint_b in left_joints
    b_right = joint_b in right_joints

    # Both joints on left side
    if a_left and not a_right and b_left and not b_right:
        return _SPORTS2D_LEFT_COLOR
    # Both joints on right side
    if a_right and not a_left and b_right and not b_left:
        return _SPORTS2D_RIGHT_COLOR
    # Mixed or center bones
    return _SPORTS2D_CENTER_COLOR


def _is_valid_point(
    pt: tuple[int, int],
    width: int,
    height: int,
    margin: int = 50,
) -> bool:
    """Check if point is within valid frame region.

    Args:
        pt: (x, y) point in pixels.
        width: Frame width.
        height: Frame height.
        margin: Margin from edge (points too close to edge are invalid).

    Returns:
        True if point is valid.
    """
    x, y = pt
    return (margin <= x < width - margin) and (margin <= y < height - margin)


def draw_skeleton_transparent(
    frame: Frame,
    pose: Pose2D,
    width: int,
    height: int,
    normalized: bool = True,
    alpha: float = 0.7,
    **kwargs,
) -> Frame:
    """Draw skeleton with transparency (overlay blend).

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        pose: Pose array (17, 2) with joint coordinates.
        width: Frame width in pixels.
        height: Frame height in pixels.
        normalized: Whether pose coordinates are normalized [0, 1].
        alpha: Transparency factor [0, 1]. 0 = invisible, 1 = opaque.
        **kwargs: Additional arguments for draw_skeleton().

    Returns:
        Frame with transparent skeleton overlay.

    Example:
        >>> frame = cv2.imread("frame.jpg")
        >>> pose = np.random.rand(17, 2).astype(np.float32) * 0.5 + 0.25
        >>> draw_skeleton_transparent(frame, pose, 1920, 1080, alpha=0.5)
    """
    # Create skeleton overlay
    overlay = np.zeros_like(frame)
    draw_skeleton(overlay, pose, width, height, normalized, **kwargs)

    # Create mask where skeleton was drawn
    mask = (overlay > 0).any(axis=2).astype(np.float32)

    # Blend
    result = frame.copy()
    for c in range(3):
        result[:, :, c] = (
            frame[:, :, c] * (1 - mask * alpha) + overlay[:, :, c] * mask * alpha
        ).astype(np.uint8)

    return result
