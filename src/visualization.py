"""Visualization utilities for figure skating analysis.

Provides functions for drawing skeleton overlays, kinematics data,
edge indicators, subtitles (with Cyrillic support), and debug HUD.

Research-based best practices from Gemini Deep Research:
- cv2.LINE_AA for smooth anti-aliased lines
- Pillow (PIL) for Unicode/Cyrillic text rendering
- Color-coded skeleton by body region
- Layered HUD architecture for focused debugging

COORDINATE SYSTEM CONVENTION:
- All visualization functions expect NORMALIZED coordinates [0,1]
- Exception: draw_skeleton() accepts both normalized and pixel coords
- Convert using: normalize_pixel_poses() / pixelize_normalized_poses() from types.py
"""

from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import (
    BLAZEPOSE_SKELETON_EDGES,
    BKey,
    BladeType,
    MotionDirection,
    assert_pose_format,
)

# Color scheme (BGR) - based on Gemini research
COLOR_LEFT_SIDE = (255, 0, 0)  # Blue - left arm/leg
COLOR_RIGHT_SIDE = (0, 0, 255)  # Red - right arm/leg
COLOR_CENTER = (0, 255, 0)  # Green - torso/head
COLOR_JOINTS = (255, 255, 255)  # White - joint landmarks
COLOR_LOW_CONFIDENCE = (0, 255, 255)  # Yellow - low confidence

# Edge indicator colors
COLOR_EDGE_INSIDE = (255, 0, 0)  # Blue (BGR) = inside edge
COLOR_EDGE_OUTSIDE = (0, 0, 255)  # Red (BGR) = outside edge
COLOR_EDGE_FLAT = (0, 255, 255)  # Yellow = flat


# Velocity color gradient (BGR): blue (slow) -> red (fast)
def _get_velocity_color(speed: float, max_speed: float = 200.0) -> tuple[int, int, int]:
    """Get color for velocity based on speed magnitude.

    Args:
        speed: Speed in pixels/sec.
        max_speed: Speed for maximum red color.

    Returns:
        BGR color tuple.
    """
    t = min(speed / max_speed, 1.0)
    # Blue (slow) -> Green -> Yellow -> Red (fast)
    if t < 0.33:
        # Blue to Green
        local_t = t / 0.33
        return (int(255 * local_t), int(255 * local_t), int(255 * (1 - local_t)))
    elif t < 0.66:
        # Green to Yellow
        local_t = (t - 0.33) / 0.33
        return (int(255 * local_t), 255, 0)
    else:
        # Yellow to Red
        local_t = (t - 0.66) / 0.34
        return (255, int(255 * (1 - local_t)), 0)


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    height: int,
    width: int,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Draw BlazePose skeleton with anti-aliased lines.

    Uses cv2.LINE_AA for smooth rendering and color-codes by body region.
    Based on Gemini research: left=blue, right=red, center=green.

    Args:
        frame: Video frame (H, W, 3) BGR.
        keypoints: (33, 2) normalized [0,1] or (33, 3) pixel coordinates.
        height: Frame height for coordinate conversion.
        width: Frame width for coordinate conversion.
        confidence_threshold: Skip keypoints below this confidence (if 3rd dim exists).

    Returns:
        Frame with skeleton overlay (modified in-place).
    """
    # Handle different input formats
    if keypoints.shape[1] == 2:  # Normalized coordinates
        keypoints_px = keypoints * np.array([width, height])
        conf_values = None
    else:  # Pixel coordinates with confidence
        keypoints_px = keypoints[:, :2]
        conf_values = keypoints[:, 2] if keypoints.shape[1] == 3 else None

    # Categorize edges by body region
    torso_edges = {
        (BKey.LEFT_SHOULDER, BKey.RIGHT_SHOULDER),
        (BKey.LEFT_SHOULDER, BKey.LEFT_HIP),
        (BKey.RIGHT_SHOULDER, BKey.RIGHT_HIP),
        (BKey.LEFT_HIP, BKey.RIGHT_HIP),
    }

    left_arm_edges = {
        (BKey.LEFT_SHOULDER, BKey.LEFT_ELBOW),
        (BKey.LEFT_ELBOW, BKey.LEFT_WRIST),
        (BKey.LEFT_WRIST, BKey.LEFT_THUMB),
        (BKey.LEFT_WRIST, BKey.LEFT_INDEX),
        (BKey.LEFT_WRIST, BKey.LEFT_PINKY),
    }

    right_arm_edges = {
        (BKey.RIGHT_SHOULDER, BKey.RIGHT_ELBOW),
        (BKey.RIGHT_ELBOW, BKey.RIGHT_WRIST),
        (BKey.RIGHT_WRIST, BKey.RIGHT_THUMB),
        (BKey.RIGHT_WRIST, BKey.RIGHT_INDEX),
        (BKey.RIGHT_WRIST, BKey.RIGHT_PINKY),
    }

    left_leg_edges = {
        (BKey.LEFT_HIP, BKey.LEFT_KNEE),
        (BKey.LEFT_KNEE, BKey.LEFT_ANKLE),
        (BKey.LEFT_ANKLE, BKey.LEFT_HEEL),
        (BKey.LEFT_ANKLE, BKey.LEFT_FOOT_INDEX),
    }

    right_leg_edges = {
        (BKey.RIGHT_HIP, BKey.RIGHT_KNEE),
        (BKey.RIGHT_KNEE, BKey.RIGHT_ANKLE),
        (BKey.RIGHT_ANKLE, BKey.RIGHT_HEEL),
        (BKey.RIGHT_ANKLE, BKey.RIGHT_FOOT_INDEX),
    }

    # Draw connections first (lines)
    for idx1, idx2 in BLAZEPOSE_SKELETON_EDGES:
        pt1 = keypoints_px[idx1].astype(int)
        pt2 = keypoints_px[idx2].astype(int)

        # Skip if points are at origin (invalid)
        if np.allclose(pt1, 0) or np.allclose(pt2, 0):
            continue

        # Determine color by region
        edge = (idx1, idx2)
        if edge in torso_edges or (idx2, idx1) in torso_edges:
            color = COLOR_CENTER
        elif edge in left_arm_edges or (idx2, idx1) in left_arm_edges:
            color = COLOR_LEFT_SIDE
        elif edge in right_arm_edges or (idx2, idx1) in right_arm_edges:
            color = COLOR_RIGHT_SIDE
        elif edge in left_leg_edges or (idx2, idx1) in left_leg_edges:
            color = COLOR_LEFT_SIDE
        elif edge in right_leg_edges or (idx2, idx1) in right_leg_edges:
            color = COLOR_RIGHT_SIDE
        else:
            color = COLOR_CENTER  # Default for face/head

        # Draw with anti-aliasing
        cv2.line(frame, tuple(pt1), tuple(pt2), color, 2, lineType=cv2.LINE_AA)

    # Draw keypoints (circles on top)
    for i, kp in enumerate(keypoints_px):
        if np.allclose(kp, 0):
            continue

        pt = kp.astype(int)
        # Check confidence if available
        if conf_values is not None and conf_values[i] < confidence_threshold:
            color = COLOR_LOW_CONFIDENCE
        else:
            color = COLOR_JOINTS

        cv2.circle(frame, tuple(pt), 4, color, -1, lineType=cv2.LINE_AA)

    return frame


def draw_velocity_vectors(
    frame: np.ndarray,
    poses: np.ndarray,
    frame_idx: int,
    fps: float,
    height: int,
    width: int,
    joint_indices: list[int] | None = None,
) -> np.ndarray:
    """Draw velocity vectors for key joints using cv2.arrowedLine.

    Args:
        frame: Video frame (H, W, 3) BGR.
        poses: Full pose sequence (num_frames, 33, 2 or 3) NORMALIZED [0,1].
        frame_idx: Current frame index.
        fps: Frame rate for velocity scaling.
        height: Frame height.
        width: Frame width.
        joint_indices: Joints to visualize (default: wrists, ankles).

    Returns:
        Frame with velocity vectors overlay.

    Raises:
        AssertionError: If poses are not in normalized [0,1] format.
    """
    # Validate input format
    assert_pose_format(poses, "normalized", context="draw_velocity_vectors")
    if joint_indices is None:
        joint_indices = [
            BKey.LEFT_WRIST,
            BKey.RIGHT_WRIST,
            BKey.LEFT_ANKLE,
            BKey.RIGHT_ANKLE,
        ]

    # Handle both (33, 2) and (33, 3) pose formats
    poses_xy = poses[:, :, :2] if poses.shape[2] == 3 else poses

    # Compute velocity using central difference
    if frame_idx > 0 and frame_idx < len(poses) - 1:
        velocity = (poses_xy[frame_idx + 1] - poses_xy[frame_idx - 1]) * fps / 2
    else:
        velocity = np.zeros_like(poses_xy[frame_idx])

    for joint_idx in joint_indices:
        pos = poses_xy[frame_idx, joint_idx]  # (x, y) normalized
        vel = velocity[joint_idx]  # (vx, vy) normalized/sec

        # Convert to pixel coordinates
        pos_px = (pos * [width, height]).astype(int)
        vel_px = vel * [width, height]  # Scale to pixels/sec

        # Speed magnitude
        speed = np.linalg.norm(vel_px)

        if speed < 5:  # Threshold for still joints
            continue

        # Color by speed (blue=slow, red=fast)
        color = _get_velocity_color(speed)

        # Draw arrow
        scale_factor = 0.1  # Scale for visibility
        end_pos = pos_px + (vel_px * scale_factor).astype(int)

        # Clamp to frame bounds
        end_pos = np.clip(end_pos, [0, 0], [width - 1, height - 1])

        cv2.arrowedLine(frame, tuple(pos_px), tuple(end_pos), color, 3, tipLength=0.3)

        # Draw speed text
        speed_mps = speed / height * 2  # Approximate conversion
        text_pos = (int(pos_px[0]) + 10, int(pos_px[1]))
        cv2.putText(
            frame,
            f"{speed_mps:.1f}m/s",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return frame


def draw_trails(
    frame: np.ndarray,
    pose_history: deque,
    joint_idx: int = BKey.LEFT_ANKLE,
    height: int = 1080,
    width: int = 1920,
) -> np.ndarray:
    """Draw motion trails with gradient transparency.

    Args:
        frame: Video frame (H, W, 3) BGR.
        pose_history: Deque of recent poses (normalized).
        joint_idx: Joint index to trace.
        height: Frame height.
        width: Frame width.

    Returns:
        Frame with trail overlay.
    """
    if len(pose_history) < 2:
        return frame

    # Create overlay for alpha blending
    overlay = frame.copy()

    points = []
    for pose in pose_history:
        # Handle both (33, 2) and (33, 3) formats
        pos_xy = pose[joint_idx, :2] if pose.shape[1] == 3 else pose[joint_idx]
        pos = pos_xy * [width, height]
        points.append(pos.astype(int))

    # Draw trail with fading opacity
    num_points = len(points)
    for i in range(num_points - 1):
        pt1 = points[i]
        pt2 = points[i + 1]

        # Skip invalid points
        if np.allclose(pt1, 0) or np.allclose(pt2, 0):
            continue

        # Fade from transparent to opaque
        alpha = (i + 1) / num_points

        # Draw line on overlay
        cv2.line(overlay, tuple(pt1), tuple(pt2), (0, 255, 255), 2, lineType=cv2.LINE_AA)

        # Blend with frame based on alpha
        cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, frame)

    return frame


def draw_edge_indicators(
    frame: np.ndarray,
    poses: np.ndarray,
    frame_idx: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Draw skating edge indicators for both feet.

    Uses the edge detection logic from element_segmenter.py:
    heel-to-foot vector angle determines edge (+1=inside, -1=outside, 0=flat).

    Args:
        frame: Video frame (H, W, 3) BGR.
        poses: Full pose sequence (num_frames, 33, 2 or 3) normalized.
        frame_idx: Current frame index.
        height: Frame height.
        width: Frame width.

    Returns:
        Frame with edge indicators overlay.
    """
    # Handle both (33, 2) and (33, 3) formats
    poses_xy = poses[:, :, :2] if poses.shape[2] == 3 else poses

    # Compute edge indicator (same as element_segmenter.py)
    left_heel = poses_xy[frame_idx, BKey.LEFT_HEEL]
    left_foot = poses_xy[frame_idx, BKey.LEFT_FOOT_INDEX]
    right_heel = poses_xy[frame_idx, BKey.RIGHT_HEEL]
    right_foot = poses_xy[frame_idx, BKey.RIGHT_FOOT_INDEX]

    left_vector = left_foot - left_heel
    right_vector = right_foot - right_heel

    # Edge: x-component sign (+1=inside, -1=outside, 0=flat)
    left_edge = np.sign(left_vector[0])
    right_edge = np.sign(right_vector[0])

    # Convert to pixel positions
    left_pos = (poses_xy[frame_idx, BKey.LEFT_FOOT_INDEX] * [width, height]).astype(int)
    right_pos = (poses_xy[frame_idx, BKey.RIGHT_FOOT_INDEX] * [width, height]).astype(int)

    for pos, edge, label in [(left_pos, left_edge, "L"), (right_pos, right_edge, "R")]:
        if edge > 0.3:
            color = COLOR_EDGE_INSIDE  # Blue = inside
            text = f"{label}: IN"
        elif edge < -0.3:
            color = COLOR_EDGE_OUTSIDE  # Red = outside
            text = f"{label}: OUT"
        else:
            color = COLOR_EDGE_FLAT  # Yellow = flat
            text = f"{label}: FLT"

        # Draw circle around foot
        cv2.circle(frame, tuple(pos), 20, color, 2, lineType=cv2.LINE_AA)

        # Draw text label
        text_pos = (int(pos[0]) + 25, int(pos[1]))
        cv2.putText(
            frame,
            text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return frame


def draw_subtitle_cyrillic(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 30,
    font_path: str = "/usr/share/fonts/TTF/DejaVuSans.ttf",
    bg_alpha: float = 0.6,
) -> np.ndarray:
    """Draw text with Cyrillic support using Pillow.

    Critical for displaying Russian coach commentary from VTT files.
    cv2.putText doesn't support Unicode characters.

    Args:
        frame: Video frame (H, W, 3) BGR.
        text: Text to draw (can contain Cyrillic characters).
        position: (x, y) position for text.
        font_size: Font size in pixels.
        font_path: Path to TTF font file supporting Cyrillic.
        bg_alpha: Background transparency (0-1).

    Returns:
        Frame with text overlay.
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    # Load font (use system font that supports Cyrillic)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        # Fallback to default font
        font = ImageFont.load_default()

    # Get text size for background
    try:
        # For newer Pillow versions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # For older Pillow versions
        text_w, text_h = draw.textsize(text, font=font)

    x, y = position

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + 5),
        (0, 0, 0),
        -1,
    )
    frame[:] = cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0)

    # Draw text
    draw.text((x, y - text_h), text, font=font, fill=(255, 255, 255))

    # Convert back to BGR
    rgb_result = np.array(pil_image)
    return cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR)


def draw_text_box(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    bg_alpha: float = 0.3,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """Draw text with semi-transparent background.

    Helper function for HUD text rendering.

    Args:
        frame: Video frame (modified in-place).
        text: Text to draw.
        position: (x, y) position for text.
        bg_alpha: Background transparency (0-1).
        font_scale: Font scale for cv2.putText.
        thickness: Text thickness.
    """
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0)

    # Text with outline
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
    )


def draw_blade_indicator_hud(
    frame: np.ndarray,
    blade_state_left: object | None,
    blade_state_right: object | None,
    x: int = 10,
    y: int = 80,
) -> np.ndarray:
    """Draw blade edge state indicator on HUD.

    Shows current blade edge (inside/outside/flat/toe_pick) with color coding:
    - Green: Inside edge
    - Red: Outside edge
    - Yellow: Flat
    - Blue: Toe pick

    Args:
        frame: Video frame (H, W, 3) BGR.
        blade_state_left: Left foot BladeState.
        blade_state_right: Right foot BladeState.
        x: X position for HUD.
        y: Y position for HUD.

    Returns:
        Frame with blade indicator overlay.
    """
    from .types import BladeType

    # Color mapping for blade types
    blade_colors = {
        BladeType.INSIDE: (0, 200, 0),  # Green
        BladeType.OUTSIDE: (0, 0, 200),  # Red
        BladeType.FLAT: (0, 200, 200),  # Yellow
        BladeType.TOE_PICK: (200, 0, 0),  # Blue
        BladeType.UNKNOWN: (128, 128, 128),  # Gray
    }

    blade_names = {
        BladeType.INSIDE: "IN",
        BladeType.OUTSIDE: "OUT",
        BladeType.FLAT: "FLAT",
        BladeType.TOE_PICK: "TOE",
        BladeType.UNKNOWN: "???",
    }

    # Left foot
    if blade_state_left is not None:
        color = blade_colors.get(blade_state_left.blade_type, (128, 128, 128))
        name = blade_names.get(blade_state_left.blade_type, "???")
        conf = blade_state_left.confidence

        # Draw colored indicator box
        box_size = 20
        cv2.rectangle(frame, (x, y), (x + box_size, y + 10), color, -1)
        cv2.rectangle(frame, (x, y), (x + box_size, y + 10), (255, 255, 255), 1)

        # Draw label
        label = f"L: {name} ({conf:.0%})"
        draw_text_box(frame, label, (x + box_size + 5, y), font_scale=0.5)

    # Right foot (below left)
    if blade_state_right is not None:
        color = blade_colors.get(blade_state_right.blade_type, (128, 128, 128))
        name = blade_names.get(blade_state_right.blade_type, "???")
        conf = blade_state_right.confidence

        box_size = 20
        cv2.rectangle(frame, (x, y + 20), (x + box_size, y + 30), color, -1)
        cv2.rectangle(frame, (x, y + 20), (x + box_size, y + 30), (255, 255, 255), 1)

        label = f"R: {name} ({conf:.0%})"
        draw_text_box(frame, label, (x + box_size + 5, y + 20), font_scale=0.5)

    return frame


def draw_spatial_axes(
    frame: np.ndarray,
    camera_pose: object,  # CameraPose from spatial_reference
    origin: tuple[int, int] | None = None,
    length: int = 40,
    font_scale: float = 0.4,
) -> np.ndarray:
    """Draw XYZ axes on frame to visualize spatial reference.

    Shows the true vertical and horizontal directions relative to gravity,
    compensating for camera tilt. This is essential for accurate angle
    measurements in skating analysis.

    Color coding (BGR):
    - X axis (blue): parallel to ice, horizontal
    - Y axis (green): forward direction (depth)
    - Z axis (red): vertical (up, opposite to gravity)

    Args:
        frame: Video frame (H, W, 3) BGR.
        camera_pose: CameraPose object with roll, pitch, yaw attributes.
        origin: Pixel position for axes origin (x, y). Defaults to bottom-left.
        length: Length of each axis in pixels.
        font_scale: Font scale for labels.

    Returns:
        Frame with axes drawn (modified in-place).
    """
    if origin is None:
        origin = (50, frame.shape[0] - 80)
    """Draw XYZ axes on frame to visualize spatial reference.

    Shows the true vertical and horizontal directions relative to gravity,
    compensating for camera tilt. This is essential for accurate angle
    measurements in skating analysis.

    Color coding (BGR):
    - X axis (blue): parallel to ice, horizontal
    - Y axis (green): forward direction (depth)
    - Z axis (red): vertical (up, opposite to gravity)

    Args:
        frame: Video frame (H, W, 3) BGR.
        camera_pose: CameraPose object with roll, pitch, yaw attributes.
        origin: Pixel position for axes origin (x, y). Defaults to bottom-left.
        length: Length of each axis in pixels.
        font_scale: Font scale for labels.

    Returns:
        Frame with axes drawn (modified in-place).
    """
    # Get rotation matrix from camera pose
    from scipy.spatial.transform import Rotation

    # Create rotation matrix
    r = Rotation.from_euler(
        "xyz", [camera_pose.roll, camera_pose.pitch, camera_pose.yaw], degrees=True
    )
    R = r.as_matrix()

    # Axis directions in world space: X=right, Y=forward, Z=up
    axes_world = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Rotate axes by camera pose (to show how they appear from camera perspective)
    axes_camera = R @ axes_world.T

    # Colors (BGR): X=blue, Y=green, Z=red
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
    labels = ["X", "Y", "Z"]

    for i, (axis, color, label) in enumerate(zip(axes_camera.T, colors, labels)):
        # Project 3D to 2D (simple orthographic)
        # For visualization, we show X (horizontal) and Z (vertical)
        if i == 1:  # Y axis (depth) - draw at an angle
            scale = 0.5
            end_x = int(origin[0] + axis[0] * length * scale)
            end_y = int(origin[1] - axis[2] * length * scale)
        else:  # X and Z axes
            end_x = int(origin[0] + axis[0] * length)
            end_y = int(origin[1] - axis[2] * length)

        # Draw axis line with anti-aliasing
        cv2.line(frame, origin, (end_x, end_y), color, 2, cv2.LINE_AA)

        # Draw label
        cv2.putText(
            frame,
            label,
            (end_x + 5, end_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    # Draw pose info text (compact, below axes)
    info_lines = [
        f"R:{camera_pose.roll:.0f}° P:{camera_pose.pitch:.0f}°",
        f"{camera_pose.source} [{camera_pose.confidence:.0%}]",
    ]

    y_offset = origin[1] + length + 5
    for line in info_lines:
        # Draw text with semi-transparent background
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (origin[0], y_offset - h - 2),
            (origin[0] + w + 4, y_offset + 2),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame,
            line,
            (origin[0], y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += 12

    return frame


def draw_debug_hud(
    frame: np.ndarray,
    element_info: dict,
    kinematics: dict,
    frame_idx: int,
    total_frames: int,
    fps: float,
    height: int,
    width: int,
    blade_state_left: object | None = None,
    blade_state_right: object | None = None,
) -> np.ndarray:
    """Draw comprehensive debug HUD.

    Layout:
    - Top-left: Element info (name, boundaries, confidence)
    - Top-left (below element): Blade edge indicators
    - Top-right: Frame counter, timestamp
    - Bottom-left: Kinematics (velocities, angles)

    Args:
        frame: Video frame (H, W, 3) BGR.
        element_info: Dict with 'type', 'start', 'end', 'confidence'.
        kinematics: Dict with kinematic metrics (hip_velocity, knee angles, etc.).
        frame_idx: Current frame index.
        total_frames: Total number of frames.
        fps: Frame rate.
        height: Frame height.
        width: Frame width.
        blade_state_left: Left foot BladeState (optional).
        blade_state_right: Right foot BladeState (optional).

    Returns:
        Frame with HUD overlay.
    """
    # Top-left: Element info
    if element_info:
        elem_type = element_info.get("type", "unknown")
        start = element_info.get("start", 0)
        end = element_info.get("end", 0)
        conf = element_info.get("confidence", 0.0)
        elem_text = f"{elem_type} [{start}:{end}] conf={conf:.2f}"
        draw_text_box(frame, elem_text, (10, 30))

    # Top-right: Frame counter
    time_sec = frame_idx / fps
    frame_text = f"Frame: {frame_idx}/{total_frames} | {time_sec:.2f}s"
    (fw, _fh), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    draw_text_box(frame, frame_text, (width - fw - 20, 30))

    # Bottom-left: Kinematics
    y_offset = height - 30
    for key, value in kinematics.items():
        text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
        draw_text_box(frame, text, (10, y_offset))
        y_offset -= 25

    # Blade edge indicators (below element info)
    if blade_state_left is not None or blade_state_right is not None:
        frame = draw_blade_indicator_hud(frame, blade_state_left, blade_state_right, x=10, y=55)

    return frame


# ============================================================================
# 3D Pose Visualization
# ============================================================================

def project_3d_to_2d(
    poses_3d: np.ndarray,
    focal_length: float = 500.0,
    camera_distance: float = 5.0,
) -> np.ndarray:
    """Project 3D poses to 2D using simple perspective projection.

    Args:
        poses_3d: (N, 17, 3) array with x, y, z in meters
        focal_length: Camera focal length in pixels
        camera_distance: Distance from camera to subject in meters

    Returns:
        poses_2d: (N, 17, 2) array with normalized [0,1] coordinates
    """
    n_frames = poses_3d.shape[0]
    poses_2d = np.zeros((n_frames, 17, 2), dtype=np.float32)

    # Simple perspective projection
    # x' = x * f / (z + D)
    # y' = y * f / (z + D)

    for frame_idx in range(n_frames):
        pose_3d = poses_3d[frame_idx]

        for joint_idx in range(17):
            x, y, z = pose_3d[joint_idx]

            # Avoid division by zero and extreme projections
            z_safe = z + camera_distance
            if abs(z_safe) < 0.1:
                z_safe = 0.1 if z_safe >= 0 else -0.1

            # Perspective projection
            scale = focal_length / z_safe
            x_proj = x * scale
            y_proj = y * scale

            # Normalize to [0, 1] with clipping
            poses_2d[frame_idx, joint_idx] = [
                np.clip(x_proj + 0.5, 0, 1),  # Center X
                np.clip(0.5 - y_proj, 0, 1),  # Center Y (flip Y)
            ]

    return poses_2d


def draw_skeleton_3d(
    frame: np.ndarray,
    pose_3d: np.ndarray,
    skeleton_edges: list[tuple[int, int]],
    height: int,
    width: int,
    focal_length: float = 500.0,
    camera_distance: float = 5.0,
) -> np.ndarray:
    """Draw 3D skeleton projected onto 2D frame.

    Args:
        frame: Input frame (H, W, 3)
        pose_3d: (17, 3) array with x, y, z in meters
        skeleton_edges: List of (joint1, joint2) connections
        height: Frame height
        width: Frame width
        focal_length: Camera focal length
        camera_distance: Camera distance

    Returns:
        Frame with skeleton overlay
    """
    frame = frame.copy()

    # Project 3D to 2D
    pose_2d = project_3d_to_2d(
        pose_3d[np.newaxis, ...],
        focal_length,
        camera_distance,
    )[0]

    # Draw edges
    for idx1, idx2 in skeleton_edges:
        pt1 = pose_2d[idx1]
        pt2 = pose_2d[idx2]

        # Convert to pixel coordinates
        x1, y1 = int(pt1[0] * width), int(pt1[1] * height)
        x2, y2 = int(pt2[0] * width), int(pt2[1] * height)

        # Check if points are valid (within frame)
        if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
            cv2.line(frame, (x1, y1), (x2, y2), COLOR_CENTER, 2, cv2.LINE_AA)

    # Draw joints
    for joint_idx in range(pose_2d.shape[0]):
        pt = pose_2d[joint_idx]
        x, y = int(pt[0] * width), int(pt[1] * height)

        if 0 <= x < width and 0 <= y < height:
            # Use depth (Z) for color coding
            z = pose_3d[joint_idx, 2]
            depth_color = _get_depth_color(z)
            cv2.circle(frame, (x, y), 4, depth_color, -1, cv2.LINE_AA)

    return frame


def _get_depth_color(z: float, z_range: float = 1.0) -> tuple[int, int, int]:
    """Get color based on depth (Z) value.

    Args:
        z: Depth value in meters
        z_range: Range for color mapping

    Returns:
        BGR color tuple
    """
    # Normalize Z to [0, 1] for color mapping
    z_norm = (z + z_range / 2) / z_range
    z_norm = np.clip(z_norm, 0, 1)

    # Color gradient: blue (far) -> green (mid) -> red (close)
    if z_norm < 0.5:
        # Blue to green
        t = z_norm * 2
        r = 0
        g = int(255 * t)
        b = int(255 * (1 - t))
    else:
        # Green to red
        t = (z_norm - 0.5) * 2
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 0

    return (b, g, r)


def draw_3d_trajectory(
    frame: np.ndarray,
    com_trajectory: np.ndarray,
    height: int,
    width: int,
    focal_length: float = 500.0,
    camera_distance: float = 5.0,
) -> np.ndarray:
    """Draw Center of Mass trajectory in 3D.

    Args:
        frame: Input frame (H, W, 3)
        com_trajectory: (N, 3) CoM trajectory over time
        height: Frame height
        width: Frame width
        focal_length: Camera focal length
        camera_distance: Camera distance

    Returns:
        Frame with trajectory overlay
    """
    frame = frame.copy()
    n_frames = com_trajectory.shape[0]

    # Project CoM trajectory to 2D (handle (N, 3) input)
    com_2d = np.zeros((n_frames, 2), dtype=np.float32)
    for i in range(n_frames):
        x, y, z = com_trajectory[i]

        # Avoid division by zero
        z_safe = z + camera_distance
        if abs(z_safe) < 0.1:
            z_safe = 0.1 if z_safe >= 0 else -0.1

        # Perspective projection
        scale = focal_length / z_safe
        x_proj = x * scale
        y_proj = y * scale

        # Normalize to [0, 1] with clipping
        com_2d[i] = [
            np.clip(x_proj + 0.5, 0, 1),
            np.clip(0.5 - y_proj, 0, 1),
        ]

    # Draw trajectory line
    points = []
    for i in range(len(com_2d)):
        x, y = com_2d[i]
        px, py = int(x * width), int(y * height)
        if 0 <= px < width and 0 <= py < height:
            points.append((px, py))

    # Draw connected line
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 255), 2, cv2.LINE_AA)

    # Draw current CoM position
    if points:
        cv2.circle(frame, points[-1], 6, (0, 255, 255), -1, cv2.LINE_AA)

    return frame


# ============================================================================
# 3D Blade Detection Visualization
# ============================================================================

def draw_ice_trace(
    frame: np.ndarray,
    trace: "IceTrace",
    height: int,
    width: int,
    focal_length: float = 500.0,
    camera_distance: float = 5.0,
) -> np.ndarray:
    """Draw ice trace - path of blade on ice surface.

    Args:
        frame: Input frame (H, W, 3)
        trace: IceTrace with points and blade types
        height: Frame height
        width: Frame width
        focal_length: Camera focal length
        camera_distance: Camera distance

    Returns:
        Frame with ice trace overlay
    """
    from .types import BladeType

    frame = frame.copy()

    if len(trace.points) < 2:
        return frame

    # Color mapping for blade types
    color_map = {
        BladeType.INSIDE: (255, 100, 100),    # Red/Cyan
        BladeType.OUTSIDE: (100, 100, 255),   # Blue/Cyan
        BladeType.FLAT: (100, 255, 100),      # Green
        BladeType.TOE_PICK: (255, 255, 0),    # Cyan
        BladeType.ROCKER: (255, 150, 0),      # Orange
        BladeType.HEEL: (150, 0, 255),        # Purple
        BladeType.UNKNOWN: (128, 128, 128),   # Gray
    }

    # Project 3D points to 2D
    points_2d = []
    for point in trace.points:
        x, y, z = point
        z_safe = z + camera_distance
        if abs(z_safe) < 0.1:
            z_safe = 0.1 if z_safe >= 0 else -0.1

        x_proj = focal_length * x / z_safe
        y_proj = focal_length * y / z_safe

        # Convert to pixel coordinates (center origin)
        px = int((x_proj + 1) * width / 2)
        py = int((y_proj + 1) * height / 2)

        if 0 <= px < width and 0 <= py < height:
            points_2d.append((px, py))

    # Draw trace segments with color based on blade type
    for i in range(len(points_2d) - 1):
        pt1 = points_2d[i]
        pt2 = points_2d[i + 1]

        # Get blade type for this segment
        if i < len(trace.blade_types):
            blade_type = trace.blade_types[i]
            color = color_map.get(blade_type, (128, 128, 128))
        else:
            color = (128, 128, 128)

        # Draw line segment
        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw current position marker
    if points_2d:
        cv2.circle(frame, points_2d[-1], 5, (0, 255, 255), -1, cv2.LINE_AA)

    return frame


def draw_blade_state_3d_hud(
    frame: np.ndarray,
    blade_state: "BladeState3D",
    position: tuple[int, int],
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw 3D blade state information on HUD.

    Args:
        frame: Input frame (H, W, 3)
        blade_state: BladeState3D with full 3D information
        position: (x, y) position for HUD
        font_scale: Font scale multiplier

    Returns:
        Frame with HUD overlay
    """
    from .types import BladeType, MotionDirection

    frame = frame.copy()
    x, y = position
    line_height = int(20 * font_scale)

    # Background box
    box_width = 220
    box_height = 140
    cv2.rectangle(frame, (x - 5, y - 5), (x + box_width, y + box_height),
                 (0, 0, 0), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x - 5, y - 5), (x + box_width, y + box_height),
                 (255, 255, 255), 1, cv2.LINE_AA)

    # Foot label
    foot_label = f"Foot: {blade_state.foot.upper()}"
    cv2.putText(frame, foot_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y += line_height

    # Blade type with color
    blade_color = {
        BladeType.INSIDE: (255, 100, 100),
        BladeType.OUTSIDE: (100, 100, 255),
        BladeType.FLAT: (100, 255, 100),
        BladeType.TOE_PICK: (255, 255, 0),
        BladeType.ROCKER: (255, 150, 0),
        BladeType.HEEL: (150, 0, 255),
        BladeType.UNKNOWN: (128, 128, 128),
    }.get(blade_state.blade_type, (255, 255, 255))

    blade_label = f"Blade: {blade_state.blade_type.value.upper()}"
    cv2.putText(frame, blade_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.6, blade_color, 1, cv2.LINE_AA)
    y += line_height

    # Motion direction
    direction_label = f"Direction: {blade_state.motion_direction.value.upper()}"
    cv2.putText(frame, direction_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    y += line_height

    # Angles
    angle_label = f"Foot: {blade_state.foot_angle:.1f}°  Ankle: {blade_state.ankle_angle:.1f}°"
    cv2.putText(frame, angle_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += line_height

    # Knee angle
    knee_label = f"Knee: {blade_state.knee_angle:.1f}°"
    cv2.putText(frame, knee_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += line_height

    # Velocity
    vx, vy, vz = blade_state.velocity_3d
    # Clip to prevent overflow
    vx, vy, vz = np.clip([vx, vy, vz], -1000.0, 1000.0)
    vel_mag = (vx**2 + vy**2 + vz**2)**0.5
    vel_mag = min(vel_mag, 99.99)  # Prevent overflow
    vel_label = f"Velocity: {vel_mag:.2f} m/s"
    cv2.putText(frame, vel_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += line_height

    # Confidence
    conf_color = (0, 255, 0) if blade_state.confidence > 0.7 else (0, 165, 255)
    conf_label = f"Conf: {blade_state.confidence:.2f}"
    cv2.putText(frame, conf_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.5, conf_color, 1, cv2.LINE_AA)

    return frame


def draw_motion_direction_arrow(
    frame: np.ndarray,
    direction: MotionDirection,
    foot_pos: tuple[int, int],
    arrow_length: int = 30,
) -> np.ndarray:
    """Draw arrow indicating motion direction.

    Args:
        frame: Input frame (H, W, 3)
        direction: MotionDirection enum
        foot_pos: (x, y) foot position on screen
        arrow_length: Length of arrow in pixels

    Returns:
        Frame with direction arrow
    """
    frame = frame.copy()
    x, y = foot_pos

    # Arrow directions (offsets)
    arrows = {
        MotionDirection.FORWARD: (0, -arrow_length),
        MotionDirection.BACKWARD: (0, arrow_length),
        MotionDirection.LEFT: (-arrow_length, 0),
        MotionDirection.RIGHT: (arrow_length, 0),
        MotionDirection.DIAGONAL_LEFT: (-int(arrow_length * 0.7), -int(arrow_length * 0.7)),
        MotionDirection.DIAGONAL_RIGHT: (int(arrow_length * 0.7), -int(arrow_length * 0.7)),
        MotionDirection.ROTATION_LEFT: None,  # Draw circular arrow
        MotionDirection.ROTATION_RIGHT: None,  # Draw circular arrow
        MotionDirection.STATIONARY: None,  # No arrow
    }

    offset = arrows.get(direction)
    if offset is None:
        if direction == MotionDirection.ROTATION_LEFT:
            # Draw counter-clockwise circle
            cv2.circle(frame, (x, y), arrow_length // 2, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "↺", (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2, cv2.LINE_AA)
        elif direction == MotionDirection.ROTATION_RIGHT:
            # Draw clockwise circle
            cv2.circle(frame, (x, y), arrow_length // 2, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "↻", (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2, cv2.LINE_AA)
        # STATIONARY: draw nothing
        return frame

    # Draw arrow
    end_x = x + offset[0]
    end_y = y + offset[1]
    cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 255, 0),
                    3, cv2.LINE_AA, tipLength=0.3)

    return frame
