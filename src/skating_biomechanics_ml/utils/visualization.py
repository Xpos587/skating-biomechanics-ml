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

from skating_biomechanics_ml.types import (
    BLAZEPOSE_SKELETON_EDGES,
    BKey,
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
    from skating_biomechanics_ml.types import BladeType

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
