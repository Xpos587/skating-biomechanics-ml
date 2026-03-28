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

# Axis alignment colors
COLOR_AXIS_GOOD = (0, 255, 0)  # Green = upright (<15°)
COLOR_AXIS_WARNING = (0, 255, 255)  # Yellow = warning (15-30°)
COLOR_AXIS_BAD = (0, 0, 255)  # Red = bad (>30°)


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

    .. DEPRECATED::
        This function uses simplified edge detection (sign of foot_vector.x).
        Use `draw_blade_indicator_hud()` with `BladeEdgeDetector` results instead.

    Uses the edge detection logic from element_segmenter.py:
    heel-to-foot vector angle determines edge (+1=inside, -1=outside, 0=flat).

    For proper blade edge visualization:
    1. Run `BladeEdgeDetector.detect_sequence()` to get BladeState objects
    2. Call `draw_blade_indicator_hud()` with the blade states
    3. States appear in HUD with confidence values

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


def calculate_trunk_angle(pose: np.ndarray) -> float:
    """Calculate trunk tilt angle from vertical.

    Args:
        pose: Single pose (33, 2) or (33, 3) normalized.

    Returns:
        Angle in degrees (0 = upright, positive = forward lean).
    """
    # Compute mid-shoulder and mid-hip
    mid_shoulder = (pose[BKey.LEFT_SHOULDER] + pose[BKey.RIGHT_SHOULDER]) / 2
    mid_hip = (pose[BKey.LEFT_HIP] + pose[BKey.RIGHT_HIP]) / 2

    # Vector from hip to shoulder
    spine_vector = mid_shoulder - mid_hip

    # Angle from vertical: atan2(x, -y) gives 0° for upright
    angle = np.arctan2(spine_vector[0], -spine_vector[1])
    return float(np.degrees(angle))


def draw_axis_indicator(
    frame: np.ndarray,
    pose: np.ndarray,
    height: int,
    width: int,
    threshold_warning: float = 15.0,
    threshold_bad: float = 30.0,
    show_angle: bool = True,
) -> np.ndarray:
    """Draw axis alignment indicator (trunk tilt from vertical).

    Visualizes:
    - Vertical reference line (gray)
    - Actual spine vector (color-coded by angle)
    - Angle value with color-coded text

    Args:
        frame: Video frame (H, W, 3) BGR.
        pose: Single pose (33, 2) or (33, 3) normalized.
        height: Frame height.
        width: Frame width.
        threshold_warning: Warning threshold in degrees.
        threshold_bad: Bad threshold in degrees.
        show_angle: Show angle value text.

    Returns:
        Frame with axis indicator overlay.
    """
    # Calculate angle
    angle = calculate_trunk_angle(pose)
    abs_angle = abs(angle)

    # Determine color and status
    if abs_angle < threshold_warning:
        color = COLOR_AXIS_GOOD
        status = "OK"
    elif abs_angle < threshold_bad:
        color = COLOR_AXIS_WARNING
        status = "WARN"
    else:
        color = COLOR_AXIS_BAD
        status = "FAIL"

    # Get keypoint positions in pixels
    poses_xy = pose[:, :2] if pose.shape[1] == 3 else pose
    mid_shoulder = ((poses_xy[BKey.LEFT_SHOULDER] + poses_xy[BKey.RIGHT_SHOULDER]) / 2) * [width, height]
    mid_hip = ((poses_xy[BKey.LEFT_HIP] + poses_xy[BKey.RIGHT_HIP]) / 2) * [width, height]

    # Draw vertical reference line (from hip upward)
    hip_pos = mid_hip.astype(int)
    vertical_top = (hip_pos[0], max(0, int(hip_pos[1] - 150)))  # 150px upward
    cv2.line(frame, tuple(hip_pos), vertical_top, (128, 128, 128), 1, cv2.LINE_AA)

    # Draw actual spine vector (color-coded)
    shoulder_pos = mid_shoulder.astype(int)
    cv2.line(frame, tuple(hip_pos), tuple(shoulder_pos), color, 3, cv2.LINE_AA)

    # Draw angle text if requested
    if show_angle:
        text = f"AXIS: {abs_angle:.1f}° [{status}]"
        text_pos = (int(hip_pos[0]) + 20, max(20, int(hip_pos[1]) - 50))
        draw_text_box(frame, text, text_pos, font_scale=0.6)

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
    length: int = 60,
    font_scale: float = 0.5,
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
        origin: Pixel position for axes origin (x, y). Defaults to bottom-right.
        length: Length of each axis in pixels.
        font_scale: Font scale for labels.

    Returns:
        Frame with axes drawn (modified in-place).
    """
    if origin is None:
        # Bottom-right corner (more visible)
        origin = (frame.shape[1] - 100, frame.shape[0] - 100)

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

    # Colors (BGR): brighter colors for visibility
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # BGR - brighter
    labels = ["X", "Y", "Z"]

    # Draw semi-transparent background for better visibility
    bg_size = length + 20
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (origin[0] - bg_size, origin[1] - bg_size),
        (origin[0] + bg_size, origin[1] + bg_size),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw origin circle to make it more visible
    cv2.circle(frame, origin, 6, (255, 255, 255), -1, cv2.LINE_AA)

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

        # Draw axis line with anti-aliasing (thicker for visibility)
        cv2.line(frame, origin, (end_x, end_y), color, 4, cv2.LINE_AA)

        # Draw circle at end of axis
        cv2.circle(frame, (end_x, end_y), 5, color, -1, cv2.LINE_AA)

        # Draw label (larger, with outline for visibility)
        label_pos = (end_x + 8, end_y + 5)
        # Text outline (black)
        cv2.putText(
            frame,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.2,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        # Text (colored)
        cv2.putText(
            frame,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.2,
            color,
            2,
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
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    width: int = 1920,
    height: int = 1080,
    camera_z: float = 3.0,
) -> np.ndarray:
    """Project 3D poses to 2D using OpenCV cv2.projectPoints().

    Args:
        poses_3d: (N, 17, 3) array with x, y, z in meters (Y-up coordinate system)
        camera_matrix: 3x3 camera intrinsic matrix (auto-generated if None)
        dist_coeffs: Distortion coefficients (zeros if None)
        width: Image width for default camera matrix
        height: Image height for default camera matrix
        camera_z: Distance of camera from subject along Z axis (meters)

    Returns:
        poses_2d: (N, 17, 2) array with normalized [0,1] coordinates
    """
    import cv2

    n_frames, n_joints, _ = poses_3d.shape
    poses_2d = np.zeros((n_frames, n_joints, 2), dtype=np.float32)

    # Default camera matrix if not provided
    if camera_matrix is None:
        # Estimate focal length based on image width (typical FOV ~60°)
        fx = fy = width  # Simple approximation: focal_length ≈ image_width
        cx = width / 2.0
        cy = height / 2.0
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # Camera pose: place camera at Z = camera_z, looking at origin
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.array([0, 0, camera_z], dtype=np.float32)  # Camera moved back by camera_z

    for frame_idx in range(n_frames):
        pose_3d = poses_3d[frame_idx].copy()

        # Flip Y axis: our poses use Y-up, OpenCV uses Y-down
        pose_3d[:, 1] = -pose_3d[:, 1]

        # Project using OpenCV
        projected, _ = cv2.projectPoints(
            pose_3d, rvec, tvec, camera_matrix, dist_coeffs
        )

        # Convert pixels to normalized [0, 1]
        for joint_idx in range(n_joints):
            px, py = projected[joint_idx][0]
            poses_2d[frame_idx, joint_idx] = [
                np.clip(px / width, 0, 1),
                np.clip(py / height, 0, 1),
            ]

    return poses_2d


def draw_skeleton_3d(
    frame: np.ndarray,
    pose_3d: np.ndarray,
    skeleton_edges: list[tuple[int, int]],
    height: int,
    width: int,
    camera_matrix: np.ndarray | None = None,
    camera_z: float = 3.0,
) -> np.ndarray:
    """Draw 3D skeleton projected onto 2D frame using cv2.projectPoints().

    Args:
        frame: Input frame (H, W, 3)
        pose_3d: (17, 3) array with x, y, z coordinates (in meters)
        skeleton_edges: List of (joint1, joint2) connections
        height: Frame height
        width: Frame width
        camera_matrix: 3x3 camera intrinsic matrix (auto-calculated if None)
        camera_z: Distance of camera from subject (meters)

    Returns:
        Frame with skeleton overlay
    """
    frame = frame.copy()

    # Project using OpenCV
    pose_2d = project_3d_to_2d(
        pose_3d[np.newaxis, ...],
        camera_matrix=camera_matrix,
        width=width,
        height=height,
        camera_z=camera_z,
    )[0]

    # Draw edges
    for idx1, idx2 in skeleton_edges:
        pt1 = pose_2d[idx1]
        pt2 = pose_2d[idx2]

        x1 = int(pt1[0] * width)
        y1 = int(pt1[1] * height)
        x2 = int(pt2[0] * width)
        y2 = int(pt2[1] * height)

        if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
            cv2.line(frame, (x1, y1), (x2, y2), COLOR_CENTER, 2, cv2.LINE_AA)

    # Draw joints
    for joint_idx in range(pose_2d.shape[0]):
        pt = pose_2d[joint_idx]
        x = int(pt[0] * width)
        y = int(pt[1] * height)

        if 0 <= x < width and 0 <= y < height:
            z = pose_3d[joint_idx, 2]
            depth_color = _get_depth_color(z)
            cv2.circle(frame, (x, y), 4, depth_color, -1, cv2.LINE_AA)

    return frame


def draw_skeleton_3d_pip(
    frame: np.ndarray,
    pose_3d: np.ndarray,
    skeleton_edges: list[tuple[int, int]],
    height: int,
    width: int,
    camera_matrix: np.ndarray | None = None,
    camera_z: float = 1.8,
) -> np.ndarray:
    """Draw 3D skeleton in top-right corner (natural background).

    Args:
        frame: Input frame (H, W, 3)
        pose_3d: (17, 3) array with x, y, z coordinates (in meters)
        skeleton_edges: List of (joint1, joint2) connections
        height: Frame height (main video)
        width: Frame width (main video)
        camera_matrix: 3x3 camera intrinsic matrix (auto-calculated if None)
        camera_z: Distance of camera from subject (smaller = larger skeleton)

    Returns:
        Frame with PIP skeleton overlay
    """
    frame = frame.copy()

    # PIP area: top-right quadrant
    pip_width = width // 2
    pip_height = height // 2
    pip_x = width - pip_width
    pip_y = 0

    # Project skeleton to PIP dimensions (closer camera = larger skeleton)
    pose_2d = project_3d_to_2d(
        pose_3d[np.newaxis, ...],
        camera_matrix=camera_matrix,
        width=pip_width,
        height=pip_height,
        camera_z=camera_z,
    )[0]

    # Draw edges (offset to PIP position)
    for idx1, idx2 in skeleton_edges:
        pt1 = pose_2d[idx1]
        pt2 = pose_2d[idx2]

        x1 = int(pt1[0] * pip_width) + pip_x
        y1 = int(pt1[1] * pip_height) + pip_y
        x2 = int(pt2[0] * pip_width) + pip_x
        y2 = int(pt2[1] * pip_height) + pip_y

        cv2.line(frame, (x1, y1), (x2, y2), COLOR_CENTER, 2, cv2.LINE_AA)

    # Draw joints (offset to PIP position)
    for joint_idx in range(pose_2d.shape[0]):
        pt = pose_2d[joint_idx]
        x = int(pt[0] * pip_width) + pip_x
        y = int(pt[1] * pip_height) + pip_y

        z = pose_3d[joint_idx, 2]
        depth_color = _get_depth_color(z)
        cv2.circle(frame, (x, y), 5, depth_color, -1, cv2.LINE_AA)

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
    camera_matrix: np.ndarray | None = None,
    camera_z: float = 3.0,
) -> np.ndarray:
    """Draw Center of Mass trajectory in 3D using cv2.projectPoints().

    Args:
        frame: Input frame (H, W, 3)
        com_trajectory: (N, 3) CoM trajectory over time
        height: Frame height
        width: Frame width
        camera_matrix: 3x3 camera intrinsic matrix (auto-generated if None)
        camera_z: Distance of camera from subject (meters)

    Returns:
        Frame with trajectory overlay
    """
    import cv2

    frame = frame.copy()
    n_frames = com_trajectory.shape[0]

    # Default camera matrix if not provided
    if camera_matrix is None:
        fx = fy = width
        cx = width / 2.0
        cy = height / 2.0
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.array([0, 0, camera_z], dtype=np.float32)

    # Flip Y for OpenCV coordinate system
    com_trajectory_copy = com_trajectory.copy()
    com_trajectory_copy[:, 1] = -com_trajectory_copy[:, 1]

    # Project using OpenCV
    projected, _ = cv2.projectPoints(
        com_trajectory_copy, rvec, tvec, camera_matrix, dist_coeffs
    )

    # Convert to normalized [0, 1]
    com_2d = np.zeros((n_frames, 2), dtype=np.float32)
    for i in range(n_frames):
        px, py = projected[i][0]
        com_2d[i] = [px / width, py / height]

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
    font_scale: float = 0.7,
) -> np.ndarray:
    """Draw 3D blade state information on HUD - compact version.

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
    line_height = int(25 * font_scale)

    # Semi-transparent background
    box_width = 180
    box_height = int(line_height * 4.5)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height),
                 (0, 0, 0), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height),
                 (255, 255, 255), 1, cv2.LINE_AA)

    # Foot + Blade (combined, color-coded)
    blade_color = {
        BladeType.INSIDE: (255, 80, 80),
        BladeType.OUTSIDE: (80, 80, 255),
        BladeType.FLAT: (80, 255, 80),
        BladeType.TOE_PICK: (255, 200, 0),
        BladeType.ROCKER: (255, 140, 0),
        BladeType.HEEL: (140, 0, 255),
        BladeType.UNKNOWN: (150, 150, 150),
    }.get(blade_state.blade_type, (200, 200, 200))

    blade_short = {
        BladeType.INSIDE: "IN",
        BladeType.OUTSIDE: "OUT",
        BladeType.FLAT: "FLAT",
        BladeType.TOE_PICK: "TOE",
        BladeType.ROCKER: "ROCK",
        BladeType.HEEL: "HEEL",
        BladeType.UNKNOWN: "???",
    }.get(blade_state.blade_type, "???")

    # Main info: Foot + Blade + Direction
    main_label = f"{blade_state.foot.upper()[0]} {blade_short} {blade_state.motion_direction.value.upper()[:5]}"
    cv2.putText(frame, main_label, (x + 5, y + int(line_height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.7, blade_color, 2, cv2.LINE_AA)
    y += line_height

    # Angles (combined)
    angle_label = f"Foot:{blade_state.foot_angle:.0f}° Ankle:{blade_state.ankle_angle:.0f}°"
    cv2.putText(frame, angle_label, (x + 5, y + int(line_height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    y += line_height

    # Velocity (simplified)
    vx, vy, vz = np.clip(list(blade_state.velocity_3d), -100.0, 100.0)
    vel_mag = min((vx**2 + vy**2 + vz**2)**0.5, 50.0)
    vel_label = f"V:{vel_mag:.1f}m/s"
    cv2.putText(frame, vel_label, (x + 5, y + int(line_height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    y += line_height

    # Confidence bar (visual)
    conf_width = int(box_width * blade_state.confidence)
    conf_color = (0, 255, 0) if blade_state.confidence > 0.7 else (0, 165, 255)
    cv2.rectangle(frame, (x + 5, y + int(line_height * 0.3)), (x + 5 + conf_width, y + int(line_height * 0.7)),
                 conf_color, -1, cv2.LINE_AA)

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
