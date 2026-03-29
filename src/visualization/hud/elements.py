"""Individual HUD element renderers.

Provides functions for rendering specific HUD elements:
- Frame counter
- Metrics display
- Phase indicator
- Blade indicator
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from src.types import BladeState3D, MetricResult
from src.visualization.config import (
    blade_flat_color,
    blade_indicator_size,
    blade_indicator_thickness,
    blade_inside_color,
    blade_outside_color,
    blade_unknown_color,
    font_color,
    font_scale,
    font_thickness,
    hud_bg_alpha,
    hud_bg_color,
    hud_padding,
)
from src.visualization.core.text import draw_text_box

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]  # OpenCV image (H, W, 3)
Position = tuple[int, int]  # (x, y) pixel coordinates


# =============================================================================
# FRAME COUNTER
# =============================================================================


def draw_frame_counter(
    frame: Frame,
    frame_idx: int,
    total_frames: int | None = None,
    position: Position = (10, 30),
) -> Frame:
    """Draw frame counter on frame.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        frame_idx: Current frame index (0-based).
        total_frames: Optional total frame count.
        position: (x, y) top-left position.

    Returns:
        Frame with frame counter drawn (modified in place).

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_frame_counter(frame, 100, 1000)
    """
    if total_frames is not None:
        text = f"Frame: {frame_idx}/{total_frames}"
    else:
        text = f"Frame: {frame_idx}"

    draw_text_box(
        frame,
        text,
        position,
        font_scale=font_scale,
        thickness=font_thickness,
        color=font_color,
        bg_color=hud_bg_color,
        bg_alpha=hud_bg_alpha,
        padding=hud_padding,
    )

    return frame


def draw_fps_counter(
    frame: Frame,
    fps: float,
    position: Position = (10, 30),
) -> Frame:
    """Draw FPS counter on frame.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        fps: Current FPS value.
        position: (x, y) top-left position.

    Returns:
        Frame with FPS counter drawn.

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_fps_counter(frame, 30.0)
    """
    text = f"FPS: {fps:.1f}"

    draw_text_box(
        frame,
        text,
        position,
        font_scale=font_scale,
        thickness=font_thickness,
        color=font_color,
        bg_color=hud_bg_color,
        bg_alpha=hud_bg_alpha,
        padding=hud_padding,
    )

    return frame


# =============================================================================
# METRICS PANEL
# =============================================================================


def draw_metrics_panel(
    frame: Frame,
    metrics: list[MetricResult],
    position: Position = (10, 30),
    title: str = "METRICS",
) -> Frame:
    """Draw metrics panel on frame.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        metrics: List of MetricResult to display.
        position: (x, y) top-left position.
        title: Panel title.

    Returns:
        Frame with metrics panel drawn.

    Example:
        >>> from src.types import MetricResult
        >>> metrics = [MetricResult(name="airtime", value=0.5, unit="s", is_good=True)]
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_metrics_panel(frame, metrics)
    """
    x, y = position

    # Draw title
    cv2.putText(
        frame,
        title,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        font_color,
        font_thickness,
        cv2.LINE_AA,
    )
    y += 25

    # Draw separator
    cv2.line(
        frame,
        (x, y),
        (x + 200, y),
        font_color,
        1,
        cv2.LINE_AA,
    )
    y += 15

    # Draw metrics
    for metric in metrics:
        # Format value
        if metric.unit == "score":
            value_str = f"{metric.value:.2f}"
        else:
            value_str = f"{metric.value:.2f} {metric.unit}"

        # Good/bad indicator
        indicator = "✓" if metric.is_good else "✗"
        color = (0, 255, 0) if metric.is_good else (0, 0, 255)

        # Draw metric name
        text = f"{metric.name}: {value_str} {indicator}"
        frame, (x, y) = draw_text_box(
            frame,
            text,
            (x, y),
            font_scale=font_scale * 0.9,
            thickness=1,
            color=color,
            bg_color=hud_bg_color,
            bg_alpha=hud_bg_alpha * 0.5,
            padding=hud_padding // 2,
        )

    return frame


# =============================================================================
# PHASE INDICATOR
# =============================================================================


def draw_phase_indicator(
    frame: Frame,
    phase: str,
    confidence: float | None = None,
    position: Position = (10, 30),
) -> Frame:
    """Draw element phase indicator on frame.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        phase: Current phase name (e.g., "takeoff", "flight", "landing").
        confidence: Optional confidence value [0, 1].
        position: (x, y) top-left position.

    Returns:
        Frame with phase indicator drawn.

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_phase_indicator(frame, "flight", 0.95)
    """
    x, y = position

    # Draw title
    cv2.putText(
        frame,
        "PHASE",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        font_color,
        font_thickness,
        cv2.LINE_AA,
    )
    y += 25

    # Draw separator
    cv2.line(
        frame,
        (x, y),
        (x + 150, y),
        font_color,
        1,
        cv2.LINE_AA,
    )
    y += 15

    # Draw phase
    text = f"Phase: {phase}"
    frame, (x, y) = draw_text_box(
        frame,
        text,
        (x, y),
        font_scale=font_scale,
        thickness=font_thickness,
        color=font_color,
        bg_color=hud_bg_color,
        bg_alpha=hud_bg_alpha,
        padding=hud_padding,
    )

    # Draw confidence if provided
    if confidence is not None:
        conf_text = f"Conf: {confidence:.2f}"
        draw_text_box(
            frame,
            conf_text,
            (x, y),
            font_scale=font_scale * 0.9,
            thickness=1,
            color=font_color,
            bg_color=hud_bg_color,
            bg_alpha=hud_bg_alpha * 0.5,
            padding=hud_padding // 2,
        )

    return frame


# =============================================================================
# BLADE INDICATOR
# =============================================================================


def draw_blade_indicator_hud(
    frame: Frame,
    blade_state: BladeState3D,
    position: Position = (10, 30),
    size: int = blade_indicator_size,
    thickness: int = blade_indicator_thickness,
) -> Frame:
    """Draw blade edge indicator on frame.

    Args:
        frame: OpenCV image (H, W, 3) BGR format.
        blade_state: BladeState3D with blade type and angles.
        position: (x, y) center position for indicator.
        size: Size of indicator arrow in pixels.
        thickness: Thickness of indicator lines.

    Returns:
        Frame with blade indicator drawn.

    Example:
        >>> from src.types import BladeState3D, BladeType, MotionDirection
        >>> state = BladeState3D(blade_type=BladeType.INSIDE, foot="left", motion_direction=MotionDirection.FORWARD,
        ...     foot_angle=-15.0, ankle_angle=90.0, knee_angle=120.0, vertical_accel=0.0,
        ...     position_3d=(0, 0, 0), velocity_3d=(0, 0, 0), confidence=0.8, frame_idx=0)
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_blade_indicator_hud(frame, state)
    """
    x, y = position

    # Get color for blade type
    color = _get_blade_color(blade_state.blade_type)

    # Draw directional arrow based on foot angle
    _draw_direction_arrow(frame, x, y, blade_state.foot_angle, size, thickness, color)

    # Draw blade type label
    label = blade_state.blade_type.name.lower()
    (text_width, text_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )

    label_x = x - text_width // 2
    label_y = y + size + 20

    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        font_thickness,
        cv2.LINE_AA,
    )

    # Draw foot angle value
    angle_text = f"{blade_state.foot_angle:.1f}°"
    (angle_width, angle_height), _ = cv2.getTextSize(
        angle_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.8,
        1,
    )

    angle_x = x - angle_width // 2
    angle_y = label_y + angle_height + 15

    cv2.putText(
        frame,
        angle_text,
        (angle_x, angle_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.8,
        font_color,
        1,
        cv2.LINE_AA,
    )

    return frame


def _get_blade_color(blade_type) -> tuple[int, int, int]:
    """Get color for blade type."""
    from src.types import BladeType

    color_map = {
        BladeType.INSIDE: blade_inside_color,
        BladeType.OUTSIDE: blade_outside_color,
        BladeType.FLAT: blade_flat_color,
        BladeType.TOE_PICK: blade_outside_color,
        BladeType.UNKNOWN: blade_unknown_color,
    }
    return color_map.get(blade_type, blade_unknown_color)


def _draw_direction_arrow(
    frame: Frame,
    x: int,
    y: int,
    angle: float,
    size: int,
    thickness: int,
    color: tuple[int, int, int],
) -> None:
    """Draw directional arrow indicating blade edge.

    Args:
        frame: OpenCV image.
        x, y: Center position.
        angle: Foot angle in degrees (negative = left/inside, positive = right/outside).
        size: Arrow size.
        thickness: Line thickness.
        color: BGR color.
    """
    import math

    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Calculate arrow points
    # Arrow points in direction of angle
    tip_x = int(x + size * math.sin(angle_rad))
    tip_y = int(y - size * math.cos(angle_rad))

    # Arrow base (wider part)
    base_x1 = int(x + size * 0.5 * math.sin(angle_rad + math.pi / 2))
    base_y1 = int(y - size * 0.5 * math.cos(angle_rad + math.pi / 2))

    base_x2 = int(x + size * 0.5 * math.sin(angle_rad - math.pi / 2))
    base_y2 = int(y - size * 0.5 * math.cos(angle_rad - math.pi / 2))

    # Draw arrow
    pts = np.array([[tip_x, tip_y], [base_x1, base_y1], [base_x2, base_y2]], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, color)

    # Draw outline
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def draw_info_text(
    frame: Frame,
    lines: list[str],
    position: Position = (10, 30),
    line_spacing: int = 20,
) -> Position:
    """Draw multiple lines of info text.

    Args:
        frame: OpenCV image.
        lines: List of text lines.
        position: Starting (x, y) position.
        line_spacing: Spacing between lines.

    Returns:
        Next position after last line.

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_info_text(frame, ["Line 1", "Line 2"], (10, 30))
        (10, 70)  # Next position
    """
    x, y = position

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA,
        )
        y += line_spacing

    return (x, y)


def draw_warning(
    frame: Frame,
    message: str,
    position: Position = (10, 30),
) -> Frame:
    """Draw warning message on frame.

    Args:
        frame: OpenCV image.
        message: Warning message text.
        position: (x, y) top-left position.

    Returns:
        Frame with warning drawn.

    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> draw_warning(frame, "Low confidence!")
    """
    # Yellow background
    bg_color = (0, 255, 255)

    draw_text_box(
        frame,
        f"⚠ {message}",
        position,
        font_scale=font_scale,
        thickness=font_thickness,
        color=(0, 0, 0),  # Black text
        bg_color=bg_color,
        bg_alpha=0.8,
        padding=hud_padding,
    )

    return frame
