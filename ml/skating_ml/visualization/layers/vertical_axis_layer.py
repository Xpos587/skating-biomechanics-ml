"""Vertical axis and trunk tilt visualization layer.

Draws:
- Dashed vertical gravity reference line through mid-hip
- Actual spine axis from mid-hip to mid-shoulder (color-coded by tilt quality)
- Angle arc between vertical and spine axis
- Degree label showing tilt angle with direction (L/R)
- Head alignment indicator showing lateral offset from spine line
"""

import math
from enum import Enum

import cv2
import numpy as np

from skating_ml.pose_estimation import H36Key
from skating_ml.visualization.config import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    LayerConfig,
    VisualizationConfig,
)
from skating_ml.visualization.core.geometry import normalized_to_pixel
from skating_ml.visualization.layers.base import Frame, Layer, LayerContext


class TiltQuality(Enum):
    """Classification of trunk tilt severity."""

    GOOD = "good"
    WARN = "warn"
    BAD = "bad"


def classify_tilt(
    tilt_deg: float,
    good_threshold: float = 5.0,
    warn_threshold: float = 10.0,
) -> TiltQuality:
    """Classify trunk tilt angle into quality levels.

    Args:
        tilt_deg: Trunk tilt in degrees (positive = right lean, negative = left).
        good_threshold: Maximum absolute tilt for GOOD quality.
        warn_threshold: Maximum absolute tilt for WARN quality.

    Returns:
        TiltQuality enum value.
    """
    abs_tilt = abs(tilt_deg)
    if abs_tilt < good_threshold:
        return TiltQuality.GOOD
    if abs_tilt < warn_threshold:
        return TiltQuality.WARN
    return TiltQuality.BAD


def _tilt_direction_label(tilt_deg: float) -> str:
    """Return direction label for tilt angle.

    Args:
        tilt_deg: Tilt angle in degrees.

    Returns:
        "R" for right lean, "L" for left lean, "" if negligible.
    """
    if abs(tilt_deg) < 1.0:
        return ""
    return "R" if tilt_deg > 0 else "L"


class VerticalAxisLayer(Layer):
    """Draw vertical reference axis, spine axis, and trunk tilt angle.

    Renders a dashed vertical gravity reference line through the athlete's
    mid-hip center, the actual spine axis from mid-hip to mid-shoulder
    (color-coded by tilt quality), an arc showing the tilt angle, a degree
    label with direction, and an optional head alignment indicator.
    """

    def __init__(
        self,
        config=None,
        viz_config=None,
        *,
        show_degree_label=True,
        show_head_alignment=True,
        good_threshold=5.0,
        warn_threshold=10.0,
        arc_radius=20,
    ):
        super().__init__(config=config or LayerConfig(enabled=True, z_index=5))
        self.viz = viz_config or VisualizationConfig()
        self.show_degree_label = show_degree_label
        self.show_head_alignment = show_head_alignment
        self.good_threshold = good_threshold
        self.warn_threshold = warn_threshold
        self.arc_radius = arc_radius

    def render(self, frame: Frame, context: LayerContext) -> Frame:
        pose = context.pose_2d
        if pose is None:
            return frame

        # Bail out if any required joint contains NaN
        required_keys = (H36Key.LHIP, H36Key.RHIP, H36Key.LSHOULDER, H36Key.RSHOULDER)
        if any(np.isnan(pose[k]).any() for k in required_keys):
            return frame

        w, h = context.frame_width, context.frame_height

        # Compute mid-hip, mid-shoulder, head in pixel coordinates
        if context.normalized:
            mid_hip = np.asarray(
                normalized_to_pixel((pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2, w, h),
                dtype=np.float64,
            )
            mid_shoulder = np.asarray(
                normalized_to_pixel((pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2, w, h),
                dtype=np.float64,
            )
            head = np.asarray(
                normalized_to_pixel(pose[H36Key.HEAD], w, h),
                dtype=np.float64,
            )
        else:
            mid_hip = ((pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2).astype(np.float64)
            mid_shoulder = ((pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2).astype(
                np.float64
            )
            head = pose[H36Key.HEAD].astype(np.float64)

        # Spine vector (use only first 2 elements to strip confidence if present)
        spine_vector = mid_shoulder[:2] - mid_hip[:2]

        # Tilt angle: same convention as compute_trunk_lean in metrics.py
        tilt_deg = float(np.degrees(np.arctan2(spine_vector[0], -spine_vector[1])))

        # Classify tilt quality and pick color
        quality = classify_tilt(tilt_deg, self.good_threshold, self.warn_threshold)
        quality_color_map = {
            TiltQuality.GOOD: COLOR_GREEN,
            TiltQuality.WARN: COLOR_YELLOW,
            TiltQuality.BAD: COLOR_RED,
        }
        quality_color = quality_color_map[quality]

        # Adaptive line lengths from torso height
        torso_height = max(np.linalg.norm(spine_vector), 1.0)
        line_ext = int(torso_height * 0.4)

        hip_x, hip_y = int(mid_hip[0]), int(mid_hip[1])
        sh_x, sh_y = int(mid_shoulder[0]), int(mid_shoulder[1])

        # 1. Draw gravity reference line (dashed, through mid-hip vertical)
        vert_top = (hip_x, hip_y - line_ext)
        vert_bottom = (hip_x, hip_y + line_ext // 2)
        self._draw_dashed_line(frame, vert_top, vert_bottom, (200, 200, 100), 1, dash=8)

        # 2. Draw spine axis (solid, quality-colored)
        cv2.line(frame, (hip_x, hip_y), (sh_x, sh_y), quality_color, 2)

        # 3. Draw angle arc if tilt is significant
        if abs(tilt_deg) > 1.0:
            self._draw_angle_arc(frame, hip_x, hip_y, sh_x, sh_y, tilt_deg, quality_color)

        # 4. Draw degree label if enabled and tilt is notable
        if self.show_degree_label and abs(tilt_deg) > 0.5:
            self._draw_degree_label(frame, hip_x, hip_y, tilt_deg, quality_color)

        # 5. Draw head alignment indicator
        if self.show_head_alignment:
            self._draw_head_alignment(frame, mid_hip, mid_shoulder, head, spine_vector)

        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_dashed_line(
        self,
        frame: Frame,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        dash: int = 8,
    ) -> None:
        """Draw a dashed line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1:
            return
        dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
        drawn = 0.0
        while drawn < dist:
            seg_end = min(drawn + dash, dist)
            sx = int(x1 + dx * drawn)
            sy = int(y1 + dy * drawn)
            ex = int(x1 + dx * seg_end)
            ey = int(y1 + dy * seg_end)
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness)
            drawn += dash * 2  # dash then gap

    def _draw_angle_arc(
        self,
        frame: Frame,
        cx: int,
        cy: int,
        target_x: int,
        target_y: int,
        tilt_deg: float,
        color: tuple[int, int, int],
        radius: int | None = None,
    ) -> None:
        """Draw angle arc from vertical to spine direction using cv2.ellipse."""
        if radius is None:
            radius = self.arc_radius

        # Spine direction in cv2.ellipse coordinate system
        # cv2.ellipse angles: 0 = 3 o'clock (right), increasing clockwise
        spine_angle = math.degrees(math.atan2(target_y - cy, target_x - cx))
        # Vertical up in cv2.ellipse coordinates = 270 degrees
        vert_angle = 270.0

        start = vert_angle
        end = spine_angle

        # Always draw the short arc (< 180 degrees)
        diff = (end - start) % 360
        if diff > 180:
            start, end = end, start

        cv2.ellipse(
            frame,
            (cx, cy),
            (radius, radius),
            0,
            start,
            end,
            color,
            1,
        )

    def _draw_degree_label(
        self,
        frame: Frame,
        hip_x: int,
        hip_y: int,
        tilt_deg: float,
        color: tuple[int, int, int],
    ) -> None:
        """Draw degree label with direction indicator."""
        from skating_ml.visualization.core.text import draw_text_outlined

        abs_tilt = abs(tilt_deg)
        direction = _tilt_direction_label(tilt_deg)
        if direction:
            label = f"{abs_tilt:.1f}\u00b0 {direction}"
        else:
            label = f"{abs_tilt:.1f}\u00b0"

        # Position label to the right of hip, offset upward
        draw_text_outlined(
            frame,
            label,
            (hip_x + 15, hip_y - 10),
            font_scale=0.5,
            thickness=1,
            color=color,
        )

    def _draw_head_alignment(
        self,
        frame: Frame,
        mid_hip: np.ndarray,
        mid_shoulder: np.ndarray,
        head: np.ndarray,
        spine_vector: np.ndarray,
    ) -> None:
        """Draw head offset indicator relative to spine line.

        Projects the head position onto the spine line. If the perpendicular
        offset exceeds 3 pixels, draws a thin indicator line.
        """
        hip = mid_hip[:2]
        shoulder = mid_shoulder[:2]
        head_pt = head[:2]

        spine_len_sq = float(np.dot(spine_vector, spine_vector))
        if spine_len_sq < 1.0:
            return

        # Project head onto spine line
        t = float(np.dot(head_pt - hip, spine_vector) / spine_len_sq)
        t = max(0.0, min(1.0, t))
        projection = hip + t * spine_vector

        offset = float(np.linalg.norm(head_pt - projection))
        if offset < 3.0:
            return

        # Draw thin line from head to its projection on spine
        hx, hy = int(head_pt[0]), int(head_pt[1])
        px, py = int(projection[0]), int(projection[1])
        cv2.line(frame, (hx, hy), (px, py), (180, 130, 255), 1)
