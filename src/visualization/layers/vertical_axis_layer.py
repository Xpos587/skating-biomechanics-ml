"""Vertical axis and trunk tilt visualization layer.

Draws:
- Dashed vertical reference line through mid-hip
- Actual spine axis (mid-hip to mid-shoulder)
- Angle arc between vertical and spine axis
- Degree label showing tilt angle
"""

import math

import cv2
import numpy as np

from src.pose_estimation import H36Key
from src.utils.geometry import angle_3pt
from src.visualization.config import COLOR_YELLOW, LayerConfig, VisualizationConfig
from src.visualization.core.geometry import normalized_to_pixel
from src.visualization.layers.base import Frame, Layer, LayerContext


class VerticalAxisLayer(Layer):
    """Draw vertical reference axis and trunk tilt angle.

    Renders a dashed vertical line through the athlete's mid-hip center,
    the actual spine axis from mid-hip to mid-shoulder, an arc showing
    the tilt angle, and a degree label.
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        viz_config: VisualizationConfig | None = None,
    ):
        super().__init__(config=config or LayerConfig(enabled=True, z_index=5))
        self.viz = viz_config or VisualizationConfig()

    def render(self, frame: Frame, context: LayerContext) -> Frame:
        pose = context.pose_2d
        if pose is None:
            return frame

        w, h = context.frame_width, context.frame_height

        # Get mid-hip and mid-shoulder in pixels
        if context.normalized:
            mid_hip = normalized_to_pixel(
                (pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2, w, h
            )
            mid_shoulder = normalized_to_pixel(
                (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2, w, h
            )
        else:
            mid_hip = ((pose[H36Key.LHIP] + pose[H36Key.RHIP]) / 2).astype(int)
            mid_shoulder = (
                (pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER]) / 2
            ).astype(int)

        hip_x, hip_y = int(mid_hip[0]), int(mid_hip[1])
        sh_x, sh_y = int(mid_shoulder[0]), int(mid_shoulder[1])

        # Vertical line length (extend both up and down from hip)
        line_len = int(h * 0.4)
        vert_top = (hip_x, hip_y - line_len)
        vert_bottom = (hip_x, hip_y + line_len // 2)

        # Draw dashed vertical line
        self._draw_dashed_line(frame, vert_top, vert_bottom, COLOR_YELLOW, 1, dash=10)

        # Draw spine axis
        cv2.line(frame, (hip_x, hip_y), (sh_x, sh_y), COLOR_YELLOW, 1)

        # Calculate tilt angle using angle_3pt
        # Point above hip on vertical, hip (vertex), shoulder
        vert_point = np.array([hip_x, hip_y - 100], dtype=np.float64)
        hip_point = np.array([hip_x, hip_y], dtype=np.float64)
        shoulder_point = np.array([sh_x, sh_y], dtype=np.float64)

        tilt_angle = angle_3pt(vert_point, hip_point, shoulder_point)

        # Draw angle arc (subtle, no label)
        if tilt_angle > 1.0:
            self._draw_angle_arc(frame, hip_x, hip_y, sh_x, sh_y, tilt_angle)

        return frame

    def _draw_dashed_line(
        self,
        frame: Frame,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        dash: int = 10,
    ) -> None:
        """Draw a dashed line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1:
            return
        dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
        drawn = 0
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
        angle_deg: float,  # noqa: ARG002
        radius: int = 20,
    ) -> None:
        """Draw angle arc from vertical to spine axis direction."""
        # Angle of spine axis in OpenCV coords (y-axis inverted)
        spine_angle = math.atan2(-(target_y - cy), target_x - cx)
        # Vertical up in OpenCV is -90 degrees in standard math, but atan2 uses image coords
        vert_angle = math.atan2(-1, 0)  # straight up = -pi/2

        # cv2.ellipse angles are in degrees, measured clockwise from 3 o'clock
        # We need to convert our angles
        start_angle = math.degrees(vert_angle)  # -90
        end_angle = math.degrees(spine_angle)

        # Ensure we draw the smaller arc
        if abs(end_angle - start_angle) > 180:
            if end_angle > start_angle:
                start_angle += 360
            else:
                end_angle += 360

        cv2.ellipse(
            frame,
            (cx, cy),
            (radius, radius),
            0,
            min(start_angle, end_angle),
            max(start_angle, end_angle),
            COLOR_YELLOW,
            1,
        )
