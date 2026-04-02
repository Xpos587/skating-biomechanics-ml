"""Joint angle visualization layer.

Draws angle arcs at key joints (knees, elbows, hips, shoulders)
with degree labels and color-coded feedback.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.pose_estimation import H36Key
from src.utils.geometry import angle_3pt
from src.visualization.config import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    LayerConfig,
    VisualizationConfig,
)
from src.visualization.core.geometry import normalized_to_pixel
from src.visualization.layers.base import Frame, Layer, LayerContext


class JointAngleSpec:
    """Specification for drawing an angle at a joint.

    Attributes:
        name: Joint name for display.
        point_a: H36Key index for first point.
        vertex: H36Key index for vertex (joint).
        point_c: H36Key index for third point.
        color: Drawing color (BGR).
        arc_radius: Radius of the angle arc in pixels.
        good_range: (min, max) degrees for green color.
        warn_range: (min, max) degrees for yellow color.
    """

    __slots__ = (
        "arc_radius",
        "color",
        "good_range",
        "name",
        "point_a",
        "point_c",
        "vertex",
        "warn_range",
    )

    def __init__(
        self,
        name: str,
        point_a: int,
        vertex: int,
        point_c: int,
        color: tuple[int, int, int] = COLOR_CYAN,
        arc_radius: int = 25,
        good_range: tuple[float, float] = (90, 180),
        warn_range: tuple[float, float] = (60, 190),
    ):
        self.name = name
        self.point_a = point_a
        self.vertex = vertex
        self.point_c = point_c
        self.color = color
        self.arc_radius = arc_radius
        self.good_range = good_range
        self.warn_range = warn_range

    def get_color_for_angle(self, angle: float) -> tuple[int, int, int]:
        """Return color based on angle quality."""
        lo_g, hi_g = self.good_range
        lo_w, hi_w = self.warn_range
        if lo_g <= angle <= hi_g:
            return COLOR_GREEN
        if lo_w <= angle <= hi_w:
            return COLOR_YELLOW
        return COLOR_RED


# Default joint specs for H3.6M 17-keypoint format
DEFAULT_JOINT_SPECS: list[JointAngleSpec] = [
    JointAngleSpec(
        "L Knee",
        H36Key.LHIP,
        H36Key.LKNEE,
        H36Key.LFOOT,
        COLOR_CYAN,
        12,
        good_range=(90, 170),
    ),
    JointAngleSpec(
        "R Knee",
        H36Key.RHIP,
        H36Key.RKNEE,
        H36Key.RFOOT,
        COLOR_CYAN,
        12,
        good_range=(90, 170),
    ),
    JointAngleSpec(
        "L Elbow",
        H36Key.LSHOULDER,
        H36Key.LELBOW,
        H36Key.LWRIST,
        COLOR_YELLOW,
        10,
        good_range=(30, 160),
    ),
    JointAngleSpec(
        "R Elbow",
        H36Key.RSHOULDER,
        H36Key.RELBOW,
        H36Key.RWRIST,
        COLOR_YELLOW,
        10,
        good_range=(30, 160),
    ),
    JointAngleSpec(
        "L Hip",
        H36Key.THORAX,
        H36Key.LHIP,
        H36Key.LKNEE,
        (255, 165, 0),  # Orange
        10,
        good_range=(80, 140),
    ),
    JointAngleSpec(
        "R Hip",
        H36Key.THORAX,
        H36Key.RHIP,
        H36Key.RKNEE,
        (255, 165, 0),  # Orange
        10,
        good_range=(80, 140),
    ),
]


class JointAngleLayer(Layer):
    """Draw angle arcs at key joints with degree labels.

    Uses H3.6M 17-keypoint format. Each joint shows:
    - Angle arc colored by quality (green/yellow/red)
    - Degree value label
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        viz_config: VisualizationConfig | None = None,
        joints: list[JointAngleSpec] | None = None,
    ):
        super().__init__(config=config or LayerConfig(enabled=True, z_index=6))
        self.viz = viz_config or VisualizationConfig()
        self.joints = joints or DEFAULT_JOINT_SPECS

    def render(self, frame: Frame, context: LayerContext) -> Frame:
        pose = context.pose_2d
        if pose is None:
            return frame

        w, h = context.frame_width, context.frame_height

        for spec in self.joints:
            # Prefer 3D angles (more accurate via kinematic constraints)
            angle = None
            if context.pose_3d is not None:
                a3 = context.pose_3d[spec.point_a]
                v3 = context.pose_3d[spec.vertex]
                c3 = context.pose_3d[spec.point_c]
                if not (np.isnan(a3).any() or np.isnan(v3).any() or np.isnan(c3).any()):
                    angle = angle_3pt(a3, v3, c3)

            # Fallback to 2D
            if angle is None:
                if context.normalized:
                    pa = normalized_to_pixel(pose[spec.point_a], w, h)
                    pv = normalized_to_pixel(pose[spec.vertex], w, h)
                    pc = normalized_to_pixel(pose[spec.point_c], w, h)
                else:
                    pa = pose[spec.point_a].astype(int)
                    pv = pose[spec.vertex].astype(int)
                    pc = pose[spec.point_c].astype(int)
                a = np.array(pa, dtype=np.float64)
                v = np.array(pv, dtype=np.float64)
                c = np.array(pc, dtype=np.float64)
                angle = angle_3pt(a, v, c)
            # 3D angle computed — still need 2D positions for arc placement
            elif context.normalized:
                pv = normalized_to_pixel(pose[spec.vertex], w, h)
                pa = normalized_to_pixel(pose[spec.point_a], w, h)
                pc = normalized_to_pixel(pose[spec.point_c], w, h)
            else:
                pa = pose[spec.point_a].astype(int)
                pv = pose[spec.vertex].astype(int)
                pc = pose[spec.point_c].astype(int)

            # Skip NaN/invalid angles
            if np.isnan(angle) or angle < 0 or angle > 360:
                continue

            color = spec.get_color_for_angle(angle)
            _vx, _vy = int(pv[0]), int(pv[1])

            # Draw subtle angle arc (thin line, no label)
            self._draw_arc(frame, pv, pa, pc, spec.arc_radius, color)

        return frame

    @staticmethod
    def _draw_arc(
        frame: Frame,
        vertex: np.ndarray,
        point_a: np.ndarray,
        point_c: np.ndarray,
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw an angle arc at vertex between point_a and point_c."""
        vx, vy = vertex
        # Angles in standard math coords (y-up), but OpenCV has y-down
        angle_a = math.degrees(math.atan2(-(point_a[1] - vy), point_a[0] - vx))
        angle_c = math.degrees(math.atan2(-(point_c[1] - vy), point_c[0] - vx))

        # Normalize to [0, 360)
        angle_a = angle_a % 360
        angle_c = angle_c % 360

        # Draw the shorter arc
        diff = (angle_c - angle_a) % 360
        if diff > 180:
            start, end = angle_c, angle_a
        else:
            start, end = angle_a, angle_c

        cv2.ellipse(
            frame,
            (int(vx), int(vy)),
            (radius, radius),
            0,
            start,
            end,
            color,
            1,
        )
