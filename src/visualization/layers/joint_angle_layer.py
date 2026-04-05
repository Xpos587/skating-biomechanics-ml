"""Joint angle visualization layer.

Draws angle arcs at key joints (knees, elbows, hips, shoulders)
with degree labels and color-coded feedback.

Features:
- Adaptive arc radius scaled to bone length
- 3D→2D arc projection (physically correct foreshortening from 3D skeleton)
- Tick marks connecting arc endpoints to skeleton bones
- Degree labels positioned along arc bisector
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from numpy.typing import NDArray


class JointAngleSpec:
    """Specification for drawing an angle at a joint.

    Attributes:
        name: Joint name for display.
        point_a: H36Key index for first point.
        vertex: H36Key index for vertex (joint).
        point_c: H36Key index for third point.
        color: Drawing color (BGR).
        min_radius: Minimum arc radius in pixels (adaptive scaling below this).
        good_range: (min, max) degrees for green color.
        warn_range: (min, max) degrees for yellow color.
    """

    __slots__ = (
        "color",
        "good_range",
        "min_radius",
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
        min_radius: int = 4,
        good_range: tuple[float, float] = (90, 180),
        warn_range: tuple[float, float] = (60, 190),
    ):
        self.name = name
        self.point_a = point_a
        self.vertex = vertex
        self.point_c = point_c
        self.color = color
        self.min_radius = min_radius
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
        min_radius=6,
        good_range=(90, 170),
    ),
    JointAngleSpec(
        "R Knee",
        H36Key.RHIP,
        H36Key.RKNEE,
        H36Key.RFOOT,
        COLOR_CYAN,
        min_radius=6,
        good_range=(90, 170),
    ),
    JointAngleSpec(
        "L Elbow",
        H36Key.LSHOULDER,
        H36Key.LELBOW,
        H36Key.LWRIST,
        COLOR_YELLOW,
        min_radius=4,
        good_range=(30, 160),
    ),
    JointAngleSpec(
        "R Elbow",
        H36Key.RSHOULDER,
        H36Key.RELBOW,
        H36Key.RWRIST,
        COLOR_YELLOW,
        min_radius=4,
        good_range=(30, 160),
    ),
    JointAngleSpec(
        "L Hip",
        H36Key.THORAX,
        H36Key.LHIP,
        H36Key.LKNEE,
        (255, 165, 0),  # Orange
        min_radius=4,
        good_range=(80, 140),
    ),
    JointAngleSpec(
        "R Hip",
        H36Key.THORAX,
        H36Key.RHIP,
        H36Key.RKNEE,
        (255, 165, 0),  # Orange
        min_radius=4,
        good_range=(80, 140),
    ),
]


class JointAngleLayer(Layer):
    """Draw angle arcs at key joints with degree labels.

    Uses H3.6M 17-keypoint format. Each joint shows:
    - Angle arc colored by quality (green/yellow/red)
    - Arc radius adapted to bone length (25% of shorter bone)
    - Tick marks at arc endpoints connecting to skeleton bones
    - 3D→2D projected arc when 3D poses are available
    - Degree value label along arc bisector
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        viz_config: VisualizationConfig | None = None,
        joints: list[JointAngleSpec] | None = None,
        show_degree_labels: bool = True,
        angle_source: str = "auto",
        arc_scale: float = 0.25,
    ):
        if angle_source not in ("auto", "2d", "3d"):
            raise ValueError(f"angle_source must be 'auto', '2d', or '3d', got '{angle_source}'")
        super().__init__(config=config or LayerConfig(enabled=True, z_index=6))
        self.viz = viz_config or VisualizationConfig()
        self.joints = joints or DEFAULT_JOINT_SPECS
        self.show_degree_labels = show_degree_labels
        self.angle_source = angle_source
        self.arc_scale = arc_scale  # fraction of shorter bone length

    def render(self, frame: Frame, context: LayerContext) -> Frame:
        pose = context.pose_2d
        if pose is None:
            return frame

        w, h = context.frame_width, context.frame_height

        for spec in self.joints:
            # Determine angle source
            angle = None
            use_3d = self.angle_source == "3d" or (
                self.angle_source == "auto" and context.pose_3d is not None
            )

            a3 = v3 = c3 = None
            if use_3d and context.pose_3d is not None:
                a3 = context.pose_3d[spec.point_a]
                v3 = context.pose_3d[spec.vertex]
                c3 = context.pose_3d[spec.point_c]
                if not (np.isnan(a3).any() or np.isnan(v3).any() or np.isnan(c3).any()):
                    angle = angle_3pt(a3, v3, c3)

            # Get 2D positions (pixel coords)
            if context.normalized:
                pa = np.array(normalized_to_pixel(pose[spec.point_a], w, h), dtype=np.float64)
                pv = np.array(normalized_to_pixel(pose[spec.vertex], w, h), dtype=np.float64)
                pc = np.array(normalized_to_pixel(pose[spec.point_c], w, h), dtype=np.float64)
            else:
                pa = pose[spec.point_a].astype(np.float64)
                pv = pose[spec.vertex].astype(np.float64)
                pc = pose[spec.point_c].astype(np.float64)

            # Fallback to 2D angle
            if angle is None:
                angle = angle_3pt(pa, pv, pc)

            # Skip NaN/invalid angles or out-of-bounds vertices
            if np.isnan(angle) or angle < 0 or angle > 360:
                continue
            if pv[0] < -1000 or pv[1] < -1000 or pv[0] > w + 1000 or pv[1] > h + 1000:
                continue

            # Adaptive radius: scale to shorter bone length
            bone_a = np.linalg.norm(pa - pv)
            bone_c = np.linalg.norm(pc - pv)
            radius = max(int(min(bone_a, bone_c) * self.arc_scale), spec.min_radius)

            color = spec.get_color_for_angle(angle)

            # Draw arc — 3D projected or 2D ellipse
            if use_3d and a3 is not None and v3 is not None and c3 is not None:
                arc_pts = self._project_3d_arc_2d(a3, v3, c3, pa, pv, pc, radius)
                if arc_pts is not None:
                    self._draw_projected_arc(frame, arc_pts, (220, 220, 220))
                else:
                    self._draw_arc(frame, pv, pa, pc, radius, (220, 220, 220))
            else:
                self._draw_arc(frame, pv, pa, pc, radius, (220, 220, 220))

            # Tick marks at arc endpoints (visual connection to bones)
            self._draw_tick(frame, pv, pa, radius, color)
            self._draw_tick(frame, pv, pc, radius, color)

            # Degree label along bisector
            if self.show_degree_labels:
                bisector = self._compute_bisector(pv, pa, pc)
                self._draw_degree_label(frame, pv, bisector, angle, color, radius)

        return frame

    # ------------------------------------------------------------------
    # 3D → 2D arc projection
    # ------------------------------------------------------------------

    @staticmethod
    def _project_3d_arc_2d(
        a_3d: NDArray[np.float64],
        v_3d: NDArray[np.float64],
        c_3d: NDArray[np.float64],
        a_2d: NDArray[np.float64],
        v_2d: NDArray[np.float64],
        c_2d: NDArray[np.float64],
        radius: float,
        n_points: int = 24,
    ) -> NDArray[np.float64] | None:
        """Project a 3D angle arc onto the 2D image plane.

        Generates arc samples in the 3D plane defined by the three joints,
        then maps each sample to 2D using a local affine transform derived
        from the (3D → 2D) keypoint correspondences.

        Returns:
            Array of shape (n_points+1, 2) with 2D pixel positions, or None
            if the joints are degenerate (collinear / zero-length bones).
        """
        va = a_3d - v_3d
        vc = c_3d - v_3d
        va_len = np.linalg.norm(va)
        vc_len = np.linalg.norm(vc)
        if va_len < 1e-6 or vc_len < 1e-6:
            return None

        # Local 2D basis in the 3D joint plane
        e1 = va / va_len
        vc_hat = vc / vc_len
        normal = np.cross(e1, vc_hat)
        normal_len = np.linalg.norm(normal)
        if normal_len < 1e-6:
            return None  # collinear in 3D
        normal /= normal_len
        e2 = np.cross(normal, e1)

        # True 3D angle and arc direction
        cos_angle = np.clip(np.dot(e1, vc_hat), -1.0, 1.0)
        sweep = np.arccos(cos_angle)
        if np.dot(np.cross(va, vc), normal) < 0:
            sweep = 2 * np.pi - sweep

        # Sample arc in the local 2D frame of the joint plane
        thetas = np.linspace(0, sweep, n_points + 1)
        arc_local = np.column_stack([radius * np.cos(thetas), radius * np.sin(thetas)])

        # Affine map: local-2D → image-2D
        #   a_local = (va_len, 0)       → a_2d - v_2d
        #   c_local = (vc·e1, vc·e2)   → c_2d - v_2d
        a_local = np.array([va_len, 0.0])
        c_local = np.array([np.dot(vc, e1), np.dot(vc, e2)])

        src = np.column_stack([a_local, c_local])  # 2×2
        dst = np.column_stack([a_2d - v_2d, c_2d - v_2d])  # 2×2

        if abs(np.linalg.det(src)) < 1e-6:
            return None

        M = dst @ np.linalg.inv(src)

        # Map all arc samples to image coordinates
        arc_2d = (M @ arc_local.T).T + v_2d
        return arc_2d

    @staticmethod
    def _draw_projected_arc(
        frame: Frame,
        points: NDArray[np.float64],
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        """Draw a projected 3D arc as an anti-aliased polyline."""
        pts = points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 2D arc (fallback / when no 3D data)
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_arc(
        frame: Frame,
        vertex: NDArray[np.float64],
        point_a: NDArray[np.float64],
        point_c: NDArray[np.float64],
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw an angle arc at vertex between point_a and point_c."""
        vx, vy = vertex
        # OpenCV y-down → negate for standard math angle
        angle_a = math.degrees(math.atan2(-(point_a[1] - vy), point_a[0] - vx))
        angle_c = math.degrees(math.atan2(-(point_c[1] - vy), point_c[0] - vx))

        angle_a %= 360
        angle_c %= 360

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
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Tick marks — connect arc endpoints to skeleton bones
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_tick(
        frame: Frame,
        vertex: NDArray[np.float64],
        bone_end: NDArray[np.float64],
        radius: int,
        color: tuple[int, int, int],
        tick_len: int = 5,
    ) -> None:
        """Draw a small tick mark at the arc endpoint along the bone direction."""
        direction = bone_end - vertex
        length = np.linalg.norm(direction)
        if length < 1e-3:
            return
        d = direction / length

        p1 = vertex + d * (radius - 1)
        p2 = vertex + d * (radius + tick_len)
        cv2.line(
            frame,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            color,
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Degree label
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bisector(
        vertex: NDArray[np.float64],
        pa: NDArray[np.float64],
        pc: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute unit bisector direction of the angle at vertex."""
        da = pa - vertex
        dc = pc - vertex
        la = np.linalg.norm(da)
        lc = np.linalg.norm(dc)
        if la < 1e-3 or lc < 1e-3:
            return np.array([1.0, -1.0])  # fallback: up-right
        bisector = da / la + dc / lc
        bl = np.linalg.norm(bisector)
        if bl < 1e-3:
            return np.array([1.0, -1.0])
        return bisector / bl

    @staticmethod
    def _draw_degree_label(
        frame: Frame,
        vertex: NDArray[np.float64],
        bisector: NDArray[np.float64],
        angle: float,
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        """Draw degree label along the arc bisector direction."""
        from src.visualization.core.text import draw_text_outlined

        vx, vy = int(vertex[0]), int(vertex[1])
        label = f"{angle:.0f}\u00b0"

        offset = radius + 14
        pos = (int(vx + bisector[0] * offset), int(vy + bisector[1] * offset))
        draw_text_outlined(frame, label, pos, font_scale=0.55, thickness=1, color=color)
