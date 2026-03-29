"""Velocity vectors layer.

Renders velocity vectors on video frames showing joint motion.
"""

from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from src.types import H36Key
from src.visualization.config import (
    LayerConfig,
)
from src.visualization.core.colors import get_heatmap_color
from src.visualization.core.geometry import (
    normalized_to_pixel,
    project_3d_to_2d,
)
from src.visualization.layers.base import Layer, LayerContext

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]
Pose2D = NDArray[np.float32]
Pose3D = NDArray[np.float32]


# =============================================================================
# VELOCITY LAYER
# =============================================================================


class VelocityLayer(Layer):
    """Layer for drawing velocity vectors.

    Shows motion direction and speed for key joints.

    Attributes:
        config: LayerConfig for this layer.
        scale: Scaling factor for vector length.
        max_length: Maximum vector length in pixels.
        color_mode: "solid", "heatmap", or "depth" coloring.
        joints: List of joint indices to show velocity for.
    """

    # Key joints for velocity visualization
    KEY_JOINTS: Final[frozenset[int]] = frozenset(
        [
            H36Key.LWRIST,
            H36Key.RWRIST,
            H36Key.LFOOT,
            H36Key.RFOOT,
        ]
    )

    def __init__(
        self,
        config: LayerConfig | None = None,
        scale: float = 5.0,
        max_length: int = 50,
        color_mode: str = "heatmap",
        joints: frozenset[int] | None = None,
    ):
        """Initialize velocity layer.

        Args:
            config: LayerConfig for this layer.
            scale: Scaling factor for vector length.
            max_length: Maximum vector length in pixels.
            color_mode: "solid", "heatmap", or "depth".
            joints: Joint indices to show velocity for (default: wrists and feet).
        """
        super().__init__(config or LayerConfig(z_index=1), name="Velocity")
        self.scale = scale
        self.max_length = max_length
        self.color_mode = color_mode
        self.joints = joints or self.KEY_JOINTS

        # Previous poses for velocity calculation
        self._prev_pose_2d: Pose2D | None = None
        self._prev_pose_3d: Pose3D | None = None

    def render(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> Frame:
        """Render velocity vectors to frame.

        Args:
            frame: OpenCV image (H, W, 3) BGR format.
            context: LayerContext with pose data.

        Returns:
            Frame with velocity vectors.
        """
        # Calculate velocities
        if context.pose_3d is not None and self._prev_pose_3d is not None:
            self._draw_velocity_3d(frame, context)
        elif context.pose_2d is not None and self._prev_pose_2d is not None:
            self._draw_velocity_2d(frame, context)

        # Store current poses for next frame
        self._prev_pose_2d = context.pose_2d.copy() if context.pose_2d is not None else None
        self._prev_pose_3d = context.pose_3d.copy() if context.pose_3d is not None else None

        return frame

    def _draw_velocity_2d(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> None:
        """Draw 2D velocity vectors."""
        if context.pose_2d is None or self._prev_pose_2d is None:
            return

        # Calculate velocities
        velocities = (context.pose_2d - self._prev_pose_2d) * self.scale

        # Get pixel coordinates
        if context.normalized:
            pose_px = normalized_to_pixel(
                context.pose_2d, context.frame_width, context.frame_height
            )
            vel_px = normalized_to_pixel(velocities, context.frame_width, context.frame_height)
        else:
            pose_px = context.pose_2d.astype(np.int32)
            vel_px = velocities.astype(np.int32)

        # Draw vectors
        for joint_idx in self.joints:
            start = tuple(pose_px[joint_idx])
            end_raw = pose_px[joint_idx] + vel_px[joint_idx]

            # Clamp vector length
            vector = end_raw - pose_px[joint_idx]
            length = np.linalg.norm(vector)
            if length > self.max_length:
                vector = vector / length * self.max_length

            end = tuple((pose_px[joint_idx] + vector).astype(int))

            # Get color
            if self.color_mode == "heatmap":
                speed = length / self.max_length
                color = get_heatmap_color(speed, 0.0, 1.0, "jet")
            else:
                color = (0, 255, 255)  # Cyan

            # Draw arrow
            cv2.arrowedLine(
                frame,
                start,
                end,
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.3,
            )

    def _draw_velocity_3d(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> None:
        """Draw 3D velocity vectors."""
        if context.pose_3d is None or self._prev_pose_3d is None:
            return

        # Calculate velocities
        velocities = (context.pose_3d - self._prev_pose_3d) * self.scale

        # Project to 2D
        pose_2d = project_3d_to_2d(
            context.pose_3d,
            context.frame_width,
            context.frame_height,
            context.focal_length,
            context.camera_distance,
        )

        # Project velocities
        vel_2d = np.zeros_like(velocities[:, :2])
        for i in range(len(velocities)):
            # Project both points
            pt1 = project_3d_to_2d(
                context.pose_3d[i],
                context.frame_width,
                context.frame_height,
                context.focal_length,
                context.camera_distance,
            )
            pt2 = project_3d_to_2d(
                context.pose_3d[i] + velocities[i],
                context.frame_width,
                context.frame_height,
                context.focal_length,
                context.camera_distance,
            )
            vel_2d[i] = np.array(pt2) - np.array(pt1)

        # Draw vectors
        for joint_idx in self.joints:
            start = tuple(pose_2d[joint_idx].astype(int))

            # Clamp vector length
            vector = vel_2d[joint_idx]
            length = np.linalg.norm(vector)
            if length > self.max_length:
                vector = vector / length * self.max_length

            end = tuple((pose_2d[joint_idx] + vector).astype(int))

            # Get color based on depth or speed
            if self.color_mode == "depth":
                from src.visualization.core.colors import get_depth_color

                depth = context.pose_3d[joint_idx, 2]
                color = get_depth_color(depth, 0.0, 2.0)
            elif self.color_mode == "heatmap":
                speed = length / self.max_length
                color = get_heatmap_color(speed, 0.0, 1.0, "jet")
            else:
                color = (0, 255, 255)  # Cyan

            # Draw arrow
            cv2.arrowedLine(
                frame,
                start,
                end,
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.3,
            )
