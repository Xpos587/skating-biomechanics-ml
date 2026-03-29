"""Skeleton overlay layer.

Renders 2D or 3D skeleton overlay on video frames.
"""

import numpy as np
from numpy.typing import NDArray

from src.visualization.config import LayerConfig
from src.visualization.layers.base import Layer, LayerContext
from src.visualization.skeleton import draw_skeleton, draw_skeleton_3d

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]


# =============================================================================
# SKELETON LAYER
# =============================================================================


class SkeletonLayer(Layer):
    """Layer for drawing skeleton overlay.

    Supports both 2D and 3D skeleton rendering.

    Attributes:
        config: LayerConfig for this layer.
        mode: "2d" or "3d" rendering mode.
        line_width: Width of skeleton lines.
        joint_radius: Radius of joint circles.
        depth_min: Minimum depth for 3D color coding.
        depth_max: Maximum depth for 3D color coding.
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        mode: str = "2d",
        line_width: int = 2,
        joint_radius: int = 4,
        depth_min: float = 0.0,
        depth_max: float = 2.0,
    ):
        """Initialize skeleton layer.

        Args:
            config: LayerConfig for this layer.
            mode: "2d" or "3d" rendering mode.
            line_width: Width of skeleton lines.
            joint_radius: Radius of joint circles.
            depth_min: Minimum depth for 3D color coding.
            depth_max: Maximum depth for 3D color coding.
        """
        super().__init__(config or LayerConfig(z_index=0), name="Skeleton")
        self.mode = mode
        self.line_width = line_width
        self.joint_radius = joint_radius
        self.depth_min = depth_min
        self.depth_max = depth_max

    def render(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> Frame:
        """Render skeleton overlay to frame.

        Args:
            frame: OpenCV image (H, W, 3) BGR format.
            context: LayerContext with pose data.

        Returns:
            Frame with skeleton overlay.
        """
        # Check if pose data available
        if self.mode == "3d":
            if context.pose_3d is None:
                return frame

            # Draw 3D skeleton
            draw_skeleton_3d(
                frame,
                context.pose_3d,
                context.frame_width,
                context.frame_height,
                line_width=self.line_width,
                joint_radius=self.joint_radius,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
                camera_distance=context.camera_distance,
                focal_length=context.focal_length,
            )
        else:
            if context.pose_2d is None:
                return frame

            # Draw 2D skeleton
            draw_skeleton(
                frame,
                context.pose_2d,
                context.frame_width,
                context.frame_height,
                normalized=context.normalized,
                confidences=context.confidences,
                line_width=self.line_width,
                joint_radius=self.joint_radius,
            )

        return frame
