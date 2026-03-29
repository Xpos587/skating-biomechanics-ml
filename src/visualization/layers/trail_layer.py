"""Motion trails layer.

Renders motion trails showing trajectory of key joints over time.
"""

from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from src.types import H36Key
from src.visualization.config import (
    LayerConfig,
)
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
# TRAIL LAYER
# =============================================================================


class TrailLayer(Layer):
    """Layer for drawing motion trails.

    Shows trajectory of key joints over recent frames.

    Attributes:
        config: LayerConfig for this layer.
        length: Number of frames to keep in trail.
        joint: Joint index to track (default: left foot).
        color: Trail color (BGR).
        width: Trail line width.
        smoothing: Apply Gaussian smoothing to trail.
    """

    # Default joint to track (left foot)
    DEFAULT_JOINT: Final[int] = H36Key.LFOOT

    def __init__(
        self,
        config: LayerConfig | None = None,
        length: int = 20,
        joint: int = DEFAULT_JOINT,
        color: tuple[int, int, int] = (255, 255, 0),
        width: int = 2,
        smoothing: bool = True,
    ):
        """Initialize trail layer.

        Args:
            config: LayerConfig for this layer.
            length: Number of frames to keep in trail.
            joint: Joint index to track.
            color: Trail color (BGR).
            width: Trail line width.
            smoothing: Apply Gaussian smoothing to trail.
        """
        super().__init__(config or LayerConfig(z_index=1), name="Trail")
        self.length = length
        self.joint = joint
        self.color = color
        self.width = width
        self.smoothing = smoothing

        # Trail history
        self._trail_2d: list[tuple[int, int]] = []
        self._trail_3d: list[tuple[int, int, int]] = []

    def render(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> Frame:
        """Render motion trail to frame.

        Args:
            frame: OpenCV image (H, W, 3) BGR format.
            context: LayerContext with pose data.

        Returns:
            Frame with motion trail.
        """
        # Add current position to trail
        if context.pose_3d is not None:
            pos = context.pose_3d[self.joint]
            self._trail_3d.append(tuple(pos))
            if len(self._trail_3d) > self.length:
                self._trail_3d.pop(0)

            # Draw 3D trail
            self._draw_trail_3d(frame, context)

        elif context.pose_2d is not None:
            pos = context.pose_2d[self.joint]
            self._trail_2d.append(tuple(pos))
            if len(self._trail_2d) > self.length:
                self._trail_2d.pop(0)

            # Draw 2D trail
            self._draw_trail_2d(frame, context)

        return frame

    def _draw_trail_2d(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> None:
        """Draw 2D motion trail."""
        if len(self._trail_2d) < 2:
            return

        # Convert to pixel coordinates
        if context.normalized:
            trail_px = [
                normalized_to_pixel(
                    np.array(pos),
                    context.frame_width,
                    context.frame_height,
                )
                for pos in self._trail_2d
            ]
        else:
            trail_px = [(int(pos[0]), int(pos[1])) for pos in self._trail_2d]

        # Apply smoothing if enabled
        if self.smoothing and len(trail_px) > 3:
            trail_array = np.array(trail_px)
            trail_array[:, 0] = gaussian_filter1d(trail_array[:, 0], sigma=1.5).astype(int)
            trail_array[:, 1] = gaussian_filter1d(trail_array[:, 1], sigma=1.5).astype(int)
            trail_px = [tuple(pt) for pt in trail_array]

        # Draw trail with fading opacity
        for i in range(len(trail_px) - 1):
            pt1 = trail_px[i]
            pt2 = trail_px[i + 1]

            # Calculate fading alpha
            alpha = (i + 1) / len(trail_px)

            # Draw segment
            if alpha > 0.1:
                # Scale color by alpha
                color = (
                    int(self.color[0] * alpha),
                    int(self.color[1] * alpha),
                    int(self.color[2] * alpha),
                )
                cv2.line(frame, pt1, pt2, color, self.width, cv2.LINE_AA)

    def _draw_trail_3d(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> None:
        """Draw 3D motion trail."""
        if len(self._trail_3d) < 2:
            return

        # Project to 2D
        trail_2d = []
        for pos_3d in self._trail_3d:
            pt_2d = project_3d_to_2d(
                np.array(pos_3d),
                context.frame_width,
                context.frame_height,
                context.focal_length,
                context.camera_distance,
            )
            trail_2d.append(pt_2d)

        # Apply smoothing if enabled
        if self.smoothing and len(trail_2d) > 3:
            trail_array = np.array(trail_2d)
            trail_array[:, 0] = gaussian_filter1d(trail_array[:, 0], sigma=1.5).astype(int)
            trail_array[:, 1] = gaussian_filter1d(trail_array[:, 1], sigma=1.5).astype(int)
            trail_2d = [tuple(pt) for pt in trail_array]

        # Draw trail with fading opacity
        for i in range(len(trail_2d) - 1):
            pt1 = trail_2d[i]
            pt2 = trail_2d[i + 1]

            # Calculate fading alpha
            alpha = (i + 1) / len(trail_2d)

            # Draw segment
            if alpha > 0.1:
                color = (
                    int(self.color[0] * alpha),
                    int(self.color[1] * alpha),
                    int(self.color[2] * alpha),
                )
                cv2.line(frame, pt1, pt2, color, self.width, cv2.LINE_AA)

    def reset(self) -> None:
        """Reset trail history.

        Call this when starting a new video or element.
        """
        self._trail_2d.clear()
        self._trail_3d.clear()
