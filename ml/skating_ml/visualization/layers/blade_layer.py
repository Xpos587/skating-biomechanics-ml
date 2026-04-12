"""Blade edge indicator layer.

Renders blade edge state indicators on video frames.
"""

import numpy as np
from numpy.typing import NDArray

from skating_ml.visualization.config import LayerConfig
from skating_ml.visualization.hud.elements import draw_blade_indicator_hud
from skating_ml.visualization.layers.base import Layer, LayerContext

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]


# =============================================================================
# BLADE LAYER
# =============================================================================


class BladeLayer(Layer):
    """Layer for drawing blade edge indicators.

    Shows current blade edge state (inside/outside/flat) with
    directional arrow and angle information.

    Attributes:
        config: LayerConfig for this layer.
        position: Position for indicator (x, y).
        size: Size of indicator arrow.
        thickness: Thickness of indicator lines.
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        position: tuple[int, int] = (10, 30),
        size: int = 30,
        thickness: int = 3,
    ):
        """Initialize blade layer.

        Args:
            config: LayerConfig for this layer.
            position: (x, y) position for indicator.
            size: Size of indicator arrow in pixels.
            thickness: Thickness of indicator lines.
        """
        super().__init__(config or LayerConfig(z_index=2), name="Blade")
        self.position = position
        self.size = size
        self.thickness = thickness

    def render(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> Frame:
        """Render blade indicator to frame.

        Args:
            frame: OpenCV image (H, W, 3) BGR format.
            context: LayerContext with blade state.

        Returns:
            Frame with blade indicator.
        """
        # Check if blade state available
        if context.blade_state is None:
            return frame

        # Draw blade indicator
        draw_blade_indicator_hud(
            frame,
            context.blade_state,
            position=self.position,
            size=self.size,
            thickness=self.thickness,
        )

        return frame
