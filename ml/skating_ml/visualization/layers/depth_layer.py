"""Depth map visualization layer.

Renders monocular depth estimation as a color-mapped overlay (turbo colormap).
Reads depth map from ``LayerContext.custom_data["depth_map"]``.
"""

from __future__ import annotations

import cv2
import numpy as np

from skating_ml.visualization.config import LayerConfig
from skating_ml.visualization.layers.base import Layer, LayerContext


class DepthMapLayer(Layer):
    """Renders depth map as semi-transparent color overlay.

    Args:
        opacity: Blending opacity (0.0 = invisible, 1.0 = full depth map).
        config: Optional LayerConfig.
    """

    def __init__(self, opacity: float = 0.4, config: LayerConfig | None = None) -> None:
        super().__init__(config=config or LayerConfig(enabled=True, z_index=-1, opacity=opacity))
        self.name = "DepthMap"

    def render(self, frame: np.ndarray, context: LayerContext) -> np.ndarray:
        depth_map = context.custom_data.get("depth_map")
        if depth_map is None:
            return frame

        # Apply turbo colormap: (H, W) float32 -> (H, W, 3) uint8 BGR
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

        # Blend with original frame
        alpha = self.opacity
        blended = cv2.addWeighted(colored, alpha, frame, 1.0 - alpha, 0)
        return blended
