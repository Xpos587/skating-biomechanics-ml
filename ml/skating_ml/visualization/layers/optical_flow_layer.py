"""Optical flow visualization layer.

Renders dense optical flow as HSV color wheel overlay.
Reads flow field from ``LayerContext.custom_data["flow_field"]``.
"""

from __future__ import annotations

import cv2
import numpy as np

from skating_ml.visualization.config import LayerConfig
from skating_ml.visualization.layers.base import Layer, LayerContext


class OpticalFlowLayer(Layer):
    """Renders optical flow as HSV color wheel overlay.

    Args:
        opacity: Blending opacity.
        config: Optional LayerConfig.
    """

    def __init__(self, opacity: float = 0.5, config: LayerConfig | None = None) -> None:
        super().__init__(config=config or LayerConfig(enabled=True, z_index=0, opacity=opacity))
        self.name = "OpticalFlow"

    def render(self, frame: np.ndarray, context: LayerContext) -> np.ndarray:
        flow = context.custom_data.get("flow_field")
        if flow is None:
            return frame

        # Convert flow to HSV color wheel visualization
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]

        # Magnitude and angle
        magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)

        # Build HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = (angle / 2).astype(np.uint8)  # Hue: direction
        hsv[:, :, 1] = 255  # Saturation: full
        # Value: normalized magnitude
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        hsv[:, :, 2] = mag_norm.astype(np.uint8)

        # HSV -> BGR
        colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blend
        alpha = self.opacity
        blended = cv2.addWeighted(colored, alpha, frame, 1.0 - alpha, 0)
        return blended
