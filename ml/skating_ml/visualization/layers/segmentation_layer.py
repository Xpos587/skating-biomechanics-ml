"""Segmentation mask visualization layer.

Renders a semi-transparent colored overlay over the segmented region.
Reads mask from ``LayerContext.custom_data["seg_mask"]``.
"""

from __future__ import annotations

import cv2
import numpy as np

from skating_ml.visualization.config import LayerConfig
from skating_ml.visualization.layers.base import Layer, LayerContext


class SegmentationMaskLayer(Layer):
    """Renders segmentation mask as semi-transparent overlay.

    Args:
        color: BGR color for the mask overlay (default: cyan).
        opacity: Blending opacity for the mask region.
        config: Optional LayerConfig.
    """

    def __init__(
        self,
        color: tuple[int, int, int] = (255, 255, 0),  # cyan BGR
        opacity: float = 0.3,
        config: LayerConfig | None = None,
    ) -> None:
        super().__init__(config=config or LayerConfig(enabled=True, z_index=-2, opacity=opacity))
        self.name = "Segmentation"
        self._color = color

    def render(self, frame: np.ndarray, context: LayerContext) -> np.ndarray:
        mask = context.custom_data.get("seg_mask")
        if mask is None:
            return frame

        overlay = frame.copy()

        # Apply color to masked region
        mask_uint8 = mask.astype(np.uint8) * 255
        colored_mask = np.full_like(frame, self._color)
        overlay[mask_uint8 > 0] = colored_mask[mask_uint8 > 0]

        # Blend
        alpha = self.opacity
        blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

        # Draw contour around mask for definition
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, self._color, 2)

        return blended
