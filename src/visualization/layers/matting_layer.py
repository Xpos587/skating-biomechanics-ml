"""Video matting visualization layer.

Applies alpha matte to blur/darken background while keeping foreground sharp.
Reads alpha from ``LayerContext.custom_data["alpha_matte"]``.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.visualization.config import LayerConfig
from src.visualization.layers.base import Layer, LayerContext


class MattingLayer(Layer):
    """Applies video matting effect: blur background, keep foreground.

    Args:
        blur_strength: Gaussian blur kernel size for background (must be odd).
        opacity: Effect opacity (0.0 = no effect, 1.0 = full matting).
        config: Optional LayerConfig.
    """

    def __init__(
        self,
        blur_strength: int = 21,
        opacity: float = 1.0,
        config: LayerConfig | None = None,
    ) -> None:
        super().__init__(config=config or LayerConfig(enabled=True, z_index=-3, opacity=opacity))
        self.name = "VideoMatting"
        self._blur = blur_strength

    def render(self, frame: np.ndarray, context: LayerContext) -> np.ndarray:
        alpha = context.custom_data.get("alpha_matte")
        if alpha is None:
            return frame

        alpha_3ch = np.stack([alpha] * 3, axis=-1)  # (H, W, 3)

        # Blur background
        blurred = cv2.GaussianBlur(frame, (self._blur, self._blur), 0)

        alpha_expanded = alpha_3ch.astype(np.float32) / 255.0
        result = frame.astype(np.float32) * alpha_expanded + blurred.astype(np.float32) * (
            1.0 - alpha_expanded
        )
        return result.astype(np.uint8)
