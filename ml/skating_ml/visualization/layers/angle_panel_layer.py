"""Angle panel layer with progress bars (Sports2D-style).

Displays a side panel listing all computed joint/segment angles
with progress bars and degree values.
"""

from __future__ import annotations

import cv2
import numpy as np

from skating_ml.visualization.config import (
    COLOR_BLACK,
    COLOR_GREEN,
    LayerConfig,
    VisualizationConfig,
    hud_bg_alpha,
    hud_bg_color,
    hud_padding,
)
from skating_ml.visualization.core.text import draw_text_outlined
from skating_ml.visualization.layers.base import Frame, Layer, LayerContext


class AnglePanelLayer(Layer):
    """Side panel showing computed angles with progress bars.

    Reads angles from ``context.custom_data["angles"]`` dict:
        {"L Knee": 120.5, "R Knee": 95.0, ...}

    Each angle shows:
    - Name label
    - Degree value
    - Progress bar (0-180° mapped to bar width)
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        viz_config: VisualizationConfig | None = None,
        max_angle: float = 180.0,
        bar_height: int = 12,
        line_spacing: int = 22,
        bar_width: int = 100,
    ):
        super().__init__(config=config or LayerConfig(enabled=True, z_index=7))
        self.viz = viz_config or VisualizationConfig()
        self.max_angle = max_angle
        self.bar_height = bar_height
        self.line_spacing = line_spacing
        self.bar_width = bar_width

    def render(self, frame: Frame, context: LayerContext) -> Frame:
        angles = context.custom_data.get("angles")
        if not angles:
            return frame

        x0 = hud_padding
        y0 = hud_padding

        # Semi-transparent background (ROI-scoped, no full-frame copy)
        total_height = len(angles) * self.line_spacing + hud_padding * 2
        bg_width = self.bar_width + 220
        from skating_ml.visualization.core.overlay import draw_overlay_rect

        draw_overlay_rect(
            frame,
            (x0 - 5, y0 - 5, bg_width + 5, total_height + 5),
            color=hud_bg_color,
            alpha=hud_bg_alpha,
        )

        for i, (name, value) in enumerate(angles.items()):
            y = y0 + i * self.line_spacing + hud_padding

            if np.isnan(value):
                draw_text_outlined(
                    frame, f"{name}:", (x0, y + 10), font_scale=0.4, thickness=1, color=COLOR_BLACK
                )
                continue

            # Label
            draw_text_outlined(
                frame, f"{name}:", (x0, y + 10), font_scale=0.4, thickness=1, color=COLOR_GREEN
            )

            # Value
            val_x = x0 + 150
            draw_text_outlined(
                frame,
                f"{value:.1f}",
                (val_x, y + 10),
                font_scale=0.4,
                thickness=1,
                color=COLOR_GREEN,
            )

            # Progress bar
            bar_x = x0 + 200
            bar_y = y + 2
            pct = min(abs(value) / self.max_angle, 1.0)
            bar_len = int(pct * self.bar_width)

            # Bar background
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + self.bar_width, bar_y + self.bar_height),
                (50, 50, 50),
                -1,
                cv2.LINE_AA,
            )
            # Bar fill
            if bar_len > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + bar_len, bar_y + self.bar_height),
                    COLOR_GREEN,
                    -1,
                    cv2.LINE_AA,
                )

        return frame
