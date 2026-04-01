"""Timer/stopwatch visualization layer.

Shows elapsed time as mm:ss.ms overlay on the video frame.
"""

import cv2

from src.visualization.config import COLOR_WHITE, LayerConfig
from src.visualization.layers.base import Frame, Layer, LayerContext


class TimerLayer(Layer):
    """Display elapsed time as mm:ss.ms in the top-right corner.

    Reads frame_idx and fps from LayerContext to compute elapsed time.
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
    ):
        super().__init__(config=config or LayerConfig(enabled=True, z_index=10))

    def render(self, frame: Frame, context: LayerContext) -> Frame:
        fps = context.fps
        if fps <= 0:
            return frame

        elapsed = context.frame_idx / fps
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        ms = int((seconds % 1) * 100)

        time_str = f"{minutes:02d}:{int(seconds):02d}.{ms:02d}"

        # Position in top-right corner
        text = time_str

        # Measure text size for background box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        margin = 10
        x = context.frame_width - tw - margin * 2
        y = margin

        # Draw background
        cv2.rectangle(
            frame,
            (x - margin, y - margin),
            (x + tw + margin, y + th + margin + baseline),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y + th),
            font,
            font_scale,
            COLOR_WHITE,
            thickness,
            cv2.LINE_AA,
        )

        return frame
