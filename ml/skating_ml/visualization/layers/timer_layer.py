"""Timer/stopwatch visualization layer.

Shows elapsed time as mm:ss.ms overlay on the video frame.
"""

from skating_ml.visualization.config import COLOR_WHITE, LayerConfig
from skating_ml.visualization.core.text import put_text
from skating_ml.visualization.layers.base import Frame, Layer, LayerContext


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

        text = f"{minutes:02d}:{int(seconds):02d}.{ms:02d}"

        # Position in top-right corner
        margin = 10
        x = context.frame_width - 120  # approximate width for "00:00.00"
        y = margin

        put_text(
            frame,
            text,
            (x, y),
            font_size=14,
            color=COLOR_WHITE,
            bg_color=(0, 0, 0),
            bg_alpha=0.6,
        )

        return frame
