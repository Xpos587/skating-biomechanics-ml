"""HUD layer.

Renders comprehensive HUD with metrics, phase, and frame info.
"""

import numpy as np
from numpy.typing import NDArray

from src.visualization.config import LayerConfig
from src.visualization.core.text import put_text
from src.visualization.hud.elements import (
    draw_frame_counter,
    draw_metrics_panel,
    draw_phase_indicator,
)
from src.visualization.layers.base import Layer, LayerContext

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]


# =============================================================================
# HUD LAYER
# =============================================================================


class HUDLayer(Layer):
    """Layer for drawing comprehensive HUD.

    Shows:
    - Frame counter
    - Metrics panel
    - Phase indicator

    Attributes:
        config: LayerConfig for this layer.
        show_frame_counter: Show frame counter.
        show_metrics: Show metrics panel.
        show_phase: Show phase indicator.
        metrics_position: Position for metrics panel.
        phase_position: Position for phase indicator.
        frame_counter_position: Position for frame counter.
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        show_frame_counter: bool = True,
        show_metrics: bool = True,
        show_phase: bool = True,
        metrics_position: tuple[int, int] = (10, 30),
        phase_position: tuple[int, int] = (10, 200),
        frame_counter_position: tuple[int, int] = (10, 30),
    ):
        """Initialize HUD layer.

        Args:
            config: LayerConfig for this layer.
            show_frame_counter: Show frame counter.
            show_metrics: Show metrics panel.
            show_phase: Show phase indicator.
            metrics_position: Position for metrics panel.
            phase_position: Position for phase indicator.
            frame_counter_position: Position for frame counter.
        """
        super().__init__(config or LayerConfig(z_index=3), name="HUD")
        self.show_frame_counter = show_frame_counter
        self.show_metrics = show_metrics
        self.show_phase = show_phase
        self.metrics_position = metrics_position
        self.phase_position = phase_position
        self.frame_counter_position = frame_counter_position

    def render(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> Frame:
        """Render HUD to frame.

        Args:
            frame: OpenCV image (H, W, 3) BGR format.
            context: LayerContext with data.

        Returns:
            Frame with HUD rendered.
        """
        # Draw frame counter
        if self.show_frame_counter:
            draw_frame_counter(
                frame,
                context.frame_idx,
                context.total_frames,
                position=self.frame_counter_position,
            )

        # Draw metrics panel
        if self.show_metrics and context.metrics:
            draw_metrics_panel(
                frame,
                context.metrics,
                position=self.metrics_position,
            )

        # Draw phase indicator
        if self.show_phase and context.phase:
            draw_phase_indicator(
                frame,
                context.phase,
                position=self.phase_position,
            )

        return frame


# =============================================================================
# SIMPLIFIED HUD LAYERS
# =============================================================================


class MinimalHUDLayer(Layer):
    """Minimal HUD with just frame counter.

    Useful for unobtrusive debugging.
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        position: tuple[int, int] = (10, 30),
    ):
        """Initialize minimal HUD layer.

        Args:
            config: LayerConfig for this layer.
            position: Position for frame counter.
        """
        super().__init__(config or LayerConfig(z_index=3), name="MinimalHUD")
        self.position = position

    def render(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> Frame:
        """Render minimal HUD to frame."""
        draw_frame_counter(
            frame,
            context.frame_idx,
            context.total_frames,
            position=self.position,
        )
        return frame


class DebugHUDLayer(Layer):
    """Comprehensive debug HUD with all information."""

    def __init__(
        self,
        config: LayerConfig | None = None,
        position: tuple[int, int] = (10, 30),
    ):
        """Initialize debug HUD layer.

        Args:
            config: LayerConfig for this layer.
            position: Starting position for HUD elements.
        """
        super().__init__(config or LayerConfig(z_index=3), name="DebugHUD")
        self.position = position

    def render(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> Frame:
        """Render debug HUD to frame."""
        x, y = self.position

        put_text(
            frame, f"Frame: {context.frame_idx}/{context.total_frames or '?'}", (x, y), font_size=14
        )
        y += 22

        put_text(frame, f"FPS: {context.fps:.1f}", (x, y), font_size=14)
        y += 22

        if context.phase:
            put_text(frame, f"Phase: {context.phase}", (x, y), font_size=14)
            y += 22

        if context.blade_state:
            put_text(frame, f"Blade: {context.blade_state.blade_type.name}", (x, y), font_size=14)
            y += 22

        if context.metrics:
            put_text(frame, f"Metrics: {len(context.metrics)}", (x, y), font_size=14)

        return frame
