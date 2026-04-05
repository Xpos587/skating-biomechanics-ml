"""HUD panel system for grouping related elements.

Provides reusable panel components with:
- Semi-transparent backgrounds
- Rounded corners
- Borders
- Automatic sizing
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from src.visualization.config import (
    corner_radius,
    hud_bg_alpha,
    hud_bg_color,
    hud_border_color,
    hud_padding,
    margin,
)

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]  # OpenCV image (H, W, 3)
PanelRenderer = Callable[[Frame, tuple[int, int]], None]


# =============================================================================
# PANEL POSITION ENUM
# =============================================================================


class PanelPosition(IntEnum):
    """Standard positions for HUD panels."""

    TOP_LEFT = 0
    """Top-left corner of frame."""

    TOP_RIGHT = 1
    """Top-right corner of frame."""

    BOTTOM_LEFT = 2
    """Bottom-left corner of frame."""

    BOTTOM_RIGHT = 3
    """Bottom-right corner of frame."""

    TOP_CENTER = 4
    """Top-center of frame."""

    BOTTOM_CENTER = 5
    """Bottom-center of frame."""


# =============================================================================
# HUD PANEL CLASS
# =============================================================================


@dataclass
class HUDPanel:
    """A reusable HUD panel with background and border.

    Attributes:
        title: Optional title text for the panel.
        position: Panel position on frame.
        bg_color: Background color (BGR).
        bg_alpha: Background transparency [0, 1].
        border_color: Border color (BGR).
        border_thickness: Border thickness in pixels.
        padding: Internal padding in pixels.
        corner_radius: Corner radius for rounded rectangle.
        margin: Margin from frame edge.
        width: Fixed width (None = auto-size).
        max_width: Maximum width for auto-sizing.
        render_fn: Optional custom render function.
    """

    title: str | None = None
    position: PanelPosition = PanelPosition.TOP_LEFT
    bg_color: tuple[int, int, int] = hud_bg_color
    bg_alpha: float = hud_bg_alpha
    border_color: tuple[int, int, int] = hud_border_color
    border_thickness: int = 2
    padding: int = hud_padding
    corner_radius: int = corner_radius
    margin: int = margin
    width: int | None = None
    max_width: int = 400
    render_fn: PanelRenderer | None = None

    def get_position(
        self,
        frame_width: int,
        frame_height: int,
        panel_width: int,
        panel_height: int,
    ) -> tuple[int, int]:
        """Get panel position based on position enum.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            panel_width: Panel width in pixels.
            panel_height: Panel height in pixels.

        Returns:
            (x, y) top-left position for panel.
        """
        if self.position == PanelPosition.TOP_LEFT:
            return (self.margin, self.margin)
        elif self.position == PanelPosition.TOP_RIGHT:
            return (frame_width - panel_width - self.margin, self.margin)
        elif self.position == PanelPosition.BOTTOM_LEFT:
            return (self.margin, frame_height - panel_height - self.margin)
        elif self.position == PanelPosition.BOTTOM_RIGHT:
            return (
                frame_width - panel_width - self.margin,
                frame_height - panel_height - self.margin,
            )
        elif self.position == PanelPosition.TOP_CENTER:
            return ((frame_width - panel_width) // 2, self.margin)
        elif self.position == PanelPosition.BOTTOM_CENTER:
            return ((frame_width - panel_width) // 2, frame_height - panel_height - self.margin)
        else:
            return (self.margin, self.margin)

    def draw_background(
        self,
        frame: Frame,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Draw panel background with optional border.

        Uses ROI-scoped blending — no full-frame copy.
        """
        from src.visualization.core.overlay import draw_overlay_rect

        draw_overlay_rect(
            frame,
            (x, y, width, height),
            color=self.bg_color,
            alpha=self.bg_alpha,
            border_color=self.border_color if self.border_thickness > 0 else None,
            border_thickness=self.border_thickness,
        )

    def draw_title(
        self,
        frame: Frame,
        x: int,
        y: int,
        width: int,
    ) -> int:
        """Draw panel title.

        Args:
            frame: OpenCV image to draw on.
            x: Top-left X position.
            y: Top-left Y position.
            width: Panel width.

        Returns:
            Y position below title.
        """
        if self.title is None:
            return y

        # Measure title size
        (_text_width, text_height), baseline = cv2.getTextSize(
            self.title,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        # Draw title text
        title_x = x + self.padding
        title_y = y + self.padding + text_height

        cv2.putText(
            frame,
            self.title,
            (int(title_x), int(title_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.border_color,
            2,
            cv2.LINE_AA,
        )

        # Draw separator line
        line_y = title_y + baseline + self.padding // 2
        cv2.line(
            frame,
            (x + self.padding, line_y),
            (x + width - self.padding, line_y),
            self.border_color,
            1,
            cv2.LINE_AA,
        )

        return line_y + self.padding // 2

    def render(
        self,
        frame: Frame,
        content_fn: PanelRenderer | None = None,
    ) -> tuple[int, int, int, int]:
        """Render panel to frame.

        Args:
            frame: OpenCV image to draw on.
            content_fn: Optional function to render content.

        Returns:
            (x, y, width, height) panel bounding box.
        """
        frame_height, frame_width = frame.shape[:2]

        # If custom render function provided, use it
        if self.render_fn is not None:
            self.render_fn(frame, (frame_width, frame_height))
            # Return dummy bbox
            return (0, 0, 0, 0)

        # Calculate panel size
        if self.width is not None:
            panel_width = self.width
            panel_height = 100  # Default height
        else:
            panel_width = 200  # Default width
            panel_height = 100  # Default height

        # Get position
        x, y = self.get_position(frame_width, frame_height, panel_width, panel_height)

        # Draw background
        self.draw_background(frame, x, y, panel_width, panel_height)

        # Draw title
        content_y = self.draw_title(frame, x, y, panel_width)

        # Draw content
        if content_fn is not None:
            content_fn(frame, (x + self.padding, content_y))

        return (x, y, panel_width, panel_height)


# =============================================================================
# PREDEFINED PANELS
# =============================================================================


@dataclass
class StandardPanels:
    """Predefined standard HUD panels."""

    @staticmethod
    def info_panel() -> HUDPanel:
        """Create standard info panel (top-left)."""
        return HUDPanel(
            title="INFO",
            position=PanelPosition.TOP_LEFT,
            width=200,
        )

    @staticmethod
    def metrics_panel() -> HUDPanel:
        """Create standard metrics panel (top-right)."""
        return HUDPanel(
            title="METRICS",
            position=PanelPosition.TOP_RIGHT,
            width=250,
        )

    @staticmethod
    def phase_panel() -> HUDPanel:
        """Create standard phase panel (bottom-left)."""
        return HUDPanel(
            title="PHASE",
            position=PanelPosition.BOTTOM_LEFT,
            width=200,
        )

    @staticmethod
    def blade_panel() -> HUDPanel:
        """Create standard blade indicator panel (bottom-right)."""
        return HUDPanel(
            title="BLADE",
            position=PanelPosition.BOTTOM_RIGHT,
            width=180,
        )
