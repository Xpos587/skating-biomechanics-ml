"""HUD layout manager.

Provides grid-based positioning system for HUD elements:
- Position definitions
- Layout calculator
- Spacing management
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from skating_ml.visualization.config import margin, panel_spacing

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]  # OpenCV image (H, W, 3)
PixelPosition = tuple[int, int]  # (x, y) pixel coordinates


# =============================================================================
# POSITION ENUM
# =============================================================================


class Position(IntEnum):
    """Standard positions for HUD elements."""

    TOP_LEFT = 0
    """Top-left corner."""

    TOP_RIGHT = 1
    """Top-right corner."""

    BOTTOM_LEFT = 2
    """Bottom-left corner."""

    BOTTOM_RIGHT = 3
    """Bottom-right corner."""

    TOP_CENTER = 4
    """Top-center."""

    BOTTOM_CENTER = 5
    """Bottom-center."""

    CENTER = 6
    """Center of frame."""


# =============================================================================
# LAYOUT CLASS
# =============================================================================


@dataclass
class HUDLayout:
    """Layout manager for HUD elements.

    Attributes:
        margin: Margin from frame edge.
        panel_spacing: Vertical spacing between elements.
        columns: Number of columns for grid layout.
        rows: Number of rows for grid layout.
    """

    margin: int = margin
    panel_spacing: int = panel_spacing
    columns: int = 2
    rows: int = 3

    def get_position(
        self,
        pos: Position | str,
        frame_width: int,
        frame_height: int,
        element_width: int = 0,
        element_height: int = 0,
    ) -> PixelPosition:
        """Get position for HUD element.

        Args:
            pos: Position enum or string name.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            element_width: Element width (for centering).
            element_height: Element height (for centering).

        Returns:
            (x, y) position for element.

        Example:
            >>> layout = HUDLayout()
            >>> layout.get_position(Position.TOP_RIGHT, 1920, 1080, 200, 100)
            (1720, 10)  # Top-right with margin
        """
        # Convert string to enum
        if isinstance(pos, str):
            pos = Position[pos.upper()]

        if pos == Position.TOP_LEFT:
            return (self.margin, self.margin)
        elif pos == Position.TOP_RIGHT:
            x = (
                frame_width - element_width - self.margin
                if element_width > 0
                else frame_width - self.margin
            )
            return (x, self.margin)
        elif pos == Position.BOTTOM_LEFT:
            y = (
                frame_height - element_height - self.margin
                if element_height > 0
                else frame_height - self.margin
            )
            return (self.margin, y)
        elif pos == Position.BOTTOM_RIGHT:
            x = (
                frame_width - element_width - self.margin
                if element_width > 0
                else frame_width - self.margin
            )
            y = (
                frame_height - element_height - self.margin
                if element_height > 0
                else frame_height - self.margin
            )
            return (x, y)
        elif pos == Position.TOP_CENTER:
            x = (frame_width - element_width) // 2 if element_width > 0 else frame_width // 2
            return (x, self.margin)
        elif pos == Position.BOTTOM_CENTER:
            x = (frame_width - element_width) // 2 if element_width > 0 else frame_width // 2
            y = (
                frame_height - element_height - self.margin
                if element_height > 0
                else frame_height - self.margin
            )
            return (x, y)
        elif pos == Position.CENTER:
            x = (frame_width - element_width) // 2 if element_width > 0 else frame_width // 2
            y = (frame_height - element_height) // 2 if element_height > 0 else frame_height // 2
            return (x, y)
        else:
            return (self.margin, self.margin)

    def get_grid_position(
        self,
        col: int,
        row: int,
        frame_width: int,
        frame_height: int,
        cell_width: int,
        cell_height: int,
    ) -> PixelPosition:
        """Get position for grid-based layout.

        Args:
            col: Column index (0-based).
            row: Row index (0-based).
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            cell_width: Cell width in pixels.
            cell_height: Cell height in pixels.

        Returns:
            (x, y) position for grid cell.

        Example:
            >>> layout = HUDLayout(columns=2, rows=2)
            >>> layout.get_grid_position(0, 0, 1920, 1080, 200, 100)
            (10, 10)  # First cell
        """
        x = self.margin + col * (cell_width + self.panel_spacing)
        y = self.margin + row * (cell_height + self.panel_spacing)
        return (x, y)

    def calculate_cell_size(
        self,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        """Calculate cell size for grid layout.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            (cell_width, cell_height) in pixels.

        Example:
            >>> layout = HUDLayout(columns=2, rows=2)
            >>> layout.calculate_cell_size(1920, 1080)
            (945, 530)  # Cell size with margins
        """
        usable_width = frame_width - 2 * self.margin - (self.columns - 1) * self.panel_spacing
        usable_height = frame_height - 2 * self.margin - (self.rows - 1) * self.panel_spacing

        cell_width = usable_width // self.columns
        cell_height = usable_height // self.rows

        return (cell_width, cell_height)


# =============================================================================
# PREDEFINED LAYOUTS
# =============================================================================


@dataclass
class LayoutConfigs:
    """Predefined layout configurations."""

    @staticmethod
    def default() -> HUDLayout:
        """Default layout (2x3 grid)."""
        return HUDLayout(
            margin=10,
            panel_spacing=15,
            columns=2,
            rows=3,
        )

    @staticmethod
    def compact() -> HUDLayout:
        """Compact layout (3x2 grid)."""
        return HUDLayout(
            margin=5,
            panel_spacing=10,
            columns=3,
            rows=2,
        )

    @staticmethod
    def sparse() -> HUDLayout:
        """Sparse layout (1x4 grid)."""
        return HUDLayout(
            margin=20,
            panel_spacing=20,
            columns=1,
            rows=4,
        )

    @staticmethod
    def presentation() -> HUDLayout:
        """Presentation layout (larger margins)."""
        return HUDLayout(
            margin=30,
            panel_spacing=20,
            columns=2,
            rows=2,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_default_layout() -> HUDLayout:
    """Get default layout configuration.

    Returns:
        HUDLayout with default settings.
    """
    return LayoutConfigs.default()


def calculate_text_position(
    text: str,
    position: Position,
    frame_width: int,
    frame_height: int,
    font_scale: float = 0.6,
    thickness: int = 2,
    margin: int = 10,
) -> PixelPosition:
    """Calculate position for text to be drawn.

    Args:
        text: Text string to draw.
        position: Desired position.
        frame_width: Frame width.
        frame_height: Frame height.
        font_scale: Font scale for cv2.getTextSize.
        thickness: Text thickness.
        margin: Margin from edge.

    Returns:
        (x, y) position for text.

    Example:
        >>> calculate_text_position("FPS: 30", Position.TOP_RIGHT, 1920, 1080)
        (1850, 32)  # Text position
    """
    # Measure text size
    import cv2

    (text_width, text_height), _baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    layout = HUDLayout(margin=margin)

    base_x, base_y = layout.get_position(position, frame_width, frame_height)

    # Adjust for text dimensions
    if position in (Position.TOP_RIGHT, Position.BOTTOM_RIGHT):
        x = base_x - text_width
    elif position in (Position.TOP_CENTER, Position.BOTTOM_CENTER, Position.CENTER):
        x = base_x - text_width // 2
    else:
        x = base_x

    # Adjust for text height
    if position in (Position.BOTTOM_LEFT, Position.BOTTOM_RIGHT, Position.BOTTOM_CENTER):
        y = base_y - text_height
    elif position == Position.CENTER:
        y = base_y - text_height // 2
    else:
        y = base_y + text_height

    return (x, y)


def clip_to_frame(
    x: int,
    y: int,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Clip element to fit within frame.

    Args:
        x: Element X position.
        y: Element Y position.
        width: Element width.
        height: Element height.
        frame_width: Frame width.
        frame_height: Frame height.

    Returns:
        (x, y, width, height) clipped to frame.

    Example:
        >>> clip_to_frame(-10, -10, 100, 100, 1920, 1080)
        (0, 0, 90, 90)  # Clipped to frame
    """
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)

    return (x, y, width, height)
