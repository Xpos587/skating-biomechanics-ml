"""Configuration classes for visualization system.

This module provides centralized configuration for all visualization components,
eliminating magic numbers and providing a single source of truth.

Example:
    >>> from skating_ml.visualization.config import VisualizationConfig, LayerConfig
    >>> config = VisualizationConfig()
    >>> layer = LayerConfig(enabled=True, z_index=1)
"""

from dataclasses import dataclass, field
from typing import Final

# =============================================================================
# COLOR CONSTANTS
# =============================================================================

# Standard colors (BGR format for OpenCV)
COLOR_WHITE: Final[tuple[int, int, int]] = (255, 255, 255)
COLOR_BLACK: Final[tuple[int, int, int]] = (0, 0, 0)
COLOR_RED: Final[tuple[int, int, int]] = (0, 0, 255)
COLOR_GREEN: Final[tuple[int, int, int]] = (0, 255, 0)
COLOR_BLUE: Final[tuple[int, int, int]] = (255, 0, 0)
COLOR_YELLOW: Final[tuple[int, int, int]] = (0, 255, 255)
COLOR_CYAN: Final[tuple[int, int, int]] = (255, 255, 0)
COLOR_MAGENTA: Final[tuple[int, int, int]] = (255, 0, 255)

# Depth color gradient (near -> far)
DEPTH_COLORS: Final[list[tuple[int, int, int]]] = [
    (255, 0, 0),  # Near - Red
    (255, 128, 0),  # Medium-Near - Orange
    (255, 255, 0),  # Medium - Yellow
    (0, 255, 0),  # Medium-Far - Green
    (0, 0, 255),  # Far - Blue
]

# Blade edge colors (for import by other modules)
blade_inside_color: Final[tuple[int, int, int]] = COLOR_GREEN
blade_outside_color: Final[tuple[int, int, int]] = COLOR_RED
blade_flat_color: Final[tuple[int, int, int]] = COLOR_YELLOW
blade_unknown_color: Final[tuple[int, int, int]] = COLOR_WHITE

# Default font settings (for import by other modules)
font_path: Final[str] = "/usr/share/fonts/TTF/DejaVuSans.ttf"
font_scale: Final[float] = 0.6
font_thickness: Final[int] = 2
font_color: Final[tuple[int, int, int]] = COLOR_WHITE

# Default HUD settings (for import by other modules)
hud_bg_color: Final[tuple[int, int, int]] = COLOR_BLACK
hud_bg_alpha: Final[float] = 0.6
hud_padding: Final[int] = 10
hud_border_color: Final[tuple[int, int, int]] = COLOR_CYAN
margin: Final[int] = 10
panel_spacing: Final[int] = 15
corner_radius: Final[int] = 5
line_width: Final[int] = 2
joint_radius: Final[int] = 4
confidence_threshold: Final[float] = 0.5

# 3D visualization defaults
camera_distance: Final[float] = 3.0
focal_length: Final[int] = 800

# Blade indicator defaults
blade_indicator_size: Final[int] = 30
blade_indicator_thickness: Final[int] = 3

# Skeleton colors (for import by other modules)
color_left_side: Final[tuple[int, int, int]] = COLOR_BLUE
color_right_side: Final[tuple[int, int, int]] = COLOR_RED
color_center: Final[tuple[int, int, int]] = COLOR_GREEN
color_joint: Final[tuple[int, int, int]] = COLOR_WHITE


# =============================================================================
# LAYER DEFINITIONS
# =============================================================================


@dataclass
class LayerConfig:
    """Configuration for a visualization layer.

    Attributes:
        enabled: Whether this layer is rendered.
        z_index: Drawing order (higher = drawn on top).
        opacity: Transparency level (0.0 = invisible, 1.0 = opaque).
    """

    enabled: bool = True
    z_index: int = 0
    opacity: float = 1.0

    def is_visible(self) -> bool:
        """Check if layer should be rendered."""
        return self.enabled and self.opacity > 0.0


# =============================================================================
# MAIN VISUALIZATION CONFIG
# =============================================================================


@dataclass
class VisualizationConfig:
    """Main configuration for the visualization system.

    This class centralizes all visualization settings, including:
    - Layout (margins, spacing, positioning)
    - Text (fonts, sizes, colors)
    - Skeleton (line widths, joint sizes, colors)
    - HUD (backgrounds, transparency)
    - 3D visualization (depth colors, projection)

    Example:
        >>> config = VisualizationConfig()
        >>> # Customize settings
        >>> config.line_width = 3
        >>> config.hud_bg_alpha = 0.8
    """

    # =========================================================================
    # LAYOUT
    # =========================================================================

    margin: int = 10
    """Margin from edge of frame in pixels."""

    panel_spacing: int = 15
    """Vertical spacing between HUD panels in pixels."""

    corner_radius: int = 5
    """Corner radius for rounded rectangles."""

    # =========================================================================
    # TEXT
    # =========================================================================

    font_path: str = "/usr/share/fonts/TTF/DejaVuSans.ttf"
    """Path to TTF font file (must support Cyrillic)."""

    font_scale: float = 0.6
    """Font scale factor for OpenCV putText."""

    font_thickness: int = 2
    """Thickness of text strokes."""

    font_color: tuple[int, int, int] = COLOR_WHITE
    """Default text color (BGR)."""

    # =========================================================================
    # SKELETON
    # =========================================================================

    line_width: int = 2
    """Width of skeleton connection lines."""

    joint_radius: int = 4
    """Radius of skeleton joint circles."""

    confidence_threshold: float = 0.5
    """Minimum confidence for drawing joints."""

    # Skeleton colors (H3.6M 17-keypoint format)
    color_left_side: tuple[int, int, int] = COLOR_BLUE
    """Color for left arm/leg joints."""

    color_right_side: tuple[int, int, int] = COLOR_RED
    """Color for right arm/leg joints."""

    color_center: tuple[int, int, int] = COLOR_GREEN
    """Color for center/torso joints."""

    color_joint: tuple[int, int, int] = COLOR_WHITE
    """Color for joint circles."""

    # =========================================================================
    # VELOCITY & TRAILS
    # =========================================================================

    velocity_scale: float = 5.0
    """Scaling factor for velocity vector length."""

    velocity_max_length: int = 50
    """Maximum length of velocity vectors in pixels."""

    trail_length: int = 20
    """Number of frames to keep in motion trail."""

    trail_color: tuple[int, int, int] = COLOR_CYAN
    """Color for motion trails."""

    trail_width: int = 2
    """Width of trail lines."""

    # =========================================================================
    # HUD
    # =========================================================================

    hud_bg_color: tuple[int, int, int] = COLOR_BLACK
    """HUD background color."""

    hud_bg_alpha: float = 0.6
    """HUD background transparency (0.0 = transparent, 1.0 = opaque)."""

    hud_text_color: tuple[int, int, int] = COLOR_WHITE
    """HUD text color."""

    hud_border_color: tuple[int, int, int] = COLOR_CYAN
    """HUD panel border color."""

    hud_padding: int = 10
    """Internal padding in HUD panels."""

    # =========================================================================
    # BLADE EDGE INDICATOR
    # =========================================================================

    blade_indicator_size: int = 30
    """Size of blade indicator arrow in pixels."""

    blade_indicator_thickness: int = 3
    """Thickness of blade indicator arrow."""

    # Blade edge colors (use module-level constants)
    blade_inside_color: tuple[int, int, int] = blade_inside_color  # type: ignore[assignment]
    """Color for inside edge indication."""

    blade_outside_color: tuple[int, int, int] = blade_outside_color  # type: ignore[assignment]
    """Color for outside edge indication."""

    blade_flat_color: tuple[int, int, int] = blade_flat_color  # type: ignore[assignment]
    """Color for flat edge indication."""

    blade_unknown_color: tuple[int, int, int] = blade_unknown_color  # type: ignore[assignment]
    """Color for unknown edge indication."""

    # =========================================================================
    # 3D VISUALIZATION
    # =========================================================================

    depth_near: float = 0.0
    """Near depth value for color mapping."""

    depth_far: float = 2.0
    """Far depth value for color mapping (meters)."""

    camera_distance: float = 3.0
    """Distance for 3D camera projection (meters)."""

    focal_length: int = 800
    """Focal length for 3D projection (pixels)."""

    # =========================================================================
    # AXIS INDICATOR
    # =========================================================================

    axis_length: int = 50
    """Length of spatial axes in pixels."""

    axis_thickness: int = 2
    """Thickness of spatial axis lines."""

    x_axis_color: tuple[int, int, int] = COLOR_RED
    """Color for X-axis (right)."""

    y_axis_color: tuple[int, int, int] = COLOR_GREEN
    """Color for Y-axis (down)."""

    z_axis_color: tuple[int, int, int] = COLOR_BLUE
    """Color for Z-axis (forward)."""

    # =========================================================================
    # SUBTITLES
    # =========================================================================

    subtitle_bg_color: tuple[int, int, int] = COLOR_BLACK
    """Subtitle background color."""

    subtitle_bg_alpha: float = 0.7
    """Subtitle background transparency."""

    subtitle_font_size: int = 32
    """Font size for subtitles (Pillow)."""

    subtitle_padding: int = 15
    """Padding around subtitle text."""


# =============================================================================
# LAYER CONFIGS (PREDEFINED)
# =============================================================================


@dataclass
class LayerConfigs:
    """Predefined configurations for standard visualization layers.

    This class provides ready-to-use configs for the 4-layer HUD system:
    - Layer 0: Raw skeleton only
    - Layer 1: + kinematics (velocity, trails)
    - Layer 2: + technical (edges, angles)
    - Layer 3: + coaching (subtitles, full HUD)
    """

    skeleton: LayerConfig = field(
        default_factory=lambda: LayerConfig(enabled=True, z_index=0, opacity=1.0)
    )
    """Skeleton overlay layer."""

    kinematics: LayerConfig = field(
        default_factory=lambda: LayerConfig(enabled=False, z_index=1, opacity=0.8)
    )
    """Velocity vectors and motion trails layer."""

    technical: LayerConfig = field(
        default_factory=lambda: LayerConfig(enabled=False, z_index=2, opacity=0.9)
    )
    """Edge indicators and joint angles layer."""

    coaching: LayerConfig = field(
        default_factory=lambda: LayerConfig(enabled=False, z_index=3, opacity=1.0)
    )
    """Subtitles and full HUD layer."""

    @classmethod
    def from_level(cls, level: int) -> "LayerConfigs":
        """Create layer configs from HUD level (0-3).

        Args:
            level: HUD level (0=skeleton, 1=+kinematics, 2=+technical, 3=+coaching)

        Returns:
            LayerConfigs with appropriate layers enabled.
        """
        configs = cls()

        if level >= 0:
            configs.skeleton.enabled = True
        if level >= 1:
            configs.kinematics.enabled = True
        if level >= 2:
            configs.technical.enabled = True
        if level >= 3:
            configs.coaching.enabled = True

        return configs


# =============================================================================
# THEME CONFIGS
# =============================================================================


@dataclass
class ThemeConfig:
    """Color theme configuration for visualization.

    Provides alternative color schemes for different contexts:
    - Default: Standard blue/red/green
    - Dark: High contrast for dark backgrounds
    - Light: Optimized for light backgrounds
    - Print: Grayscale for documentation
    """

    name: str = "default"
    """Theme name."""

    skeleton_left: tuple[int, int, int] = COLOR_BLUE
    skeleton_right: tuple[int, int, int] = COLOR_RED
    skeleton_center: tuple[int, int, int] = COLOR_GREEN
    skeleton_joint: tuple[int, int, int] = COLOR_WHITE

    velocity: tuple[int, int, int] = COLOR_YELLOW
    trail: tuple[int, int, int] = COLOR_CYAN

    hud_bg: tuple[int, int, int] = COLOR_BLACK
    hud_text: tuple[int, int, int] = COLOR_WHITE
    hud_border: tuple[int, int, int] = COLOR_CYAN

    blade_inside: tuple[int, int, int] = COLOR_GREEN
    blade_outside: tuple[int, int, int] = COLOR_RED
    blade_flat: tuple[int, int, int] = COLOR_YELLOW

    @classmethod
    def default_theme(cls) -> "ThemeConfig":
        """Default color theme (blue/red/green)."""
        return cls(name="default")

    @classmethod
    def dark_theme(cls) -> "ThemeConfig":
        """High contrast theme for dark backgrounds."""
        return cls(
            name="dark",
            skeleton_left=(200, 200, 255),
            skeleton_right=(255, 200, 200),
            skeleton_center=(200, 255, 200),
            skeleton_joint=(255, 255, 255),
            velocity=(255, 255, 0),
            trail=(0, 255, 255),
            hud_bg=(0, 0, 0),
            hud_text=(255, 255, 255),
            hud_border=(0, 255, 255),
        )

    @classmethod
    def light_theme(cls) -> "ThemeConfig":
        """Optimized theme for light backgrounds."""
        return cls(
            name="light",
            skeleton_left=(0, 0, 200),
            skeleton_right=(200, 0, 0),
            skeleton_center=(0, 150, 0),
            skeleton_joint=(50, 50, 50),
            velocity=(200, 150, 0),
            trail=(0, 150, 200),
            hud_bg=(255, 255, 255),
            hud_text=(0, 0, 0),
            hud_border=(0, 100, 100),
        )

    @classmethod
    def print_theme(cls) -> "ThemeConfig":
        """Grayscale theme for documentation."""
        gray_levels = [0, 64, 128, 192, 255]
        return cls(
            name="print",
            skeleton_left=(gray_levels[2], gray_levels[2], gray_levels[2]),
            skeleton_right=(gray_levels[1], gray_levels[1], gray_levels[1]),
            skeleton_center=(gray_levels[3], gray_levels[3], gray_levels[3]),
            skeleton_joint=(gray_levels[0], gray_levels[0], gray_levels[0]),
            velocity=(gray_levels[2], gray_levels[2], gray_levels[1]),
            trail=(gray_levels[1], gray_levels[2], gray_levels[2]),
            hud_bg=(gray_levels[4], gray_levels[4], gray_levels[4]),
            hud_text=(gray_levels[0], gray_levels[0], gray_levels[0]),
            hud_border=(gray_levels[1], gray_levels[1], gray_levels[1]),
        )


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================


def get_debug_config() -> VisualizationConfig:
    """Get configuration optimized for debugging.

    Returns:
        VisualizationConfig with higher visibility settings.
    """
    config = VisualizationConfig()
    config.line_width = 3
    config.joint_radius = 6
    config.font_scale = 0.7
    config.hud_bg_alpha = 0.8
    return config


def get_presentation_config() -> VisualizationConfig:
    """Get configuration optimized for presentation.

    Returns:
        VisualizationConfig with cleaner, professional appearance.
    """
    config = VisualizationConfig()
    config.line_width = 2
    config.joint_radius = 3
    config.font_scale = 0.5
    config.hud_bg_alpha = 0.5
    config.trail_length = 15
    return config


def get_analysis_config() -> VisualizationConfig:
    """Get configuration optimized for technical analysis.

    Returns:
        VisualizationConfig with detailed technical overlays.
    """
    config = VisualizationConfig()
    config.line_width = 2
    config.joint_radius = 4
    config.font_scale = 0.6
    config.velocity_scale = 10.0
    config.trail_length = 30
    return config
