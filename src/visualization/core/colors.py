"""Color utilities for visualization.

Provides functions for:
- Depth-based color coding (3D visualization)
- Heatmap gradients (velocity, acceleration)
- Blade edge state colors
- Color blending and interpolation
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.types import BladeType
from src.visualization.config import (
    DEPTH_COLORS,
    blade_flat_color,
    blade_inside_color,
    blade_outside_color,
    blade_unknown_color,
)

# =============================================================================
# TYPE ALIASES
# =============================================================================

Color = tuple[int, int, int]  # BGR format for OpenCV
ColorArray = NDArray[np.int32]  # Shape (..., 3) for vectorized operations


# =============================================================================
# DEPTH COLOR CODING
# =============================================================================


def get_depth_color(
    depth: float,
    depth_min: float = 0.0,
    depth_max: float = 2.0,
    color_map: list[Color] | None = None,
) -> Color:
    """Convert depth value to color using gradient.

    Maps depth to a color gradient from near (red) to far (blue).
    Useful for 3D skeleton visualization where Z-axis indicates depth.

    Args:
        depth: Depth value in meters.
        depth_min: Minimum depth for color mapping (meters).
        depth_max: Maximum depth for color mapping (meters).
        color_map: List of BGR colors for gradient (default: red->orange->yellow->green->blue).

    Returns:
        BGR color tuple for the depth value.

    Example:
        >>> get_depth_color(0.5, 0.0, 2.0)
        (255, 128, 0)  # Orange (medium-near)
    """
    if color_map is None:
        color_map = DEPTH_COLORS

    # Clamp depth to range
    depth_clamped = max(depth_min, min(depth_max, depth))

    # Normalize to [0, 1]
    t = (depth_clamped - depth_min) / (depth_max - depth_min) if depth_max > depth_min else 0.5

    # Map to color gradient
    num_colors = len(color_map)
    idx = t * (num_colors - 1)
    idx_low = int(idx)
    idx_high = min(idx_low + 1, num_colors - 1)

    # Interpolate between adjacent colors
    if idx_low == idx_high:
        return color_map[idx_low]

    local_t = idx - idx_low
    return interpolate_color(color_map[idx_low], color_map[idx_high], local_t)


def get_depth_colors_vectorized(
    depths: NDArray[np.float32],
    depth_min: float = 0.0,
    depth_max: float = 2.0,
) -> ColorArray:
    """Vectorized depth-to-color conversion for arrays.

    Args:
        depths: Array of depth values (N,) or (N, M).
        depth_min: Minimum depth for color mapping.
        depth_max: Maximum depth for color mapping.

    Returns:
        Array of BGR colors with shape (..., 3).

    Example:
        >>> depths = np.array([0.0, 1.0, 2.0])
        >>> colors = get_depth_colors_vectorized(depths)
        >>> colors.shape
        (3, 3)
    """
    # Flatten for processing
    original_shape = depths.shape
    depths_flat = depths.ravel()

    # Normalize to [0, 1]
    if depth_max > depth_min:
        t = (depths_flat - depth_min) / (depth_max - depth_min)
    else:
        t = np.full_like(depths_flat, 0.5)

    t = np.clip(t, 0.0, 1.0)

    # Map to color gradient
    num_colors = len(DEPTH_COLORS)
    indices = t * (num_colors - 1)
    idx_low = indices.astype(int)
    idx_high = np.minimum(idx_low + 1, num_colors - 1)
    local_t = indices - idx_low

    # Vectorized interpolation
    colors = np.zeros((len(depths_flat), 3), dtype=np.int32)

    for i in range(len(depths_flat)):
        c1 = np.array(DEPTH_COLORS[idx_low[i]])
        c2 = np.array(DEPTH_COLORS[idx_high[i]])
        colors[i] = (c1 * (1 - local_t[i]) + c2 * local_t[i]).astype(np.int32)

    return colors.reshape((*original_shape, 3))


# =============================================================================
# HEATMAP COLORS
# =============================================================================


def get_heatmap_color(
    value: float,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "jet",
) -> Color:
    """Convert value to heatmap color.

    Supports common colormaps for scientific visualization:
    - jet: Blue -> cyan -> yellow -> red (classic)
    - viridis: Purple -> blue -> green -> yellow
    - magma: Black -> purple -> orange -> yellow
    - hot: Black -> red -> yellow -> white

    Args:
        value: Value to map to color.
        vmin: Minimum value for mapping.
        vmax: Maximum value for mapping.
        cmap: Colormap name ("jet", "viridis", "magma", "hot").

    Returns:
        BGR color tuple.

    Example:
        >>> get_heatmap_color(0.5, 0.0, 1.0, "jet")
        (0, 255, 255)  # Cyan (midpoint)
    """
    # Normalize to [0, 1]
    t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    t = max(0.0, min(1.0, t))

    # Colormap definitions (RGB, will convert to BGR)
    if cmap == "jet":
        # Blue -> cyan -> yellow -> red
        if t < 0.25:
            # Blue to cyan
            local_t = t / 0.25
            r, g, b = 0, int(255 * local_t), 255
        elif t < 0.5:
            # Cyan to yellow
            local_t = (t - 0.25) / 0.25
            r, g, b = int(255 * local_t), 255, int(255 * (1 - local_t))
        elif t < 0.75:
            # Yellow to red
            local_t = (t - 0.5) / 0.25
            r, g, b = 255, int(255 * (1 - local_t)), 0
        else:
            # Red to dark red
            local_t = (t - 0.75) / 0.25
            r, g, b = int(255 * (1 - local_t * 0.5)), 0, 0

    elif cmap == "viridis":
        # Purple -> blue -> green -> yellow
        if t < 0.33:
            local_t = t / 0.33
            r, g, b = int(68 + 39 * local_t), int(1 + 130 * local_t), int(119 + 59 * local_t)
        elif t < 0.66:
            local_t = (t - 0.33) / 0.33
            r, g, b = int(33 + 16 * local_t), int(144 + 60 * local_t), int(141 + 53 * local_t)
        else:
            local_t = (t - 0.66) / 0.34
            r, g, b = int(49 + 206 * local_t), int(204 + 51 * local_t), int(194 - 63 * local_t)

    elif cmap == "magma":
        # Black -> purple -> orange -> yellow
        if t < 0.33:
            local_t = t / 0.33
            r, g, b = int(3 + 13 * local_t), int(3 + 10 * local_t), int(13 + 35 * local_t)
        elif t < 0.66:
            local_t = (t - 0.33) / 0.33
            r, g, b = int(16 + 117 * local_t), int(13 + 34 * local_t), int(48 + 98 * local_t)
        else:
            local_t = (t - 0.66) / 0.34
            r, g, b = int(133 + 122 * local_t), int(47 + 158 * local_t), int(146 + 109 * local_t)

    elif cmap == "hot":
        # Black -> red -> yellow -> white
        if t < 0.33:
            local_t = t / 0.33
            r, g, b = int(255 * local_t), 0, 0
        elif t < 0.66:
            local_t = (t - 0.33) / 0.33
            r, g, b = 255, int(255 * local_t), 0
        else:
            local_t = (t - 0.66) / 0.34
            r, g, b = 255, 255, int(255 * local_t)

    else:
        # Default to grayscale
        gray = int(255 * t)
        r, g, b = gray, gray, gray

    # Convert RGB to BGR
    return (b, g, r)


# =============================================================================
# BLADE EDGE COLORS
# =============================================================================


def get_blade_color(
    blade_type: BladeType,
    inside_color: Color | None = None,
    outside_color: Color | None = None,
    flat_color: Color | None = None,
    unknown_color: Color | None = None,
) -> Color:
    """Get color for blade edge type.

    Args:
        blade_type: BladeType enum value.
        inside_color: Color for inside edge (default: green).
        outside_color: Color for outside edge (default: red).
        flat_color: Color for flat edge (default: yellow).
        unknown_color: Color for unknown edge (default: white).

    Returns:
        BGR color tuple for the blade type.

    Example:
        >>> from src.types import BladeType
        >>> get_blade_color(BladeType.INSIDE)
        (0, 255, 0)  # Green
    """
    if inside_color is None:
        inside_color = blade_inside_color
    if outside_color is None:
        outside_color = blade_outside_color
    if flat_color is None:
        flat_color = blade_flat_color
    if unknown_color is None:
        unknown_color = blade_unknown_color

    color_map = {
        BladeType.INSIDE: inside_color,
        BladeType.OUTSIDE: outside_color,
        BladeType.FLAT: flat_color,
        BladeType.TOE_PICK: outside_color,  # Use outside color for toe pick
        BladeType.UNKNOWN: unknown_color,
    }

    return color_map.get(blade_type, unknown_color)


# =============================================================================
# COLOR MANIPULATION
# =============================================================================


def interpolate_color(c1: Color, c2: Color, t: float) -> Color:
    """Linear interpolation between two colors.

    Args:
        c1: First color (BGR).
        c2: Second color (BGR).
        t: Interpolation factor [0, 1]. 0 = c1, 1 = c2.

    Returns:
        Interpolated BGR color tuple.

    Example:
        >>> interpolate_color((255, 0, 0), (0, 0, 255), 0.5)
        (128, 0, 128)  # Purple (midpoint of red and blue)
    """
    t = max(0.0, min(1.0, t))
    r = int(c1[2] * (1 - t) + c2[2] * t)
    g = int(c1[1] * (1 - t) + c2[1] * t)
    b = int(c1[0] * (1 - t) + c2[0] * t)
    return (b, g, r)


def blend_colors(
    colors: list[Color],
    weights: list[float] | None = None,
) -> Color:
    """Blend multiple colors with optional weights.

    Args:
        colors: List of BGR colors to blend.
        weights: Optional weights for each color (default: equal).

    Returns:
        Blended BGR color tuple.

    Raises:
        ValueError: If colors and weights have different lengths.

    Example:
        >>> blend_colors([(255, 0, 0), (0, 0, 255)], [0.7, 0.3])
        (77, 0, 179)  # Red-weighted purple
    """
    if weights is None:
        weights = [1.0 / len(colors)] * len(colors)

    if len(colors) != len(weights):
        raise ValueError("colors and weights must have same length")

    total_weight = sum(weights)
    if total_weight == 0:
        return (0, 0, 0)

    r = sum(c[2] * w for c, w in zip(colors, weights, strict=False)) / total_weight
    g = sum(c[1] * w for c, w in zip(colors, weights, strict=False)) / total_weight
    b = sum(c[0] * w for c, w in zip(colors, weights, strict=False)) / total_weight

    return (int(b), int(g), int(r))


def fade_color(color: Color, alpha: float) -> Color:
    """Fade a color towards black by alpha factor.

    Args:
        color: BGR color to fade.
        alpha: Fade factor [0, 1]. 0 = black, 1 = original color.

    Returns:
        Faded BGR color tuple.

    Example:
        >>> fade_color((255, 0, 0), 0.5)
        (128, 0, 0)  # Dimmed red
    """
    alpha = max(0.0, min(1.0, alpha))
    return (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))


def complementary_color(color: Color) -> Color:
    """Get complementary color (RGB inversion).

    Args:
        color: BGR color tuple.

    Returns:
        Complementary BGR color tuple.

    Example:
        >>> complementary_color((255, 0, 0))  # Blue
        (0, 255, 255)  # Yellow (complementary)
    """
    return (255 - color[0], 255 - color[1], 255 - color[2])


# =============================================================================
# PRESET COLOR PALETTES
# =============================================================================

COLOR_PALETTE_MATPLOTLIB: Final[dict[str, list[Color]]] = {
    "tab10": [
        (31, 119, 180),  # Blue
        (255, 127, 14),  # Orange
        (44, 160, 44),  # Green
        (214, 39, 39),  # Red
        (148, 103, 189),  # Purple
        (140, 86, 75),  # Brown
        (227, 119, 194),  # Pink
        (127, 127, 127),  # Gray
        (188, 189, 34),  # Olive
        (23, 190, 207),  # Cyan
    ],
    "Set2": [
        (102, 194, 165),
        (252, 141, 98),
        (141, 160, 203),
        (231, 138, 195),
        (166, 216, 84),
        (255, 217, 47),
        (229, 196, 148),
        (179, 179, 179),
    ],
}


def get_palette_color(index: int, palette: str = "tab10") -> Color:
    """Get color from a palette by index.

    Args:
        index: Color index (wraps around if exceeds palette size).
        palette: Palette name ("tab10", "Set2").

    Returns:
        BGR color from palette.

    Example:
        >>> get_palette_color(0, "tab10")
        (180, 119, 31)  # First color in tab10
    """
    palettes = COLOR_PALETTE_MATPLOTLIB

    if palette not in palettes:
        palette = "tab10"

    colors = palettes[palette]
    return colors[index % len(colors)]
