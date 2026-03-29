"""Core visualization utilities.

This module provides foundational utilities for:
- Color gradients and palettes
- Text rendering with Cyrillic support
- Coordinate transformations

These are low-level utilities used by other visualization modules.
"""

from src.visualization.core.colors import (
    blend_colors,
    get_blade_color,
    get_depth_color,
    get_heatmap_color,
    interpolate_color,
)
from src.visualization.core.text import (
    draw_text_box,
    measure_text_size,
    render_cyrillic_text,
)
from src.visualization.core.geometry import (
    normalized_to_pixel,
    pixel_to_normalized,
    project_3d_to_2d,
)

__all__ = [
    # Colors
    "get_depth_color",
    "get_heatmap_color",
    "get_blade_color",
    "blend_colors",
    "interpolate_color",
    # Text
    "render_cyrillic_text",
    "draw_text_box",
    "measure_text_size",
    # Geometry
    "normalized_to_pixel",
    "pixel_to_normalized",
    "project_3d_to_2d",
]
