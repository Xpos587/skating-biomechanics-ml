"""Core visualization utilities.

This module provides foundational utilities for:
- Color gradients and palettes
- Text rendering with Cyrillic support
- Coordinate transformations
- Semi-transparent overlay primitives

These are low-level utilities used by other visualization modules.
"""

from skating_ml.visualization.core.colors import (
    blend_colors,
    get_blade_color,
    get_depth_color,
    get_heatmap_color,
    interpolate_color,
)
from skating_ml.visualization.core.geometry import (
    normalized_to_pixel,
    pixel_to_normalized,
    project_3d_to_2d,
)
from skating_ml.visualization.core.overlay import draw_overlay_rect
from skating_ml.visualization.core.text import (
    draw_text_box,
    measure_text_size,
    put_text,
    render_cyrillic_text,
)

__all__ = [
    "blend_colors",
    "draw_overlay_rect",
    "draw_text_box",
    "get_blade_color",
    # Colors
    "get_depth_color",
    "get_heatmap_color",
    "interpolate_color",
    "measure_text_size",
    # Geometry
    "normalized_to_pixel",
    "pixel_to_normalized",
    "project_3d_to_2d",
    # Text
    "put_text",
    "render_cyrillic_text",
]
