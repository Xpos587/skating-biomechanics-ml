"""Visualization system for figure skating biomechanics analysis.

This package provides a modular, configurable visualization system for:
- Skeleton overlay (2D and 3D)
- Velocity vectors and motion trails
- Blade edge indicators
- HUD with biomechanics metrics
- Multi-layer debugging interface

Architecture:
    config.py        - Configuration classes (VisualizationConfig, LayerConfig, ThemeConfig)
    core/            - Core utilities (colors, text, geometry)
    skeleton/        - Skeleton drawing
    hud/             - HUD components
    layers/          - Visualization layers
    renderers/       - 2D/3D renderers (TODO)

Example:
    >>> from src.visualization import VisualizationConfig, LayerConfigs, SkeletonLayer
    >>> config = VisualizationConfig()
    >>> layers = LayerConfigs.from_level(2)  # Enable skeleton + kinematics + technical
    >>> skeleton_layer = SkeletonLayer()
"""

# =============================================================================
# NEW MODULAR API
# =============================================================================

# Config
from src.visualization.config import (
    LayerConfig,
    LayerConfigs,
    ThemeConfig,
    VisualizationConfig,
    get_analysis_config,
    get_debug_config,
    get_presentation_config,
)

# Core
from src.visualization.core.colors import (
    blend_colors,
    get_blade_color,
    get_depth_color,
    get_heatmap_color,
    interpolate_color,
)
from src.visualization.core.geometry import (
    normalized_to_pixel,
    pixel_to_normalized,
    project_3d_to_2d,
)
from src.visualization.core.text import (
    draw_text_box,
    measure_text_size,
    render_cyrillic_text,
)

# Skeleton
from src.visualization.skeleton import (
    draw_skeleton,
    draw_skeleton_3d,
    draw_skeleton_3d_pip,
    get_joint_color,
    get_joint_radius,
    get_skeleton_color,
)

# HUD
from src.visualization.hud import (
    HUDLayout,
    HUDPanel,
    PanelPosition,
    Position,
    draw_blade_indicator_hud,
    draw_frame_counter,
    draw_metrics_panel,
    draw_phase_indicator,
    get_default_layout,
)

# Layers
from src.visualization.layers import (
    BladeLayer,
    HUDLayer,
    Layer,
    LayerContext,
    SkeletonLayer,
    TrailLayer,
    VelocityLayer,
    render_layers,
)

# =============================================================================
# LEGACY API (for backward compatibility)
# =============================================================================

# Import legacy functions from old visualization.py module
# These are maintained for backward compatibility with existing tests
# Import using importlib to avoid circular import
import importlib.util
import sys

try:
    # Load the old visualization.py as a separate module
    spec = importlib.util.spec_from_file_location(
        "src.visualization_legacy", "src/visualization.py"
    )
    if spec and spec.loader:
        _legacy_viz = importlib.util.module_from_spec(spec)
        sys.modules["src.visualization_legacy"] = _legacy_viz
        spec.loader.exec_module(_legacy_viz)

        # Re-export legacy functions
        draw_debug_hud = _legacy_viz.draw_debug_hud
        draw_edge_indicators = _legacy_viz.draw_edge_indicators
        draw_subtitle_cyrillic = _legacy_viz.draw_subtitle_cyrillic
        draw_trails = _legacy_viz.draw_trails
        draw_velocity_vectors = _legacy_viz.draw_velocity_vectors
        draw_spatial_axes = _legacy_viz.draw_spatial_axes
        draw_3d_trajectory = _legacy_viz.draw_3d_trajectory
        draw_ice_trace = _legacy_viz.draw_ice_trace
        draw_blade_state_3d_hud = _legacy_viz.draw_blade_state_3d_hud
        draw_motion_direction_arrow = _legacy_viz.draw_motion_direction_arrow
        draw_axis_indicator = _legacy_viz.draw_axis_indicator
        calculate_trunk_angle = _legacy_viz.calculate_trunk_angle
    else:
        raise ImportError("Could not load visualization.py")
except Exception:
    # If old visualization.py is not available, provide stubs
    def _not_implemented(*args, **kwargs):
        raise NotImplementedError("Legacy function not available in modular visualization")

    draw_debug_hud = _not_implemented  # type: ignore
    draw_edge_indicators = _not_implemented  # type: ignore
    draw_subtitle_cyrillic = _not_implemented  # type: ignore
    draw_trails = _not_implemented  # type: ignore
    draw_velocity_vectors = _not_implemented  # type: ignore
    draw_spatial_axes = _not_implemented  # type: ignore
    draw_3d_trajectory = _not_implemented  # type: ignore
    draw_ice_trace = _not_implemented  # type: ignore
    draw_blade_state_3d_hud = _not_implemented  # type: ignore
    draw_motion_direction_arrow = _not_implemented  # type: ignore
    draw_axis_indicator = _not_implemented  # type: ignore
    calculate_trunk_angle = _not_implemented  # type: ignore

__all__ = [
    # Config
    "VisualizationConfig",
    "LayerConfig",
    "LayerConfigs",
    "ThemeConfig",
    "get_debug_config",
    "get_presentation_config",
    "get_analysis_config",
    # Core colors
    "get_depth_color",
    "get_heatmap_color",
    "get_blade_color",
    "blend_colors",
    "interpolate_color",
    # Core geometry
    "normalized_to_pixel",
    "pixel_to_normalized",
    "project_3d_to_2d",
    # Core text
    "render_cyrillic_text",
    "draw_text_box",
    "measure_text_size",
    # Skeleton
    "draw_skeleton",
    "draw_skeleton_3d",
    "draw_skeleton_3d_pip",
    "get_joint_color",
    "get_joint_radius",
    "get_skeleton_color",
    # HUD
    "HUDLayout",
    "HUDPanel",
    "PanelPosition",
    "Position",
    "draw_frame_counter",
    "draw_metrics_panel",
    "draw_phase_indicator",
    "draw_blade_indicator_hud",
    "get_default_layout",
    # Layers
    "Layer",
    "LayerContext",
    "SkeletonLayer",
    "VelocityLayer",
    "TrailLayer",
    "BladeLayer",
    "HUDLayer",
    "render_layers",
    # Legacy (for backward compatibility)
    "draw_debug_hud",
    "draw_edge_indicators",
    "draw_subtitle_cyrillic",
    "draw_trails",
    "draw_velocity_vectors",
    "draw_spatial_axes",
    "draw_3d_trajectory",
    "draw_ice_trace",
    "draw_blade_state_3d_hud",
    "draw_motion_direction_arrow",
    "draw_axis_indicator",
    "calculate_trunk_angle",
]
