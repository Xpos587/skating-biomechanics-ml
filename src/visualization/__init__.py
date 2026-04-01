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

Example:
    >>> from src.visualization import VisualizationConfig, LayerConfigs, SkeletonLayer
    >>> config = VisualizationConfig()
    >>> layers = LayerConfigs.from_level(2)  # Enable skeleton + kinematics + technical
    >>> skeleton_layer = SkeletonLayer()
"""

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
    JointAngleLayer,
    Layer,
    LayerContext,
    SkeletonLayer,
    TimerLayer,
    TrailLayer,
    VelocityLayer,
    VerticalAxisLayer,
    render_layers,
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

__all__ = [
    "BladeLayer",
    "HUDLayer",
    "JointAngleLayer",
    # HUD
    "HUDLayout",
    "HUDPanel",
    # Layers
    "Layer",
    "LayerConfig",
    "LayerConfigs",
    "LayerContext",
    "PanelPosition",
    "Position",
    "SkeletonLayer",
    "ThemeConfig",
    "TimerLayer",
    "TrailLayer",
    "VelocityLayer",
    "VerticalAxisLayer",
    # Config
    "VisualizationConfig",
    "blend_colors",
    "draw_blade_indicator_hud",
    "draw_frame_counter",
    "draw_metrics_panel",
    "draw_phase_indicator",
    # Skeleton
    "draw_skeleton",
    "draw_skeleton_3d",
    "draw_skeleton_3d_pip",
    "draw_text_box",
    "get_analysis_config",
    "get_blade_color",
    "get_debug_config",
    "get_default_layout",
    # Core colors
    "get_depth_color",
    "get_heatmap_color",
    "get_joint_color",
    "get_joint_radius",
    "get_presentation_config",
    "get_skeleton_color",
    "interpolate_color",
    "measure_text_size",
    # Core geometry
    "normalized_to_pixel",
    "pixel_to_normalized",
    "project_3d_to_2d",
    # Core text
    "render_cyrillic_text",
    "render_layers",
]
