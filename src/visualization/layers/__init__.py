"""Visualization layer system.

Provides modular layer-based rendering:
- Base layer class
- Skeleton overlay layer
- Velocity vectors layer
- Motion trails layer
- Blade indicator layer
- HUD layer
"""

from src.visualization.layers.base import Layer, LayerContext, render_layers
from src.visualization.layers.blade_layer import BladeLayer
from src.visualization.layers.hud_layer import HUDLayer
from src.visualization.layers.skeleton_layer import SkeletonLayer
from src.visualization.layers.trail_layer import TrailLayer
from src.visualization.layers.velocity_layer import VelocityLayer

__all__ = [
    # Base
    "Layer",
    "LayerContext",
    "render_layers",
    # Layers
    "SkeletonLayer",
    "VelocityLayer",
    "TrailLayer",
    "BladeLayer",
    "HUDLayer",
]
