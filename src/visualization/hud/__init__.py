"""HUD (Heads-Up Display) components.

Provides modular components for rendering debug information overlays:
- Panel system for grouping related elements
- Individual HUD elements (frame counter, metrics display, etc.)
- Layout manager for positioning
"""

from src.visualization.hud.elements import (
    draw_blade_indicator_hud,
    draw_frame_counter,
    draw_metrics_panel,
    draw_phase_indicator,
)
from src.visualization.hud.layout import (
    HUDLayout,
    Position,
    get_default_layout,
)
from src.visualization.hud.panel import (
    HUDPanel,
    PanelPosition,
)

__all__ = [
    # Elements
    "draw_frame_counter",
    "draw_metrics_panel",
    "draw_phase_indicator",
    "draw_blade_indicator_hud",
    # Panel
    "HUDPanel",
    "PanelPosition",
    # Layout
    "HUDLayout",
    "Position",
    "get_default_layout",
]
