"""HUD (Heads-Up Display) components.

Provides modular components for rendering debug information overlays:
- Panel system for grouping related elements
- Individual HUD elements (frame counter, metrics display, etc.)
- Layout manager for positioning
"""

from skating_ml.visualization.hud.elements import (
    draw_blade_indicator_hud,
    draw_frame_counter,
    draw_metrics_panel,
    draw_phase_indicator,
)
from skating_ml.visualization.hud.layout import (
    HUDLayout,
    Position,
    get_default_layout,
)
from skating_ml.visualization.hud.panel import (
    HUDPanel,
    PanelPosition,
)

__all__ = [
    # Layout
    "HUDLayout",
    # Panel
    "HUDPanel",
    "PanelPosition",
    "Position",
    "draw_blade_indicator_hud",
    # Elements
    "draw_frame_counter",
    "draw_metrics_panel",
    "draw_phase_indicator",
    "get_default_layout",
]
