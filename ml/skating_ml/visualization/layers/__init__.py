"""Visualization layer system.

Provides modular layer-based rendering:
- Base layer class
- Skeleton overlay layer
- Velocity vectors layer
- Motion trails layer
- Blade indicator layer
- HUD layer
- Vertical axis layer
- Joint angle layer
- Timer layer
- Angle panel layer
"""

from skating_ml.visualization.layers.angle_panel_layer import AnglePanelLayer
from skating_ml.visualization.layers.base import Layer, LayerContext, render_layers
from skating_ml.visualization.layers.blade_layer import BladeLayer
from skating_ml.visualization.layers.depth_layer import DepthMapLayer
from skating_ml.visualization.layers.foot_tracker_layer import FootTrackerLayer
from skating_ml.visualization.layers.hud_layer import HUDLayer
from skating_ml.visualization.layers.joint_angle_layer import JointAngleLayer
from skating_ml.visualization.layers.matting_layer import MattingLayer
from skating_ml.visualization.layers.optical_flow_layer import OpticalFlowLayer
from skating_ml.visualization.layers.segmentation_layer import SegmentationMaskLayer
from skating_ml.visualization.layers.skeleton_layer import SkeletonLayer
from skating_ml.visualization.layers.timer_layer import TimerLayer
from skating_ml.visualization.layers.trail_layer import TrailLayer
from skating_ml.visualization.layers.velocity_layer import VelocityLayer
from skating_ml.visualization.layers.vertical_axis_layer import VerticalAxisLayer

__all__ = [
    "AnglePanelLayer",
    "BladeLayer",
    "DepthMapLayer",
    "FootTrackerLayer",
    "HUDLayer",
    "JointAngleLayer",
    "Layer",
    "LayerContext",
    "MattingLayer",
    "OpticalFlowLayer",
    "SegmentationMaskLayer",
    "SkeletonLayer",
    "TimerLayer",
    "TrailLayer",
    "VelocityLayer",
    "VerticalAxisLayer",
    "render_layers",
]
