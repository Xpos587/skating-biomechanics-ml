"""Skeleton drawing module.

Provides functions for drawing human skeleton overlays on video frames.
Supports both 2D and 3D H3.6M 17-keypoint format.
"""

from skating_ml.visualization.skeleton.drawer import (
    draw_skeleton,
    draw_skeleton_3d,
    draw_skeleton_3d_pip,
)
from skating_ml.visualization.skeleton.joints import (
    get_joint_color,
    get_joint_radius,
    get_skeleton_color,
)

__all__ = [
    # Drawer
    "draw_skeleton",
    "draw_skeleton_3d",
    "draw_skeleton_3d_pip",
    # Joints
    "get_joint_color",
    "get_joint_radius",
    "get_skeleton_color",
]
