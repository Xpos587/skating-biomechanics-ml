"""Multi-person detection and tracking module."""

from .blade_edge_detector_3d import BladeEdgeDetector3D
from .person_detector import BoundingBox, PersonDetector
from .pose_tracker import PoseTracker, Track
from .spatial_reference import CameraPose, SpatialReferenceDetector

__all__ = [
    "BladeEdgeDetector3D",
    "BoundingBox",
    "CameraPose",
    "PersonDetector",
    "PoseTracker",
    "SpatialReferenceDetector",
    "Track",
]
