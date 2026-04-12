"""3D pose estimation using AthletePose3D models."""

# Re-export H3.6M types from pose_estimation module (primary source)
from skating_ml.pose_estimation import H36M_KEYPOINT_NAMES, H36M_SKELETON_EDGES, H36Key

from .anchor_projection import anchor_project, blend_by_confidence
from .athletepose_extractor import AthletePose3DExtractor, extract_3d_poses
from .corrective_pipeline import CorrectiveLens
from .kinematic_constraints import (
    apply_kinematic_constraints,
    enforce_bone_lengths,
    enforce_joint_angle_limits,
)
from .normalizer_3d import (
    Pose3DNormalizer,
    calculate_body_heights,
    get_head_center_3d,
    get_hip_center_3d,
)

__all__ = [
    "H36M_KEYPOINT_NAMES",
    "H36M_SKELETON_EDGES",
    "AthletePose3DExtractor",
    "CorrectiveLens",
    "H36Key",
    "Pose3DNormalizer",
    "anchor_project",
    "apply_kinematic_constraints",
    "blend_by_confidence",
    "calculate_body_heights",
    "enforce_bone_lengths",
    "enforce_joint_angle_limits",
    "extract_3d_poses",
    "get_head_center_3d",
    "get_hip_center_3d",
]
