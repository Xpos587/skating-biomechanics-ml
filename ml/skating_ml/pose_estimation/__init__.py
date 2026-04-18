"""Pose estimation module for figure skating analysis.

This module provides H3.6M 17-keypoint pose extraction as the primary format.
Uses RTMO via rtmlib (COCO 17kp) as the sole backend.

Architecture:
    Video -> PoseExtractor (rtmlib RTMO) -> H3.6M 17kp

Multi-GPU:
    Video -> MultiGPUPoseExtractor -> distribute across GPUs -> H3.6M 17kp
"""

from skating_ml.pose_estimation.h36m import (
    H36M_KEYPOINT_NAMES,
    H36M_SKELETON_EDGES,
    H36Key,
)
from skating_ml.pose_estimation.multi_gpu_extractor import MultiGPUPoseExtractor
from skating_ml.pose_estimation.normalizer import PoseNormalizer
from skating_ml.pose_estimation.pose_extractor import PoseExtractor, extract_poses

__all__ = [
    "H36M_KEYPOINT_NAMES",
    "H36M_SKELETON_EDGES",
    "H36Key",
    "MultiGPUPoseExtractor",
    "PoseExtractor",
    "PoseNormalizer",
    "extract_poses",
]
