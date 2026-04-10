"""Pose estimation module for figure skating analysis.

This module provides H3.6M 17-keypoint pose extraction as the primary format.
Uses RTMPose via rtmlib (HALPE26 26kp with foot keypoints) as the sole backend.

Architecture:
    Video -> RTMPoseExtractor (rtmlib BodyWithFeet) -> H3.6M 17kp + foot keypoints
"""

from src.pose_estimation.h36m import (
    H36M_KEYPOINT_NAMES,
    H36M_SKELETON_EDGES,
    H36Key,
)
from src.pose_estimation.normalizer import PoseNormalizer
from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor, extract_rtmpose_poses

__all__ = [
    "H36M_KEYPOINT_NAMES",
    "H36M_SKELETON_EDGES",
    "H36Key",
    "PoseNormalizer",
    "RTMPoseExtractor",
    "extract_rtmpose_poses",
]
