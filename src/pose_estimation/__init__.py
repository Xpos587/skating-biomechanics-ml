"""Pose estimation module for figure skating analysis.

This module provides H3.6M 17-keypoint pose extraction as the primary format.
Uses YOLO26-Pose backend for fast, accurate 2D pose estimation.

Architecture:
    Video → H36MExtractor (YOLO26-Pose) → H3.6M 17kp → 3D Lifter → 3D poses

Extractors:
    - H36MExtractor: YOLO26-Pose backend (17kp COCO → 17kp H3.6M integrated)
    - YOLOPoseExtractor: Raw YOLO26-Pose (17kp COCO format)
"""

from src.pose_estimation.h36m_extractor import (
    H36M_KEYPOINT_NAMES,
    H36M_SKELETON_EDGES,
    H36Key,
    H36MExtractor,
    extract_h36m_poses,
)
from src.pose_estimation.normalizer import PoseNormalizer
from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor, extract_rtmpose_poses
from src.pose_estimation.yolo_extractor import YOLOPoseExtractor

__all__ = [
    "H36M_KEYPOINT_NAMES",
    "H36M_SKELETON_EDGES",
    "H36Key",
    "H36MExtractor",
    "PoseNormalizer",
    "RTMPoseExtractor",
    "YOLOPoseExtractor",
    "extract_h36m_poses",
    "extract_rtmpose_poses",
]
