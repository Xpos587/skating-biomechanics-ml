"""Алгоритмы трекинга людей для мульти-персональной ассоциации поз.

Предоставляет пос frame-to-frame реидентификацию:
- Sports2D: Венгерский алгоритм по расстояниям ключевых точек (scipy)
- DeepSORT: Appearance-based ReID (требуется deep-sort-realtime)
- SkeletalIdentity: 3D bone length biometric profiles for re-ID
- TrackletMerger: post-hoc tracklet merging for occlusion recovery
"""

from .deepsort_tracker import DeepSORTTracker
from .skeletal_identity import (
    SkeletalIdentityExtractor,
    compute_2d_skeletal_ratios,
    compute_bone_lengths_3d,
    compute_identity_profile,
    identity_similarity,
)
from .sports2d import Sports2DTracker
from .tracklet_merger import (
    Tracklet,
    TrackletMerger,
    build_tracklets,
)

__all__ = [
    "DeepSORTTracker",
    "SkeletalIdentityExtractor",
    "Sports2DTracker",
    "Tracklet",
    "TrackletMerger",
    "build_tracklets",
    "compute_2d_skeletal_ratios",
    "compute_bone_lengths_3d",
    "compute_identity_profile",
    "identity_similarity",
]
