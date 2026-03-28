"""MotionAGFormer model architecture.

Adapted from https://github.com/TaatiTeam/MotionAGFormer
WACV 2024 - "MotionAGFormer: Enhancing 3D Pose Estimation with a Transformer-GCNFormer Network"
"""

from .MotionAGFormer import MotionAGFormer, MotionAGFormerBlock, create_layers

__all__ = ["MotionAGFormer", "MotionAGFormerBlock", "create_layers"]
