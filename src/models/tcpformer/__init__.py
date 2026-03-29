"""TCPFormer 3D pose lifting model.

Memory-Induced Transformer for monocular 3D human pose estimation.
Based on: https://github.com/AsukaCamellia/TCPFormer

Architecture:
- DSTFormerBlock for spatial attention
- MemoryInducedBlock for temporal modeling
- Adaptive fusion of attention and graph branches
"""

from .TCPFormer import MemoryInducedTransformer

__all__ = ["MemoryInducedTransformer"]
