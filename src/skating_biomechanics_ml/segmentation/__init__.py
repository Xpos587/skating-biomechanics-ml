"""Automatic video segmentation into skating elements.

This module provides functionality to automatically segment training videos
into individual skating element demonstrations for dataset building.
"""

from skating_biomechanics_ml.segmentation.element_segmenter import ElementSegmenter

__all__ = [
    "ElementSegmenter",
]
