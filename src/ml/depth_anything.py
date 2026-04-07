"""Depth Anything V2 wrapper for monocular depth estimation.

Uses ONNX Runtime for inference. Input: RGB frame. Output: relative depth map (H, W).

Model: Depth Anything V2 Small (24.8M params, ~200MB VRAM)
Source: https://github.com/DepthAnything/Depth-Anything-V2
ONNX: https://huggingface.co/DepthAnything/Depth-Anything-V2-Small
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ml.model_registry import ModelRegistry  # noqa: TC001 — used at runtime

logger = logging.getLogger(__name__)

MODEL_ID = "depth_anything"
INPUT_SIZE = 518


class DepthEstimator:
    """Monocular depth estimation via Depth Anything V2.

    Args:
        registry: ModelRegistry with "depth_anything" registered.
            The model must be registered before construction::

                reg.register("depth_anything", vram_mb=200, path="data/models/depth_anything_v2_small.onnx")
                est = DepthEstimator(reg)
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._input_size = INPUT_SIZE
        # Infer input name from session
        details = self._session.get_input_details()
        self._input_name = details[0]["name"]

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map for a single frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            Depth map (H, W) float32 normalized to [0, 1].
        """
        h, w = frame.shape[:2]

        # Resize to model input, keep aspect ratio with padding
        img = cv2.resize(
            frame, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR
        )

        # BGR -> RGB, HWC -> NCHW, normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]  # (1, 3, H, W)

        # Inference
        output = self._session.run(None, {self._input_name: blob})[0]

        # Output shape: (1, H_out, W_out) -> (H_out, W_out)
        depth = output[0].astype(np.float32)

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)

        # Resize back to original frame size
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth
