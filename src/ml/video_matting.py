"""RobustVideoMatting wrapper for video background removal.

Uses ONNX Runtime. Input: RGB frame + optional mask. Output: alpha matte.

Model: RobustVideoMatting MobileNetV3 (~40MB VRAM)
Source: https://github.com/PeterL1n/RobustVideoMatting
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "video_matting"


class VideoMatting:
    """Video background removal via RobustVideoMatting.

    Args:
        registry: ModelRegistry with "video_matting" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        details = self._session.get_input_details()
        self._input_names = [d["name"] for d in details]
        self._r1 = None  # Recurrent state frame 1
        self._r2 = None  # Recurrent state frame 2
        self._downsample_ratio = 0.25

    def matting(self, frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Generate alpha matte for a frame.

        Args:
            frame: BGR image (H, W, 3) uint8.
            mask: Optional binary mask (H, W) to guide matting.

        Returns:
            Alpha matte (H, W) float32 in [0, 1].
        """
        h, w = frame.shape[:2]
        # RVM expects RGB
        src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = src[np.newaxis]  # (1, H, W, 3)

        inputs = {}
        for _i, name in enumerate(self._input_names):
            if "src" in name.lower() and "r1" not in name:
                inputs[name] = src
            elif "r1" in name.lower():
                inputs[name] = (
                    self._r1
                    if self._r1 is not None
                    else np.zeros((1, 1, h // 4, w // 4), dtype=np.float32)
                )
            elif "r2" in name.lower():
                inputs[name] = (
                    self._r2
                    if self._r2 is not None
                    else np.zeros((1, 1, h // 4, w // 4), dtype=np.float32)
                )
            elif "downsample" in name.lower():
                inputs[name] = np.array([self._downsample_ratio], dtype=np.float32)

        outputs = self._session.run(None, inputs)

        # Update recurrent states
        r1_idx, r2_idx = None, None
        for i, name in enumerate(self._input_names):
            if "r1" in name.lower():
                r1_idx = i
            elif "r2" in name.lower():
                r2_idx = i
        # Outputs typically include fgr, pha, r1, r2
        for out in outputs:
            if out.shape == (1, 1, h // 4, w // 4):
                if self._r1 is None:
                    self._r1 = out
                elif self._r2 is None:
                    self._r2 = out

        # Find alpha output
        alpha = np.ones((h, w), dtype=np.float32)
        for out in outputs:
            if out.ndim == 4 and out.shape[1] == 1:
                a = out[0, 0]  # (H, W)
                if a.shape[0] != h or a.shape[1] != w:
                    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
                alpha = np.clip(a, 0, 1)
                break

        return alpha

    def reset(self) -> None:
        """Reset recurrent states (call when starting a new video)."""
        self._r1 = None
        self._r2 = None
