"""RobustVideoMatting wrapper for video background removal.

Uses ONNX Runtime. Input: RGB frame + optional mask. Output: alpha matte.

Model: RobustVideoMatting MobileNetV3 (~40MB VRAM)
Source: https://github.com/PeterL1n/RobustVideoMatting
ONNX: https://huggingface.co/LPDoctor/video_matting
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from skating_ml.extras.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "video_matting"
NUM_RECURRENT = 4  # r1, r2, r3, r4


class VideoMatting:
    """Video background removal via RobustVideoMatting.

    Args:
        registry: ModelRegistry with "video_matting" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._rec_states: list[np.ndarray] = []
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

        # RVM expects RGB, NCHW, float32 [0, 1]
        src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = src.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        inputs: dict[str, np.ndarray] = {}
        for name in self._input_names:
            if name == "src":
                inputs[name] = src
            elif name == "downsample_ratio":
                inputs[name] = np.array([self._downsample_ratio], dtype=np.float32)
            elif name.startswith("r") and name.endswith("i"):
                idx = int(name[1]) - 1
                if idx < len(self._rec_states):
                    inputs[name] = self._rec_states[idx]
                else:
                    inputs[name] = np.zeros((1, 1, 1, 1), dtype=np.float32)

        outputs = self._session.run(self._output_names, inputs)

        # Extract outputs by name
        output_map = dict(zip(self._output_names, outputs, strict=True))

        # Update recurrent states
        self._rec_states = []
        for i in range(NUM_RECURRENT):
            key = f"r{i + 1}o"
            if key in output_map:
                self._rec_states.append(output_map[key].copy())

        # Extract alpha matte
        alpha = np.ones((h, w), dtype=np.float32)
        if "pha" in output_map:
            a = output_map["pha"][0, 0]  # (H, W)
            if a.shape[0] != h or a.shape[1] != w:
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = np.clip(a, 0, 1)

        return alpha

    def reset(self) -> None:
        """Reset recurrent states (call when starting a new video)."""
        self._rec_states = []
