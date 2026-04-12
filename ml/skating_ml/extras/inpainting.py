"""LAMA (Large Mask Inpainting) wrapper.

Uses ONNX Runtime. Input: RGB image + mask. Output: inpainted RGB image.

Model: LAMA Dilated (45.6M params, ~300MB VRAM)
Source: https://github.com/advimman/lama
ONNX: https://huggingface.co/Carve/LaMa-ONNX
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from skating_ml.extras.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "lama"
INPUT_SIZE = 512


class ImageInpainter:
    """Image inpainting via LAMA.

    Args:
        registry: ModelRegistry with "lama" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint masked regions of a frame.

        Args:
            frame: BGR image (H, W, 3) uint8.
            mask: Binary mask (H, W) bool, True = region to inpaint.

        Returns:
            Inpainted BGR image (H, W, 3) uint8.
        """
        h, w = frame.shape[:2]

        # Resize to model input (512x512)
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(
            mask.astype(np.uint8), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST
        )

        # Prepare inputs — LAMA expects RGB [0, 1] and mask [0, 1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = img_rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 512, 512)
        mask_tensor = (msk > 0).astype(np.float32)[np.newaxis, np.newaxis]  # (1, 1, 512, 512)

        inputs = {
            "image": img_tensor,
            "mask": mask_tensor,
        }

        output = self._session.run([self._output_name], inputs)[0]

        # Convert output to BGR uint8
        result = output[0].transpose(1, 2, 0)  # (512, 512, 3)
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Resize back to original
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

        # Composite: keep original where mask is False, use inpainted where True
        mask_3ch = np.stack([mask] * 3, axis=-1)
        final = np.where(mask_3ch, result, frame)

        return final
