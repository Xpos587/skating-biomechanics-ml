"""SAM 2 wrapper for image segmentation.

Uses ONNX Runtime for inference. Input: RGB image + point prompt. Output: binary mask.

Model: SAM 2.1 Tiny (38.9M params, ~200MB VRAM)
Source: https://github.com/facebookresearch/sam2
ONNX: https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX

The ONNX export consists of two models:
- vision_encoder.onnx: extracts image embeddings (input: pixel_values)
- prompt_encoder_mask_decoder.onnx: takes embeddings + prompts, produces masks
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from skating_ml.extras.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "segment_anything"
VISION_ENCODER_ID = "segment_anything_ve"
PROMPT_DECODER_ID = "segment_anything_pd"
INPUT_SIZE = 1024


class SegmentAnything:
    """Image segmentation via SAM 2.1.

    Args:
        registry: ModelRegistry with "segment_anything_ve" and
            "segment_anything_pd" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._ve_session = registry.get(VISION_ENCODER_ID)
        self._pd_session = registry.get(PROMPT_DECODER_ID)
        self._input_size = INPUT_SIZE

        # Vision encoder I/O
        self._ve_input_name = self._ve_session.get_inputs()[0].name
        self._ve_output_names = [o.name for o in self._ve_session.get_outputs()]

        # Prompt decoder I/O
        self._pd_output_names = [o.name for o in self._pd_session.get_outputs()]

    def segment(
        self,
        frame: np.ndarray,
        point: tuple[int, int] | None = None,
        box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray | None:
        """Segment the image using a point or box prompt.

        Args:
            frame: BGR image (H, W, 3) uint8.
            point: (x, y) pixel coordinate as prompt, or None.
            box: (x1, y1, x2, y2) pixel box as prompt, or None.

        Returns:
            Binary mask (H, W) bool, or None if no prompt provided.
        """
        if point is None and box is None:
            return None

        h, w = frame.shape[:2]

        # Prepare image — SAM2 expects RGB, normalized with ImageNet stats
        img = cv2.resize(
            frame, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR
        )
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) - [123.675, 116.28, 103.53]) / [
            58.395,
            57.12,
            57.375,
        ]
        img_tensor = img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        # Step 1: Vision encoder — extract image embeddings
        ve_outputs = self._ve_session.run(self._ve_output_names, {self._ve_input_name: img_tensor})

        # Build embeddings dict from vision encoder outputs
        embeddings = {}
        for i, out in enumerate(ve_outputs):
            name = self._ve_output_names[i]
            embeddings[name] = out

        # Step 2: Prompt encoder + mask decoder
        pd_inputs: dict[str, np.ndarray] = {}

        # Add image embeddings
        for name, value in embeddings.items():
            pd_inputs[name] = value

        # Prepare point prompt
        # Shape: [batch_size=1, 1, num_points, 2]
        point_coords = np.zeros((1, 1, 0, 2), dtype=np.float32)
        point_labels = np.zeros((1, 1, 0), dtype=np.int64)

        if point is not None:
            sx = self._input_size / w
            sy = self._input_size / h
            point_coords = np.array(
                [[[point[0] * sx, point[1] * sy]]], dtype=np.float32
            )  # (1, 1, 1, 2)
            point_labels = np.array([[1]], dtype=np.int64)  # (1, 1) foreground

        pd_inputs["input_points"] = point_coords
        pd_inputs["input_labels"] = point_labels

        # Box prompt (optional)
        if box is not None:
            sx = self._input_size / w
            sy = self._input_size / h
            box_coords = np.array(
                [[box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]], dtype=np.float32
            )  # (1, 4)
            pd_inputs["input_boxes"] = box_coords
        else:
            pd_inputs["input_boxes"] = np.zeros((1, 0, 4), dtype=np.float32)

        try:
            outputs = self._pd_session.run(self._pd_output_names, pd_inputs)

            # Find pred_masks output
            pred_masks = None
            for name, out in zip(self._pd_output_names, outputs, strict=True):
                if "pred_masks" in name:
                    pred_masks = out
                    break

            if pred_masks is None:
                logger.warning("SAM2: no pred_masks in output")
                return None

            # pred_masks shape: [batch, num_prompts, num_masks, H, W]
            # Take first prompt, first (best) mask
            mask = pred_masks[0, 0, 0]  # (H_in, W_in)
            mask = mask > 0.0  # threshold

            # Resize to original frame size
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            return mask
        except Exception as e:
            logger.warning("SAM2 inference failed: %s", e)
            return None
