"""NeuFlowV2 optical flow wrapper.

Uses ONNX Runtime for inference. Input: frame pair (BGR). Output: dense flow field (H, W, 2).

Model: NeuFlowV2 (mixed training)
Source: https://github.com/neufieldrobotics/NeuFlow_v2
ONNX: https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow

NOTE: The ONNX model has fixed input size 432x768. Frames are resized to this
resolution before inference, and the flow field is resized back to the original size.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from skating_ml.extras.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "optical_flow"
FLOW_HEIGHT = 432
FLOW_WIDTH = 768


class OpticalFlowEstimator:
    """Dense optical flow estimation via NeuFlowV2.

    Args:
        registry: ModelRegistry with "optical_flow" registered.

    Supports two usage patterns:
    - ``estimate(frame1, frame2)`` -- explicit frame pair
    - ``estimate_from_previous(frame)`` -- caches previous frame automatically
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._prev_frame: np.ndarray | None = None
        self._input_names = [i.name for i in self._session.get_inputs()]

    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Estimate optical flow between two frames.

        Args:
            frame1: BGR image (H, W, 3) uint8.
            frame2: BGR image (H, W, 3) uint8, same size as frame1.

        Returns:
            Flow field (H, W, 2) float32, resized to match input frame size.

        Raises:
            ValueError: If frames have different sizes.
        """
        if frame1.shape[:2] != frame2.shape[:2]:
            raise ValueError(
                f"Frames must have the same size: {frame1.shape[:2]} vs {frame2.shape[:2]}"
            )

        h, w = frame1.shape[:2]

        # Resize to model input size (432x768)
        img1 = cv2.resize(frame1, (FLOW_WIDTH, FLOW_HEIGHT), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(frame2, (FLOW_WIDTH, FLOW_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB, HWC -> NCHW, normalize to [0, 1]
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0

        inputs = {
            self._input_names[0]: img1[np.newaxis],  # (1, 3, 432, 768)
            self._input_names[1]: img2[np.newaxis],
        }

        # Inference — output shape: (1, 2, 432, 768)
        output = self._session.run(None, inputs)[0]

        # (1, 2, H, W) -> (H, W, 2)
        flow = output[0].transpose(1, 2, 0).astype(np.float32)

        # Resize flow back to original frame size
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        return flow

    def estimate_from_previous(self, frame: np.ndarray) -> np.ndarray | None:
        """Estimate flow from previously cached frame.

        On first call, caches the frame and returns None.
        On subsequent calls, estimates flow between previous and current frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            Flow field (H, W, 2) float32, or None on first call.
        """
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return None

        flow = self.estimate(self._prev_frame, frame)
        self._prev_frame = frame.copy()
        return flow

    def reset(self) -> None:
        """Clear cached previous frame."""
        self._prev_frame = None
