"""FootTrackNet wrapper for specialized foot detection.

Uses ONNX Runtime. Input: RGB image. Output: person + foot bounding boxes.

Model: FootTrackNet (2.53M params, ~30MB VRAM)
Source: Qualcomm AI Hub — https://aihub.qualcomm.com/iot/models/foot_track_net
ONNX: Export via qai-hub-models package (see download script)

The model uses a CenterNet-style architecture with heatmap, bbox regression,
and landmark outputs at stride 4.

Output classes: 0=face, 1=person (class 2 heatmap channel may exist for feet).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from skating_ml.extras.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_ID = "foot_tracker"
STRIDE = 4
INPUT_H = 480
INPUT_W = 640


class FootTracker:
    """Person and foot detection via FootTrackNet.

    Args:
        registry: ModelRegistry with "foot_tracker" registered.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._session = registry.get(MODEL_ID)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect persons and feet in a frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            List of dicts with keys: ``bbox`` (x1,y1,x2,y2), ``class_id``
            (0=face, 1=person), ``confidence`` (float).
        """
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)

        # Prepare input: RGB, [0, 1], NCHW
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = img_rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 480, 640)

        outputs = self._session.run(self._output_names, {self._input_name: blob})

        # Build output map
        output_map = dict(zip(self._output_names, outputs, strict=True))

        detections: list[dict] = []

        # Try CenterNet-style decoding: heatmap + bbox regression
        heatmap = None
        bbox_reg = None
        for name, out in output_map.items():
            name_lower = name.lower()
            if "heatmap" in name_lower:
                heatmap = out  # (num_classes, H/stride, W/stride)
            elif "bbox" in name_lower or "reg" in name_lower:
                bbox_reg = out  # (num_classes*4, H/stride, W/stride)

        if heatmap is not None and bbox_reg is not None:
            detections = self._decode_centernet(heatmap, bbox_reg, h, w)
        else:
            # Fallback: try raw detection format (N, 6+)
            raw = next(iter(output_map.values()))
            if raw.ndim == 2 and raw.shape[1] >= 6:
                for det in raw:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    if conf > 0.3:
                        sx, sy = w / INPUT_W, h / INPUT_H
                        detections.append(
                            {
                                "bbox": [
                                    float(x1 * sx),
                                    float(y1 * sy),
                                    float(x2 * sx),
                                    float(y2 * sy),
                                ],
                                "class_id": int(cls),
                                "confidence": float(conf),
                            }
                        )

        return detections

    def _decode_centernet(
        self,
        heatmap: np.ndarray,
        bbox_reg: np.ndarray,
        orig_h: int,
        orig_w: int,
        score_threshold: float = 0.5,
    ) -> list[dict]:
        """Decode CenterNet-style heatmap + bbox regression outputs."""
        from scipy.ndimage import maximum_filter

        detections: list[dict] = []
        num_classes = heatmap.shape[0]
        fm_h, fm_w = heatmap.shape[1], heatmap.shape[2]

        for cls_id in range(min(num_classes, 2)):  # face=0, person=1
            hm = heatmap[cls_id]  # (fm_h, fm_w)

            # Apply sigmoid if not already
            if hm.max() > 1.0:
                hm = 1.0 / (1.0 + np.exp(-hm))

            # NMS via local maximum
            hm_max = maximum_filter(hm, size=3)
            peaks = (hm == hm_max) & (hm > score_threshold)

            ys, xs = np.where(peaks)
            for y, x in zip(ys, xs, strict=True):
                conf = float(hm[y, x])
                # Get bbox offsets (4 values per class)
                offset_idx = cls_id * 4
                top = bbox_reg[offset_idx, y, x]
                left = bbox_reg[offset_idx + 1, y, x]
                right = bbox_reg[offset_idx + 2, y, x]
                bottom = bbox_reg[offset_idx + 3, y, x]

                # Convert from feature map coords to image coords
                cx = (x + 0.5) * STRIDE
                cy = (y + 0.5) * STRIDE
                x1 = cx - left * STRIDE
                y1 = cy - top * STRIDE
                x2 = cx + right * STRIDE
                y2 = cy + bottom * STRIDE

                # Scale to original frame size
                sx = orig_w / INPUT_W
                sy = orig_h / INPUT_H
                detections.append(
                    {
                        "bbox": [float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)],
                        "class_id": cls_id,
                        "confidence": conf,
                    }
                )

        return detections
