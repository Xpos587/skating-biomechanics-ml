"""Foot tracking visualization layer.

Renders bounding boxes around detected persons.
Reads detections from ``LayerContext.custom_data["foot_detections"]``.
"""

import cv2
import numpy as np

from skating_ml.visualization.config import LayerConfig
from skating_ml.visualization.layers.base import Layer, LayerContext

_CLASS_COLORS = {
    0: (255, 200, 0),  # face — yellow
    1: (0, 255, 0),  # person — green
}

_CLASS_LABELS = {
    0: "Face",
    1: "Person",
}


class FootTrackerLayer(Layer):
    """Renders foot tracker bounding boxes."""

    def __init__(self, config: LayerConfig | None = None) -> None:
        super().__init__(config=config or LayerConfig(enabled=True, z_index=1, opacity=0.8))
        self.name = "FootTracker"

    def render(self, frame: np.ndarray, context: LayerContext) -> np.ndarray:
        detections = context.custom_data.get("foot_detections")
        if detections is None:
            return frame

        for det in detections:
            cls_id = det.get("class_id", 1)
            if cls_id != 1:  # only draw person bboxes
                continue
            conf = det.get("confidence", 0.0)
            x1, y1, x2, y2 = det["bbox"]
            color = _CLASS_COLORS.get(cls_id, (255, 255, 255))
            label = f"{_CLASS_LABELS.get(cls_id, '?')} {conf:.0%}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - 20), (int(x1) + tw + 6, int(y1)), color, -1)
            cv2.putText(
                frame,
                label,
                (int(x1) + 3, int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame
