"""Fast semi-transparent rectangle overlay — ROI-scoped, no full-frame copy."""

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]
Rect = tuple[int, int, int, int]  # (x, y, w, h)


def draw_overlay_rect(
    frame: Frame,
    rect: Rect,
    color: tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.6,
    border_color: tuple[int, int, int] | None = None,
    border_thickness: int = 0,
) -> Frame:
    """Draw a semi-transparent filled rectangle using ROI-scoped blending.

    Unlike cv2.addWeighted on the full frame, this only copies the affected
    region, applies the blend, and writes it back.

    Args:
        frame: OpenCV image (H, W, 3) BGR — modified in place.
        rect: (x, y, width, height) — clipped to frame bounds.
        color: Fill color (BGR).
        alpha: Fill opacity [0, 1]. 0 = invisible, 1 = opaque.
        border_color: Border color (BGR). None = no border.
        border_thickness: Border thickness in pixels. 0 = no border.

    Returns:
        frame (same object, modified in place).
    """
    if alpha <= 0.0:
        return frame

    rx, ry, rw, rh = rect
    fh, fw = frame.shape[:2]

    # Clip to frame bounds
    x1 = max(rx, 0)
    y1 = max(ry, 0)
    x2 = min(rx + rw, fw)
    y2 = min(ry + rh, fh)
    if x1 >= x2 or y1 >= y2:
        return frame

    roi = frame[y1:y2, x1:x2]
    if alpha >= 1.0:
        roi[:, :] = color
    else:
        overlay = roi.copy()
        overlay[:, :] = color
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)

    # Border (drawn on top of blended region)
    if border_color is not None and border_thickness > 0:
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            border_color,
            border_thickness,
            cv2.LINE_AA,
        )

    return frame
