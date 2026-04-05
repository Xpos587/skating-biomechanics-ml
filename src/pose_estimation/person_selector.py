"""Interactive person selection using matplotlib (Sports2D-style).

Provides:
- compute_bboxes_from_poses(): Calculate bounding boxes from pose arrays
- point_in_bbox(): Hit-test for click detection
- select_persons_interactive(): Matplotlib GUI for person selection
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2)


def compute_bboxes_from_poses(poses: NDArray[np.float32]) -> list[BBox]:
    """Compute bounding boxes from pose arrays.

    Args:
        poses: (N_persons, 17, 2) pose array.

    Returns:
        List of (x1, y1, x2, y2) bounding boxes.
    """
    bboxes = []
    for i in range(len(poses)):
        row = poses[i]
        mask = ~np.isnan(row[:, 0]) & ~np.isnan(row[:, 1])
        if not mask.any():
            bboxes.append((0, 0, 0, 0))
            continue
        pts = row[mask]
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def point_in_bbox(x: int, y: int, bbox: BBox) -> bool:
    """Check if point (x, y) is inside bounding box."""
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def select_persons_interactive(
    video_path: str | Path,
    poses: NDArray[np.float32],
    bboxes: list[BBox] | None = None,
) -> list[int]:
    """Interactive person selection using matplotlib.

    Shows first frame with numbered bounding boxes.
    User clicks on persons to select them.
    Press Enter or close window to confirm.

    Args:
        video_path: Path to video file.
        poses: (N_persons, 17, 2) pose array for first frame.
        bboxes: Optional pre-computed bounding boxes.

    Returns:
        List of selected person indices.
    """
    try:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not available, falling back to CLI selection.")
        return _cli_fallback(poses, bboxes)

    import cv2

    # Read first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Cannot read video: {video_path}")
        return list(range(len(poses)))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if bboxes is None:
        bboxes = compute_bboxes_from_poses(poses)

    selected: list[int] = []
    n_persons = len(bboxes)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Select persons to track")

    ax.imshow(frame_rgb)
    ax.set_title("Click on persons to select, then press Enter", fontsize=12)
    ax.axis("off")

    rectangles = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.5,
            edgecolor="white",
            facecolor="none",
        )
        ax.add_patch(rect)
        rectangles.append(rect)
        ax.text(
            x1,
            y1 - 5,
            str(i),
            fontsize=10,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.6},
        )

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        for i, bbox in enumerate(bboxes):
            if point_in_bbox(x, y, bbox):
                if i in selected:
                    selected.remove(i)
                    rectangles[i].set_edgecolor("white")
                    rectangles[i].set_linewidth(1.5)
                else:
                    selected.append(i)
                    rectangles[i].set_edgecolor("darkorange")
                    rectangles[i].set_linewidth(3)
                fig.canvas.draw_idle()
                break

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not selected:
        selected = list(range(n_persons))

    print(f"Selected persons: {selected}")
    return selected


def _cli_fallback(
    poses: NDArray[np.float32],
    bboxes: list[BBox] | None = None,
) -> list[int]:
    """CLI fallback when matplotlib is not available."""
    n = len(poses)
    if n == 1:
        print("Auto-selecting person 0 (only one detected).")
        return [0]

    print(f"Detected {n} persons:")
    for i in range(n):
        if bboxes and i < len(bboxes):
            x1, y1, x2, y2 = bboxes[i]
            print(f"  [{i}] bbox=({x1},{y1},{x2},{y2})")
        else:
            print(f"  [{i}]")

    try:
        choice = input("Enter person index: ").strip()
        return [int(choice)]
    except (ValueError, EOFError):
        return [0]
