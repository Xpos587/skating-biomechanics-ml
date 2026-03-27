#!/usr/bin/env python3
"""Create skeleton overlay visualizations for dataset videos."""

import cv2
import numpy as np
from pathlib import Path

from skating_biomechanics_ml.pose_2d import PoseExtractor
from skating_biomechanics_ml.utils.video import extract_frames, get_video_meta

# COCO 17 keypoints skeleton edges (pairs of indices)
SKELETON_EDGES = [
    (0, 1), (0, 2),  # head
    (1, 3), (2, 4),  # ears to shoulders
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Colors for visualization
COLORS = {
    "person": (0, 255, 0),  # green
    "skeleton": (255, 0, 0),  # blue
    "keypoints": (0, 0, 255),  # red
}


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Draw skeleton on frame.

    Args:
        frame: Input frame (BGR).
        keypoints: Keypoints (17, 3) with x, y, confidence.
        threshold: Minimum confidence to display keypoint.

    Returns:
        Frame with skeleton overlay.
    """
    result = frame.copy()

    if keypoints is None or len(keypoints) < 17:
        return result

    h, w = frame.shape[:2]

    # Convert normalized coordinates back to pixel space
    # COCO keypoints are in pixel coordinates already
    for i, (x, y, conf) in enumerate(keypoints):
        if conf < threshold:
            continue

        cx, cy = int(x), int(y)
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        # Draw keypoint
        cv2.circle(result, (cx, cy), 5, COLORS["keypoints"], -1)

        # Draw index
        cv2.putText(result, str(i), (cx + 8, cy - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Draw skeleton edges
    for idx1, idx2 in SKELETON_EDGES:
        if idx1 >= len(keypoints) or idx2 >= len(keypoints):
            continue

        x1, y1, c1 = keypoints[idx1]
        x2, y2, c2 = keypoints[idx2]

        if c1 < threshold or c2 < threshold:
            continue

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))

        if not (0 <= pt1[0] < w and 0 <= pt1[1] < h):
            continue
        if not (0 <= pt2[0] < w and 0 <= pt2[1] < h):
            continue

        cv2.line(result, pt1, pt2, COLORS["skeleton"], 2)

    return result


def create_skeleton_video(
    video_path: Path,
    output_path: Path,
    max_frames: int = 300,
    display_frames: int = 5,
):
    """Create video with skeleton overlay.

    Args:
        video_path: Input video path.
        output_path: Output video path.
        max_frames: Maximum number of frames to process.
        display_frames: Number of frames to skip between displays.
    """
    print(f"Processing {video_path.name}...")

    extractor = PoseExtractor(model_size="s")
    meta = get_video_meta(video_path)

    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    processed = 0

    for frame in extract_frames(video_path, max_frames=max_frames):
        if frame_count % display_frames == 0:
            # Extract pose
            keypoints = extractor.extract_frame(frame)

            if keypoints is not None:
                # Draw skeleton
                annotated = draw_skeleton(frame, keypoints)
                out.write(annotated)
                processed += 1
            else:
                # No person detected, write original
                out.write(frame)
                processed += 1

        frame_count += 1

        if frame_count >= max_frames:
            break

    cap.release()
    out.release()

    print(f"  Processed {processed} frames")
    print(f"  Output: {output_path}")


def save_sample_frames(video_path: Path, output_dir: Path, num_samples: int = 5):
    """Save sample frames with skeleton overlay as images.

    Args:
        video_path: Input video path.
        output_dir: Output directory for images.
        num_samples: Number of sample frames to save.
    """
    print(f"Saving sample frames from {video_path.name}...")

    extractor = PoseExtractor(model_size="s")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = get_video_meta(video_path).num_frames
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    for i, target_frame in enumerate(sample_indices):
        # Extract specific frame
        for j, frame in enumerate(extract_frames(video_path)):
            if j == target_frame:
                keypoints = extractor.extract_frame(frame)

                if keypoints is not None:
                    annotated = draw_skeleton(frame, keypoints)

                    # Add frame info
                    h, w = annotated.shape[:2]
                    cv2.putText(annotated, f"Frame {j}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    output_path = output_dir / f"{video_path.stem}_frame{i+1}.jpg"
                    cv2.imwrite(str(output_path), annotated)
                    print(f"  Saved: {output_path.name}")
                break


def main():
    """Create visualizations for all dataset videos."""
    dataset_dir = Path("data/dataset")
    videos_dir = dataset_dir / "videos"
    output_dir = dataset_dir / "visualizations"

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(videos_dir.glob("*.mp4"))

    print(f"Found {len(video_files)} videos")
    print()

    # Create sample frames for each video
    for video_path in video_files:
        video_output = output_dir / "frames" / video_path.stem
        save_sample_frames(video_path, video_output, num_samples=5)

    print(f"\n=== Visualizations complete ===")
    print(f"Location: {output_dir}")


if __name__ == "__main__":
    main()
