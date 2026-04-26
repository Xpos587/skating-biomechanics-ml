#!/usr/bin/env python3
"""
Convert FineFS dataset to YOLO pose format.

FineFS format:
- skeleton/{id}.npz: Shape (T, 17, 3), H3.6M 17kp format
- Coordinates: 3D world coords (meters), NOT normalized
- Z=0 indicates missing keypoints
- Element timing: MM-SS,MM-SS format in annotation JSONs

YOLO pose format:
- images/*.jpg: Frame images
- labels/*.txt: One line per person
  class_id x_center y_center width height kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
  All coordinates normalized to [0,1]
  visibility: 0=occluded, 1=visible
"""

import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# FineFS dataset paths
FINEFS_ROOT = Path("/home/michael/Github/skating-biomechanics-ml/data/datasets/finefs")
SKELETON_DIR = FINEFS_ROOT / "skeleton"
ANNOTATION_DIR = FINEFS_ROOT / "annotation"

# Output paths
OUTPUT_ROOT = Path(
    "/home/michael/Github/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/finefs"
)
TRAIN_DIR = OUTPUT_ROOT / "train"
VAL_DIR = OUTPUT_ROOT / "val"

# YOLO format parameters
CLASS_ID = 0  # person class
NUM_KEYPOINTS = 17  # H3.6M format
MIN_VISIBLE_KEYPOINTS = 5  # Minimum visible keypoints to keep a frame
BBOX_PADDING = 0.2  # 20% padding around keypoints

# H3.6M keypoint names (for reference)
H36M_KEYPOINT_NAMES = [
    "hip_center",
    "left_hip",
    "right_hip",
    "spine",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "pelvis",
    "thorax",
    "neck",
    "head_top",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
][:17]  # Take first 17 for H3.6M


def load_annotation(skeleton_id: int) -> dict:
    """Load annotation JSON for a skeleton."""
    ann_path = ANNOTATION_DIR / f"{skeleton_id}.json"
    if not ann_path.exists():
        return None
    with open(ann_path) as f:
        return json.load(f)


def parse_timing(time_str: str) -> list[tuple[int, int]]:
    """
    Parse timing string "start-end,start-end" into frame ranges.

    Args:
        time_str: Timing string like "0-33,0-36"

    Returns:
        List of (start_frame, end_frame) tuples
    """
    if not time_str:
        return []

    ranges = []
    for part in time_str.split(","):
        try:
            start, end = part.strip().split("-")
            ranges.append((int(start), int(end)))
        except (ValueError, AttributeError):
            continue

    return ranges


def normalize_keypoints(pose_3d: np.ndarray) -> np.ndarray:
    """
    Convert 3D skeleton to normalized 2D keypoints.

    FineFS coordinates are in meters, roughly in range [-1, 1].
    We normalize to [0, 1] by: (coord - min) / (max - min)

    Args:
        pose_3d: (17, 3) array with x, y, z coordinates in meters

    Returns:
        (17, 3) array with x, y normalized to [0,1] and visibility flag
    """
    # Take x, y coordinates (discard z)
    keypoints_2d = pose_3d[:, :2].copy()

    # Get visible keypoints for normalization range
    visible_mask = pose_3d[:, 2] != 0

    if visible_mask.any():
        # Normalize using visible keypoints only
        visible_kpts = keypoints_2d[visible_mask]
        x_min, x_max = visible_kpts[:, 0].min(), visible_kpts[:, 0].max()
        y_min, y_max = visible_kpts[:, 1].min(), visible_kpts[:, 1].max()

        # Add 10% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        # Normalize to [0, 1]
        keypoints_2d[:, 0] = (keypoints_2d[:, 0] - x_min) / (x_max - x_min)
        keypoints_2d[:, 1] = (keypoints_2d[:, 1] - y_min) / (y_max - y_min)
    else:
        # No visible keypoints, center at 0.5
        keypoints_2d[:, :] = 0.5

    # Clip to [0, 1]
    keypoints_2d = np.clip(keypoints_2d, 0.0, 1.0)

    # Create visibility flag (0 if z=0, else 1)
    visibility = (pose_3d[:, 2] != 0).astype(np.float32)

    # Stack: x, y, visibility
    keypoints_norm = np.column_stack([keypoints_2d, visibility])

    return keypoints_norm


def compute_bbox(
    keypoints: np.ndarray, padding: float = BBOX_PADDING
) -> tuple[float, float, float, float]:
    """
    Compute bounding box from keypoints with padding.

    Args:
        keypoints: (17, 3) array with x, y, visibility (normalized)
        padding: Padding fraction around bbox

    Returns:
        (x_center, y_center, width, height) normalized to [0,1]
    """
    # Get visible keypoints only
    visible = keypoints[:, 2] > 0
    if not visible.any():
        # Default bbox if no visible keypoints
        return 0.5, 0.5, 0.1, 0.1

    visible_kpts = keypoints[visible, :2]

    # Get min/max with padding
    x_min, y_min = visible_kpts.min(axis=0)
    x_max, y_max = visible_kpts.max(axis=0)

    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    x_min -= width * padding
    x_max += width * padding
    y_min -= height * padding
    y_max += height * padding

    # Clip to [0, 1]
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(1.0, x_max)
    y_max = min(1.0, y_max)

    # Convert to center format
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def create_yolo_label(keypoints: np.ndarray) -> str:
    """
    Create YOLO pose label string.

    Args:
        keypoints: (17, 3) array with x, y, visibility (normalized)

    Returns:
        YOLO format label string
    """
    # Compute bounding box
    x_center, y_center, width, height = compute_bbox(keypoints)

    # Format: class_id x_center y_center width height kp1_x kp1_y kp1_v ...
    parts = [str(CLASS_ID), f"{x_center:.6f}", f"{y_center:.6f}", f"{width:.6f}", f"{height:.6f}"]

    # Add keypoints
    for kp in keypoints:
        parts.extend(
            [
                f"{kp[0]:.6f}",  # x
                f"{kp[1]:.6f}",  # y
                f"{int(kp[2])}",  # visibility (0 or 1)
            ]
        )

    return " ".join(parts)


def filter_low_quality_frames(
    pose_sequence: np.ndarray, min_visible: int = MIN_VISIBLE_KEYPOINTS
) -> np.ndarray:
    """
    Filter frames with too few visible keypoints.

    Args:
        pose_sequence: (T, 17, 3) array
        min_visible: Minimum number of visible keypoints

    Returns:
        Boolean mask of valid frames
    """
    # Count visible keypoints per frame (z != 0)
    visible_counts = (pose_sequence[:, :, 2] != 0).sum(axis=1)

    # Keep frames with enough visible keypoints
    valid_frames = visible_counts >= min_visible

    return valid_frames


def extract_element_frames(
    pose_sequence: np.ndarray, time_ranges: list[tuple[int, int]]
) -> list[int]:
    """
    Extract frame indices for specific elements from timing annotations.

    Args:
        pose_sequence: (T, 17, 3) array
        time_ranges: List of (start, end) frame tuples

    Returns:
        List of frame indices to extract
    """
    if not time_ranges:
        # If no timing info, sample frames at 2 fps (assuming 25 fps original)
        # Sample every 12th frame
        return list(range(0, len(pose_sequence), 12))

    frames = []
    for start, end in time_ranges:
        # Validate bounds
        start = max(0, min(start, len(pose_sequence) - 1))
        end = max(0, min(end, len(pose_sequence)))

        # Sample frames in this range at 2 fps (every 12th frame at 25 fps)
        range_frames = list(range(start, end + 1, 12))
        frames.extend(range_frames)

    return sorted(set(frames))  # Remove duplicates


def convert_skeleton(skeleton_id: int, split: str) -> int:
    """
    Convert one skeleton file to YOLO format.

    Args:
        skeleton_id: Skeleton ID
        split: 'train' or 'val'

    Returns:
        Number of frames converted
    """
    # Load skeleton data
    skel_path = SKELETON_DIR / f"{skeleton_id}.npz"
    if not skel_path.exists():
        return 0

    try:
        data = np.load(skel_path)
        poses = data["reconstruction"]  # (T, 17, 3)
    except Exception as e:
        warnings.warn(f"Failed to load skeleton {skeleton_id}: {e}")
        return 0

    # Load annotation for timing info
    annotation = load_annotation(skeleton_id)

    # Extract element timing
    all_time_ranges = []
    if annotation:
        for elem_key, elem_data in annotation.get("executed_element", {}).items():
            time_str = elem_data.get("time", "")
            ranges = parse_timing(time_str)
            all_time_ranges.extend(ranges)

    # Extract frames to convert
    frame_indices = extract_element_frames(poses, all_time_ranges)

    # Filter low-quality frames
    valid_frames_mask = filter_low_quality_frames(poses)
    frame_indices = [f for f in frame_indices if valid_frames_mask[f]]

    if not frame_indices:
        return 0

    # Setup output directories
    output_dir = TRAIN_DIR if split == "train" else VAL_DIR
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Convert each frame
    converted_count = 0
    for frame_idx in frame_indices:
        pose = poses[frame_idx]  # (17, 3)

        # Normalize keypoints (using actual coordinate range)
        keypoints_norm = normalize_keypoints(pose)

        # Create YOLO label
        label_str = create_yolo_label(keypoints_norm)

        # Save label
        label_filename = f"{skeleton_id}_frame{frame_idx:06d}.txt"
        label_path = labels_dir / label_filename
        with open(label_path, "w") as f:
            f.write(label_str + "\n")

        # Create dummy black image (since we don't have videos)
        # In real scenario, you'd extract actual frames from videos
        image_filename = f"{skeleton_id}_frame{frame_idx:06d}.jpg"
        image_path = images_dir / image_filename

        # Create a black image as placeholder (640x640 for YOLO)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), dummy_image)

        converted_count += 1

    return converted_count


def main():
    """Main conversion function."""
    print("Converting FineFS dataset to YOLO pose format...")

    # Get all skeleton files
    skeleton_files = sorted(SKELETON_DIR.glob("*.npz"))
    skeleton_ids = [int(f.stem) for f in skeleton_files]

    print(f"Found {len(skeleton_ids)} skeleton files")

    # Split 80/20 at video level
    split_idx = int(len(skeleton_ids) * 0.8)
    train_ids = skeleton_ids[:split_idx]
    val_ids = skeleton_ids[split_idx:]

    print(f"Train: {len(train_ids)} videos")
    print(f"Val: {len(val_ids)} videos")

    # Convert training set
    print("\nConverting training set...")
    train_frames = 0
    for skel_id in tqdm(train_ids, desc="Train"):
        frames = convert_skeleton(skel_id, "train")
        train_frames += frames

    # Convert validation set
    print("\nConverting validation set...")
    val_frames = 0
    for skel_id in tqdm(val_ids, desc="Val"):
        frames = convert_skeleton(skel_id, "val")
        val_frames += frames

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Train frames: {train_frames}")
    print(f"Val frames: {val_frames}")
    print(f"Total frames: {train_frames + val_frames}")
    print("\nOutput directories:")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Val: {VAL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
