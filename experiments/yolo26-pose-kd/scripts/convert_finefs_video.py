#!/usr/bin/env python3
"""
Convert FineFS dataset (video + skeleton) to YOLO pose format for KD training.

FineFS format:
- video/{id}.mp4: Figure skating competition videos
- skeleton/{id}.npz: Shape (T, 17, 3), H3.6M 17kp format (3D world coords in meters)
- annotation/{id}.json: Element timing, scores, etc.
  - timing format: "MM-SS,MM-SS" (minutes-seconds, start-end)
- FPS: 25 (estimated from skeleton length / video duration)

Key differences from convert_finefs.py:
- Uses actual video frames instead of dummy black images
- All keypoints treated as visible (z is depth in meters, NOT a visibility flag)
- Proper timing parsing: MM-SS to frame index
- Video-level 80/20 train/val split

YOLO pose format:
- images/{id}_f{frame:06d}.jpg: Frame images
- labels/{id}_f{frame:06d}.txt: One line per person
  class_id x_center y_center width height kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
  All coordinates normalized to [0,1]

Usage:
    python3 convert_finefs_video.py [--skip-existing]
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# --- Paths (server) ---
FINEFS_ROOT = Path("/root/data/datasets/raw/FineFS/data")
VIDEO_DIR = FINEFS_ROOT / "video"
SKELETON_DIR = FINEFS_ROOT / "skeleton"
ANNOTATION_DIR = FINEFS_ROOT / "annotation"

# --- Output ---
OUTPUT_ROOT = Path("/root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/finefs")
TRAIN_DIR = OUTPUT_ROOT / "train"
VAL_DIR = OUTPUT_ROOT / "val"

# --- Parameters ---
CLASS_ID = 0
NUM_KEYPOINTS = 17
MIN_VISIBLE_KEYPOINTS = 5  # Keep frames with >= 5 visible keypoints
BBOX_PADDING = 0.2  # 20% padding around keypoints
FPS = 25  # FineFS skeleton FPS
SAMPLE_EVERY = 12  # Sample every 12 frames at 25fps = ~2fps
IMG_SIZE = (640, 640)  # Resize for YOLO training
RANDOM_SEED = 42

# H3.6M 17 keypoints (FineFS order)
H36M_KEYPOINTS = [
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
][:17]


def parse_timing_mmss(time_str: str, fps: int = FPS) -> list[tuple[int, int]]:
    """
    Parse FineFS timing "MM-SS,MM-SS" into frame ranges.

    Example: "0-33,0-36" -> [(825, 900)]  (0:33=825th frame, 0:36=900th frame at 25fps)

    Args:
        time_str: Timing string like "0-33,0-36"
        fps: Frames per second

    Returns:
        List of (start_frame, end_frame) tuples
    """
    if not time_str:
        return []

    ranges = []
    parts = time_str.split(",")
    if len(parts) != 2:
        return []

    try:
        start_min, start_sec = parts[0].strip().split("-")
        end_min, end_sec = parts[1].strip().split("-")

        start_frame = (int(start_min) * 60 + int(start_sec)) * fps
        end_frame = (int(end_min) * 60 + int(end_sec)) * fps
        ranges.append((start_frame, end_frame))
    except (ValueError, AttributeError):
        return []

    return ranges


def get_element_frames(
    annotation: dict | None,
    total_frames: int,
    fps: int = FPS,
    sample_every: int = SAMPLE_EVERY,
) -> list[int]:
    """
    Get frame indices to extract from annotation timing.

    If no annotation or no element timing, sample full video.
    Otherwise, sample within element timing ranges only.

    Args:
        annotation: Parsed JSON annotation dict
        total_frames: Total number of frames in the skeleton
        fps: Video FPS
        sample_every: Sample every N-th frame

    Returns:
        Sorted list of frame indices
    """
    if annotation:
        all_ranges = []
        for elem_key, elem_data in annotation.get("executed_element", {}).items():
            time_str = elem_data.get("time", "")
            ranges = parse_timing_mmss(time_str, fps)
            all_ranges.extend(ranges)

        if all_ranges:
            # Merge overlapping ranges and sample
            merged = merge_ranges(all_ranges)
            frames = []
            for start, end in merged:
                start = max(0, start)
                end = min(total_frames - 1, end)
                for f in range(start, end + 1, sample_every):
                    if f < total_frames:
                        frames.append(f)
            return sorted(set(frames))

    # Fallback: sample full video
    return list(range(0, total_frames, sample_every))


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent frame ranges."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def compute_global_normalization(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute global normalization range from all frames of a video.

    Uses the full video's keypoint range to normalize coordinates.
    This ensures bbox varies across frames (not always 1.0x1.0).

    Args:
        poses: (T, 17, 3) array [x_meters, y_meters, z_meters]

    Returns:
        (norm_min, norm_range) where norm_min = [x_min, y_min], norm_range = [x_range, y_range]
    """
    # Get visible keypoints (non-zero in all axes)
    visible_mask = np.any(poses != 0.0, axis=2)  # (T, 17)
    visible_kpts = poses[visible_mask][:, :2]  # (N, 2)

    if len(visible_kpts) == 0:
        return np.array([0.0, 0.0]), np.array([1.0, 1.0])

    x_min, y_min = visible_kpts.min(axis=0)
    x_max, y_max = visible_kpts.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Prevent division by zero
    if x_range < 1e-6:
        x_range = 1.0
    if y_range < 1e-6:
        y_range = 1.0

    # Add 15% padding for normalization range
    x_min -= 0.15 * x_range
    x_max += 0.15 * x_range
    y_min -= 0.15 * y_range
    y_max += 0.15 * y_range

    norm_min = np.array([x_min, y_min])
    norm_range = np.array([x_max - x_min, y_max - y_min])

    return norm_min, norm_range


def normalize_skeleton_to_yolo(
    pose_3d: np.ndarray,
    norm_min: np.ndarray = None,
    norm_range: np.ndarray = None,
) -> np.ndarray:
    """
    Convert FineFS 3D skeleton (meters) to YOLO normalized 2D keypoints.

    Uses global normalization range to ensure bbox varies across frames.
    Falls back to per-frame normalization if norm_min/norm_range not provided.

    Args:
        pose_3d: (17, 3) array [x_meters, y_meters, z_meters]
        norm_min: [x_min, y_min] global normalization offset
        norm_range: [x_range, y_range] global normalization scale

    Returns:
        (17, 3) array [x_norm, y_norm, visibility]
    """
    keypoints_2d = pose_3d[:, :2].copy()

    if norm_min is not None and norm_range is not None:
        # Global normalization
        keypoints_2d = (keypoints_2d - norm_min) / norm_range
    else:
        # Fallback: per-frame normalization
        x_min, x_max = keypoints_2d[:, 0].min(), keypoints_2d[:, 0].max()
        y_min, y_max = keypoints_2d[:, 1].min(), keypoints_2d[:, 1].max()
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)
        keypoints_2d[:, 0] = (keypoints_2d[:, 0] - x_min) / x_range
        keypoints_2d[:, 1] = (keypoints_2d[:, 1] - y_min) / y_range

    # Clip to [0, 1]
    keypoints_2d = np.clip(keypoints_2d, 0.0, 1.0)

    # All keypoints are visible (z is depth, not visibility)
    visibility = np.ones(17, dtype=np.float32)

    return np.column_stack([keypoints_2d, visibility])


def compute_bbox_from_keypoints(
    keypoints: np.ndarray, padding: float = BBOX_PADDING
) -> tuple[float, float, float, float]:
    """
    Compute YOLO-format bounding box from normalized keypoints.

    Args:
        keypoints: (17, 3) [x_norm, y_norm, visibility]
        padding: Padding fraction

    Returns:
        (x_center, y_center, width, height) in [0, 1]
    """
    visible = keypoints[:, 2] > 0
    if not visible.any():
        return 0.5, 0.5, 0.1, 0.1

    visible_kpts = keypoints[visible, :2]
    x_min, y_min = visible_kpts.min(axis=0)
    x_max, y_max = visible_kpts.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min

    # Add padding
    x_min -= width * padding
    x_max += width * padding
    y_min -= height * padding
    y_max += height * padding

    # Clip to [0, 1]
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(1.0, x_max)
    y_max = min(1.0, y_max)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def create_yolo_label(keypoints: np.ndarray) -> str:
    """
    Create YOLO pose label line.

    Format: class_id cx cy w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
    """
    cx, cy, w, h = compute_bbox_from_keypoints(keypoints)

    parts = [str(CLASS_ID), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for kp in keypoints:
        parts.extend([f"{kp[0]:.6f}", f"{kp[1]:.6f}", f"{int(kp[2])}"])

    return " ".join(parts)


def count_visible_keypoints(pose_3d: np.ndarray) -> int:
    """
    Count 'visible' keypoints in a frame.

    Since FineFS z is depth (not visibility), we use a different heuristic:
    a keypoint is considered 'missing' if x==0 AND y==0 AND z==0 (all zeros).
    """
    # A keypoint is invisible only if ALL coordinates are zero
    zero_mask = np.all(pose_3d == 0.0, axis=1)
    return int((~zero_mask).sum())


def convert_video(video_id: int, split: str, skip_existing: bool = False) -> dict:
    """
    Convert one FineFS video to YOLO pose format.

    Reads video frames sequentially (no seeking) for speed.
    Pre-filters skeleton frames before opening the video.

    Args:
        video_id: Video/skeleton ID
        split: 'train' or 'val'
        skip_existing: Skip if output image already exists

    Returns:
        Dict with conversion stats
    """
    result = {"id": video_id, "frames_extracted": 0, "frames_skipped": 0, "error": None}

    # Load skeleton
    skel_path = SKELETON_DIR / f"{video_id}.npz"
    if not skel_path.exists():
        result["error"] = "no_skeleton"
        return result

    try:
        data = np.load(skel_path)
        poses = data["reconstruction"]  # (T, 17, 3)
    except Exception as e:
        result["error"] = f"skeleton_load: {e}"
        return result

    total_frames = poses.shape[0]

    # Load annotation
    ann_path = ANNOTATION_DIR / f"{video_id}.json"
    annotation = None
    if ann_path.exists():
        try:
            with open(ann_path) as f:
                annotation = json.load(f)
        except Exception:
            pass

    # Get frame indices to extract
    frame_indices = get_element_frames(annotation, total_frames)

    if not frame_indices:
        result["error"] = "no_frames"
        return result

    # Pre-filter: keep only frames with enough visible keypoints
    frame_set = set()
    for fi in frame_indices:
        if fi >= total_frames:
            continue
        if count_visible_keypoints(poses[fi]) >= MIN_VISIBLE_KEYPOINTS:
            frame_set.add(fi)

    if not frame_set:
        result["error"] = "no_valid_frames"
        return result

    # Compute global normalization from all frames (for proper bbox variation)
    norm_min, norm_range = compute_global_normalization(poses)

    # Open video
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        result["error"] = "no_video"
        return result

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result["error"] = "video_open_failed"
        return result

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output directories
    output_dir = TRAIN_DIR if split == "train" else VAL_DIR
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    # Read video sequentially up to max needed frame (no seeking!)
    max_frame = max(frame_set)
    vid_frame_idx = 0

    while vid_frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if vid_frame_idx in frame_set:
            pose = poses[vid_frame_idx]

            # Resize to IMG_SIZE
            frame_resized = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

            # Normalize skeleton to YOLO format (global normalization for proper bbox)
            keypoints_norm = normalize_skeleton_to_yolo(pose, norm_min, norm_range)

            # Create YOLO label
            label_str = create_yolo_label(keypoints_norm)

            # Save
            stem = f"{video_id}_f{vid_frame_idx:06d}"
            img_path = images_dir / f"{stem}.jpg"
            lbl_path = labels_dir / f"{stem}.txt"

            if skip_existing and img_path.exists() and lbl_path.exists():
                converted += 1
            else:
                cv2.imwrite(str(img_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(lbl_path, "w") as f:
                    f.write(label_str + "\n")
                converted += 1

        vid_frame_idx += 1

    cap.release()

    result["frames_extracted"] = converted
    result["frames_skipped"] = len(frame_set) - converted
    result["video_fps"] = video_fps
    result["video_frame_count"] = video_frame_count
    result["skeleton_frames"] = total_frames

    return result


def main():
    parser = argparse.ArgumentParser(description="Convert FineFS to YOLO pose format")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already converted frames"
    )
    parser.add_argument("--dry-run", action="store_true", help="Count frames without converting")
    args = parser.parse_args()

    print("=" * 60)
    print("FineFS -> YOLO Pose Conversion (with video frames)")
    print("=" * 60)

    # Verify inputs
    print(f"\nVideo dir: {VIDEO_DIR}")
    print(f"  Videos: {len(list(VIDEO_DIR.glob('*.mp4')))}")
    print(f"Skeleton dir: {SKELETON_DIR}")
    print(f"  Skeletons: {len(list(SKELETON_DIR.glob('*.npz')))}")
    print(f"Annotation dir: {ANNOTATION_DIR}")
    print(f"  Annotations: {len(list(ANNOTATION_DIR.glob('*.json')))}")
    print(f"\nOutput: {OUTPUT_ROOT}")
    print(
        f"  FPS: {FPS}, Sample every: {SAMPLE_EVERY} frames (~{FPS / SAMPLE_EVERY:.1f} fps output)"
    )
    print(f"  Min visible keypoints: {MIN_VISIBLE_KEYPOINTS}")
    print(f"  Image size: {IMG_SIZE}")

    # Find videos that have both skeleton and video
    skeleton_ids = set(int(f.stem) for f in SKELETON_DIR.glob("*.npz"))
    video_ids = set(int(f.stem) for f in VIDEO_DIR.glob("*.mp4"))
    common_ids = sorted(skeleton_ids & video_ids)

    print(f"\nCommon IDs (skeleton + video): {len(common_ids)}")
    if len(skeleton_ids) - len(common_ids):
        print(f"  WARNING: {len(skeleton_ids) - len(common_ids)} skeletons without video")
    if len(video_ids) - len(common_ids):
        print(f"  WARNING: {len(video_ids) - len(common_ids)} videos without skeleton")

    if not common_ids:
        print("\nERROR: No common IDs found. Check that video.zip is extracted.")
        return

    # Video-level split (80/20)
    random.seed(RANDOM_SEED)
    random.shuffle(common_ids)
    split_idx = int(len(common_ids) * 0.8)
    train_ids = sorted(common_ids[:split_idx])
    val_ids = sorted(common_ids[split_idx:])

    print("\nSplit:")
    print(f"  Train: {len(train_ids)} videos")
    print(f"  Val: {len(val_ids)} videos")

    if args.dry_run:
        # Estimate frame counts without actually converting
        print("\n--- DRY RUN ---")
        total_train = 0
        total_val = 0
        for vid in tqdm(train_ids, desc="Estimating train"):
            d = np.load(SKELETON_DIR / f"{vid}.npz")
            poses = d["reconstruction"]
            ann_path = ANNOTATION_DIR / f"{vid}.json"
            ann = None
            if ann_path.exists():
                with open(ann_path) as f:
                    ann = json.load(f)
            frames = get_element_frames(ann, poses.shape[0])
            # Estimate visible frames
            for fi in frames:
                if fi < poses.shape[0]:
                    n_vis = count_visible_keypoints(poses[fi])
                    if n_vis >= MIN_VISIBLE_KEYPOINTS:
                        total_train += 1
        for vid in tqdm(val_ids, desc="Estimating val"):
            d = np.load(SKELETON_DIR / f"{vid}.npz")
            poses = d["reconstruction"]
            ann_path = ANNOTATION_DIR / f"{vid}.json"
            ann = None
            if ann_path.exists():
                with open(ann_path) as f:
                    ann = json.load(f)
            frames = get_element_frames(ann, poses.shape[0])
            for fi in frames:
                if fi < poses.shape[0]:
                    n_vis = count_visible_keypoints(poses[fi])
                    if n_vis >= MIN_VISIBLE_KEYPOINTS:
                        total_val += 1
        print("\nEstimated output:")
        print(f"  Train: {total_train} frames")
        print(f"  Val: {total_val} frames")
        print(f"  Total: {total_train + total_val} frames")
        return

    # --- Actual conversion ---
    print("\n--- CONVERTING ---\n")

    train_total = 0
    train_errors = 0
    val_total = 0
    val_errors = 0

    print("Converting TRAIN set...")
    for vid in tqdm(train_ids, desc="Train"):
        result = convert_video(vid, "train", skip_existing=args.skip_existing)
        train_total += result["frames_extracted"]
        if result["error"]:
            train_errors += 1

    print("\nConverting VAL set...")
    for vid in tqdm(val_ids, desc="Val"):
        result = convert_video(vid, "val", skip_existing=args.skip_existing)
        val_total += result["frames_extracted"]
        if result["error"]:
            val_errors += 1

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Train: {train_total} frames from {len(train_ids)} videos ({train_errors} errors)")
    print(f"Val: {val_total} frames from {len(val_ids)} videos ({val_errors} errors)")
    print(f"Total: {train_total + val_total} frames")
    print("\nOutput directories:")
    print(f"  Train images: {TRAIN_DIR / 'images'}")
    print(f"  Train labels: {TRAIN_DIR / 'labels'}")
    print(f"  Val images: {VAL_DIR / 'images'}")
    print(f"  Val labels: {VAL_DIR / 'labels'}")

    # Disk usage
    for d in [TRAIN_DIR, VAL_DIR]:
        if d.exists():
            img_count = len(list((d / "images").glob("*.jpg")))
            lbl_count = len(list((d / "labels").glob("*.txt")))
            print(f"  {d.name}: {img_count} images, {lbl_count} labels")


if __name__ == "__main__":
    main()
