"""SkatingVerse dataset converter.

Converts SkatingVerse mp4 videos to unified numpy format using rtmlib Wholebody
pose extraction. Saves as numpy npy (NOT zarr) because sequences have variable length.

Data format:
    train.txt: each line = "filename_without_ext label_id"
    answer.txt: each line = "filename_without_ext" (test split, no labels)
    mapping.txt: each line = "class_name label_id"
    train_videos/ / test_videos/: mp4 files

Skipped classes:
    12 = "No Basic" (not a skating element)
    27 = "Sequence" (very long ~25s compilation videos)
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from data_tools.validate import ValidationError, validate_skeleton

# Classes to skip during conversion
SKIP_CLASSES = {12, 27}  # NoBasic, Sequence

# Minimum valid frames per sample (videos are short clips)
MIN_FRAMES = 10

# Confidence threshold: skip frames with mean body confidence below this
MIN_CONFIDENCE = 0.3


def normalize(p: np.ndarray) -> np.ndarray:
    """Root-center + spine-length normalize.

    Args:
        p: Pose array of shape (T, V, C) float32, where T=time, V=17 keypoints, C=channels

    Returns:
        Normalized poses of same shape
    """
    # Root-center: subtract mid-hip (keypoints 11-12 in H3.6M format)
    mid_hip = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid_hip

    # Spine-length normalization: scale by distance from mid-hip to shoulders (keypoints 5-6)
    shoulders = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(shoulders - mid_hip, axis=1, keepdims=True)

    # Avoid division by zero
    spine = np.maximum(spine, 0.01)

    return p / spine


def _load_split_index(index_path: Path) -> list[tuple[str, int | None]]:
    """Parse a train.txt or answer.txt file.

    train.txt: "filename_without_extension label_id"
    answer.txt: "filename_without_extension" (no label, for test split)

    Returns:
        List of (filename_without_ext, label_id_or_None) tuples
    """
    entries = []
    with index_path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                name, label = parts[0], int(parts[1])
                entries.append((name, label))
            elif len(parts) == 1:
                entries.append((parts[0], None))
    return entries


def _load_mapping(mapping_path: Path) -> dict[int, str]:
    """Parse mapping.txt into {label_id: class_name}.

    Each line: "class_name label_id"

    Returns:
        Dict mapping integer label ID to string class name
    """
    mapping = {}
    with mapping_path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                continue
            name, label = parts[0], int(parts[1])
            mapping[label] = name
    return mapping


def extract_single_video(
    video_path: Path,
    wb: Any,
    frame_skip: int = 4,
) -> np.ndarray | None:
    """Extract H3.6M 17-keypoint skeleton from a single video.

    Uses rtmlib Wholebody to detect poses frame-by-frame, then extracts
    the first 17 COCO body keypoints (which correspond to H3.6M).

    Args:
        video_path: Path to mp4 file
        wb: rtmlib Wholebody model instance (created ONCE, reused)
        frame_skip: Extract every Nth frame (default 4)

    Returns:
        (T, 17, 2) float32 array of pixel coordinates, or None if extraction fails
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < MIN_FRAMES:
        cap.release()
        return None

    frames_body = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # rtmlib Wholebody returns (keypoints, scores) tuple
        result = wb(frame)
        if result is None:
            frame_idx += 1
            continue

        keypoints, scores = result

        # keypoints: (N_persons, 133, 2), scores: (N_persons, 133)
        if keypoints is None or len(keypoints) == 0:
            frame_idx += 1
            continue

        # Take the first (largest/best) person
        person_kps = keypoints[0]  # (133, 2)
        person_scores = scores[0]  # (133,)

        # First 17 keypoints = COCO body = H3.6M compatible
        body_kps = person_kps[:17, :].copy()  # (17, 2)
        body_conf = person_scores[:17]  # (17,)

        # Skip frames with low mean confidence
        mean_conf = float(np.mean(body_conf))
        if mean_conf < MIN_CONFIDENCE:
            frame_idx += 1
            continue

        frames_body.append(body_kps)
        frame_idx += 1

    cap.release()

    if len(frames_body) < MIN_FRAMES:
        return None

    return np.stack(frames_body, axis=0).astype(np.float32)  # (T, 17, 2)


def convert_skatingverse(
    raw_dir: Path | str,
    output_dir: Path | str,
    max_per_class: int = 0,
    frame_skip: int = 4,
) -> dict[str, Any]:
    """Convert SkatingVerse mp4 videos to unified numpy format.

    Processes train.txt and answer.txt, extracts skeletons using rtmlib,
    normalizes, validates, and saves as numpy npy files.

    The conversion is resumable: if output files already exist for a split,
    that split is skipped.

    Args:
        raw_dir: Path containing train.txt, answer.txt, mapping.txt,
                 train_videos/, test_videos/
        output_dir: Where to save unified dataset
        max_per_class: Max samples per class (0 = no limit)
        frame_skip: Extract every Nth frame (default 4)

    Returns:
        Statistics dict with counts, classes, and creation time
    """
    from rtmlib import Wholebody

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class mapping
    mapping_path = raw_dir / "mapping.txt"
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping.txt not found in {raw_dir}")

    mapping = _load_mapping(mapping_path)
    print(f"Loaded {len(mapping)} class mappings", flush=True)
    print(
        f"Skipping classes: {SKIP_CLASSES} ({[mapping[k] for k in SKIP_CLASSES if k in mapping]})",
        flush=True,
    )

    # Create Wholebody model ONCE for all videos
    print("Loading rtmlib Wholebody model...", flush=True)
    wb = Wholebody(mode="balanced", backend="onnxruntime")
    print("Model loaded.", flush=True)

    stats: dict[str, Any] = {}
    errors: list[tuple[str, int, str, list[ValidationError]]] = []

    splits = [
        ("train", "train.txt", "train_videos"),
        ("test", "answer.txt", "test_videos"),
    ]

    for split, index_file, videos_dir in splits:
        index_path = raw_dir / index_file
        videos_path = raw_dir / videos_dir

        if not index_path.exists():
            print(f"Warning: {index_file} not found, skipping {split} split...", flush=True)
            continue
        if not videos_path.exists():
            print(f"Warning: {videos_dir}/ not found, skipping {split} split...", flush=True)
            continue

        # Resumable: skip if output already exists
        poses_path = output_dir / f"{split}_poses.npy"
        labels_path = output_dir / f"{split}_labels.npy"
        if poses_path.exists() and labels_path.exists():
            poses_obj = np.load(poses_path, allow_pickle=True)
            stats[split] = len(poses_obj)
            print(
                f"Skipping {split}: {poses_path} already exists ({stats[split]} samples)",
                flush=True,
            )
            continue

        # Load index
        entries = _load_split_index(index_path)
        print(f"\nProcessing {split} split: {len(entries)} entries in {index_file}", flush=True)

        # Filter out skipped classes (only for labeled entries)
        filtered = [
            (name, label) for name, label in entries if label is None or label not in SKIP_CLASSES
        ]
        print(f"After filtering skipped classes: {len(filtered)} entries", flush=True)

        # Apply max_per_class limit (only for labeled entries)
        if max_per_class > 0:
            class_counts: dict[int, int] = {}
            limited: list[tuple[str, int | None]] = []
            for name, label in filtered:
                if label is None:
                    # Unlabeled (test split): apply a flat cap
                    if len([x for x in limited if x[1] is None]) < max_per_class * 5:
                        limited.append((name, label))
                else:
                    count = class_counts.get(label, 0)
                    if count < max_per_class:
                        limited.append((name, label))
                        class_counts[label] = count + 1
            filtered = limited
            print(f"After max_per_class={max_per_class}: {len(filtered)} entries", flush=True)

        # Process videos
        poses_list: list[np.ndarray] = []
        labels_arr: list[int] = []
        meta_samples: list[dict[str, Any]] = []
        skipped = 0
        failed = 0

        for idx, (name, label) in enumerate(
            tqdm(filtered, desc=f"Extracting {split}", unit="video")
        ):
            video_file = videos_path / f"{name}.mp4"
            if not video_file.exists():
                failed += 1
                continue

            # Extract skeleton
            pose_seq = extract_single_video(video_file, wb=wb, frame_skip=frame_skip)

            if pose_seq is None:
                skipped += 1
                continue

            # Normalize (pixel coords -> normalized coords)
            pose_seq = normalize(pose_seq)

            # Validate after normalization
            val_errors = validate_skeleton(
                pose_seq,
                sample_id=f"{split}_{name}",
                max_coord=100.0,
            )
            if val_errors:
                errors.append((split, idx, name, val_errors))
                skipped += 1
                continue

            poses_list.append(pose_seq)
            labels_arr.append(label if label is not None else -1)
            meta_samples.append(
                {
                    "video": f"{name}.mp4",
                    "original_label": label,
                    "class_name": mapping.get(label, "unlabeled")
                    if label is not None
                    else "unlabeled",
                    "frames": pose_seq.shape[0],
                }
            )

        # Save as numpy object array (variable-length sequences)
        if poses_list:
            poses_obj = np.array(poses_list, dtype=object)
            labels_np = np.array(labels_arr, dtype=np.int32)
            np.save(poses_path, poses_obj)
            np.save(labels_path, labels_np)

        stats[split] = len(poses_list)
        stats[f"{split}_skipped"] = skipped
        stats[f"{split}_failed"] = failed

        print(f"\n{split}: {len(poses_list)} valid, {skipped} skipped, {failed} failed", flush=True)
        print(f"Saved to {poses_path}", flush=True)

        # Save per-split metadata
        meta_path = output_dir / f"{split}_meta.json"
        with meta_path.open("w") as f:
            json.dump(meta_samples, f, indent=2)

    # Print validation errors summary
    if errors:
        print(f"\nValidation errors in {len(errors)} samples:", flush=True)
        for split, _idx, name, err_list in errors[:5]:
            print(f"  {split}/{name}: {len(err_list)} errors", flush=True)
            for err in err_list[:2]:
                print(f"    - {err.field}: {err.message}", flush=True)
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more", flush=True)

    # Save global metadata
    stats["classes"] = len(mapping)
    stats["skip_classes"] = list(SKIP_CLASSES)
    stats["frame_skip"] = frame_skip
    stats["created"] = str(np.datetime64("now"))

    meta_path = output_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nMetadata saved to {meta_path}", flush=True)
    print(f"Total: {stats.get('train', 0)} train, {stats.get('test', 0)} test", flush=True)

    return stats


def load_unified(
    output_dir: Path | str,
    split: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load unified SkatingVerse dataset.

    Args:
        output_dir: Path containing unified dataset files
        split: Either 'train' or 'test'

    Returns:
        Tuple of (poses list, labels array)
            - poses: list of (T, 17, 2) float32 arrays (variable T)
            - labels: (N,) int32 array of class labels
    """
    output_dir = Path(output_dir)

    poses_path = output_dir / f"{split}_poses.npy"
    labels_path = output_dir / f"{split}_labels.npy"

    if not poses_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Unified {split} split not found in {output_dir}")

    poses_obj = np.load(poses_path, allow_pickle=True)
    labels = np.load(labels_path)

    # Convert object array back to list
    poses_list = list(poses_obj)

    return poses_list, labels
