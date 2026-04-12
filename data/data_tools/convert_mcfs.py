"""MCFS (Multi-camera Figure Skating) dataset converter.

Converts MCFS segments.pkl to unified numpy format.

MCFS contains 2668 pre-segmented skeleton sequences extracted from 271 videos.
Each segment is a (T, 17, 2) float32 array with a string label.

Note: MCFS provides 5-fold cross-validation splits (split0-split4) at the video
level (271 videos), but segments.pkl contains 2668 sub-segments without a direct
mapping to video files. Therefore all segments are saved as a single 'all' split.
The label mapping from mapping.txt (130 classes including NONE) is preserved in meta.json.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from data_tools.validate import ValidationError, validate_skeleton


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


def _load_label_mapping(raw_dir: Path) -> dict[str, int]:
    """Load label mapping from mapping.txt.

    Args:
        raw_dir: Path containing mapping.txt

    Returns:
        Dict mapping label name to integer ID
    """
    mapping_path = raw_dir / "mapping.txt"
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping.txt not found in {raw_dir}")

    mapping: dict[str, int] = {}
    with mapping_path.open("r") as f:
        for line in f:
            idx_str, name = line.strip().split(" ", 1)
            mapping[name] = int(idx_str)

    return mapping


def convert_mcfs(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Convert MCFS segments.pkl to unified numpy format.

    Args:
        raw_dir: Path containing segments.pkl and mapping.txt
        output_dir: Where to save unified dataset (will create if needed)

    Returns:
        Statistics dict with sample count, classes, label mapping, and creation time
    """
    import pickle

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load label mapping
    label_mapping = _load_label_mapping(raw_dir)
    print(f"Loaded {len(label_mapping)} label mappings from mapping.txt")

    # Load segments
    segments_path = raw_dir / "segments.pkl"
    if not segments_path.exists():
        raise FileNotFoundError(f"segments.pkl not found in {raw_dir}")

    with segments_path.open("rb") as f:
        segments = pickle.load(f)

    print(f"Loaded {len(segments)} segments from segments.pkl")

    # Process each segment
    poses_list = []
    labels_arr = []
    errors = []

    for idx, (pose_seq_orig, label_str) in enumerate(tqdm(segments, desc="Converting MCFS")):
        pose_seq = pose_seq_orig

        # Squeeze trailing dimension if present
        if pose_seq.ndim == 4 and pose_seq.shape[-1] == 1:
            pose_seq = pose_seq.reshape(pose_seq.shape[:-1])

        # Take only xy channels if 3D: (T, 17, 3) -> (T, 17, 2)
        if pose_seq.shape[2] > 2:
            pose_seq = pose_seq[:, :, :2]

        # Skip empty sequences
        if pose_seq.shape[0] == 0:
            errors.append(
                (
                    idx,
                    [
                        ValidationError(
                            sample_id=f"mcfs_{idx}",
                            field="frames",
                            message="Empty sequence (0 frames)",
                        )
                    ],
                )
            )
            continue

        # Normalize (poses are already 2D normalized, but apply for consistency)
        pose_seq = normalize(pose_seq)

        # Validate after normalization.
        # MCFS data is already normalized at a different scale than FSC.
        # After our normalize(), valid samples have max coord ~100-101.
        # Use max_coord=500 to keep ~99% of data while filtering truly broken
        # samples (raw pixel coords, zero spines) whose normalized max exceeds 500.
        val_errors = validate_skeleton(pose_seq, sample_id=f"mcfs_{idx}", max_coord=500.0)
        if val_errors:
            errors.append((idx, val_errors))
            continue  # Skip invalid samples

        # Map string label to int ID
        if label_str not in label_mapping:
            errors.append(
                (
                    idx,
                    [
                        ValidationError(
                            sample_id=f"mcfs_{idx}",
                            field="label",
                            message=f"Unknown label: {label_str!r}",
                        )
                    ],
                )
            )
            continue

        poses_list.append(pose_seq.astype(np.float32))
        labels_arr.append(label_mapping[label_str])

    # Save as numpy object array (for variable-length sequences)
    poses_obj = np.array(poses_list, dtype=object)
    labels_np = np.array(labels_arr, dtype=np.int32)

    poses_path = output_dir / "all_poses.npy"
    labels_path = output_dir / "all_labels.npy"

    np.save(poses_path, poses_obj)
    np.save(labels_path, labels_np)

    print(f"Saved {len(poses_list)} valid samples to {poses_path}")

    # Print validation errors summary
    if errors:
        print(f"\nValidation errors found in {len(errors)} samples:")
        for idx, err_list in errors[:5]:  # Show first 5
            print(f"  mcfs_{idx}: {len(err_list)} errors")
            for err in err_list[:2]:  # Show first 2 errors per sample
                print(f"    - {err.field}: {err.message}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Build stats
    unique_labels = sorted(set(labels_arr))
    stats: dict[str, Any] = {
        "all": len(poses_list),
        "classes": len(unique_labels),
        "created": str(np.datetime64("now")),
        "label_mapping": {k: v for k, v in label_mapping.items() if v in unique_labels},
    }

    # Save metadata
    meta_path = output_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nMetadata saved to {meta_path}")
    print(f"Total: {len(poses_list)} samples, {len(unique_labels)} classes")

    return stats


def load_unified(output_dir: Path, split: str = "all") -> tuple[list[np.ndarray], np.ndarray]:
    """Load unified MCFS dataset.

    Args:
        output_dir: Path containing unified dataset files
        split: Split name (default: 'all' — MCFS has no predefined train/test split)

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
