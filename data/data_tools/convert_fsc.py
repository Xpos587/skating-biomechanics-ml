"""FSC (Figure Skating Classification) dataset converter.

Converts FSC pkl files to unified numpy format.
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


def convert_fsc(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Convert FSC pkl files to unified numpy format.

    Args:
        raw_dir: Path containing train_data.pkl, train_label.pkl, test_data.pkl, test_label.pkl
        output_dir: Where to save unified dataset (will create if needed)

    Returns:
        Statistics dict with train/test counts, classes, and creation time
    """
    import pickle

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    errors = []

    for split in ["train", "test"]:
        print(f"\nProcessing {split} split...")

        # Load data and labels
        data_path = raw_dir / f"{split}_data.pkl"
        label_path = raw_dir / f"{split}_label.pkl"

        if not data_path.exists() or not label_path.exists():
            print(f"Warning: {split} data files not found, skipping...")
            continue

        with data_path.open("rb") as f:
            data_list = pickle.load(f)
        with label_path.open("rb") as f:
            labels_list = pickle.load(f)

        print(f"Loaded {len(data_list)} samples")

        # Process each sequence
        poses_list = []
        labels_arr = []

        for idx, (pose_seq_orig, label) in enumerate(
            tqdm(zip(data_list, labels_list, strict=True), total=len(data_list))
        ):
            # Squeeze trailing dimension if present: (T, 17, 3, 1) -> (T, 17, 3)
            pose_seq = pose_seq_orig
            if pose_seq.ndim == 4 and pose_seq.shape[-1] == 1:
                pose_seq = pose_seq.reshape(pose_seq.shape[:-1])

            # Take only xy channels (first 2): (T, 17, 3) -> (T, 17, 2)
            pose_seq = pose_seq[:, :, :2]

            # Skip empty sequences
            if pose_seq.shape[0] == 0:
                errors.append(
                    (
                        split,
                        idx,
                        [
                            ValidationError(
                                sample_id=f"{split}_{idx}",
                                field="frames",
                                message="Empty sequence (0 frames)",
                            )
                        ],
                    )
                )
                continue

            # Normalize FIRST (pixel coords → normalized coords)
            pose_seq = normalize(pose_seq)

            # Validate AFTER normalization (use relaxed bounds for extreme poses)
            val_errors = validate_skeleton(pose_seq, sample_id=f"{split}_{idx}", max_coord=100.0)
            if val_errors:
                errors.append((split, idx, val_errors))
                continue  # Skip invalid samples

            poses_list.append(pose_seq.astype(np.float32))
            labels_arr.append(label)

        # Save as numpy object array (for variable-length sequences)
        poses_obj = np.array(poses_list, dtype=object)
        labels_np = np.array(labels_arr, dtype=np.int32)

        poses_path = output_dir / f"{split}_poses.npy"
        labels_path = output_dir / f"{split}_labels.npy"

        np.save(poses_path, poses_obj)
        np.save(labels_path, labels_np)

        stats[split] = len(poses_list)
        print(f"Saved {len(poses_list)} valid samples to {poses_path}")

    # Print validation errors summary
    if errors:
        print(f"\nValidation errors found in {len(errors)} samples:")
        for split, idx, err_list in errors[:5]:  # Show first 5
            print(f"  {split}_{idx}: {len(err_list)} errors")
            for err in err_list[:2]:  # Show first 2 errors per sample
                print(f"    - {err.field}: {err.message}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Save metadata
    num_classes = 64  # FSC has 64 element classes
    stats["classes"] = num_classes
    stats["created"] = str(np.datetime64("now"))

    meta_path = output_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nMetadata saved to {meta_path}")
    print(f"Total: {stats.get('train', 0)} train, {stats.get('test', 0)} test samples")

    return stats


def load_unified(output_dir: Path, split: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Load unified FSC dataset.

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
