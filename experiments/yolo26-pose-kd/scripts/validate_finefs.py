#!/usr/bin/env python3
"""
Validate FineFS YOLO conversion output.

Checks:
- Frame counts match expected
- Keypoint visibility distribution
- Label format correctness
- Coordinate ranges [0,1]
"""

from collections import Counter
from pathlib import Path

import numpy as np

# Paths
OUTPUT_ROOT = Path(
    "/home/michael/Github/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/finefs"
)
TRAIN_DIR = OUTPUT_ROOT / "train"
VAL_DIR = OUTPUT_ROOT / "val"


def analyze_split(split_dir: Path, split_name: str):
    """Analyze one split (train/val)."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing {split_name} split")
    print(f"{'=' * 60}")

    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"

    # Count files
    label_files = sorted(labels_dir.glob("*.txt"))
    image_files = sorted(images_dir.glob("*.jpg"))

    print(f"Label files: {len(label_files)}")
    print(f"Image files: {len(image_files)}")

    if len(label_files) != len(image_files):
        print("⚠️  WARNING: Mismatch between labels and images!")

    # Analyze labels
    all_keypoints = []
    visibility_counts = Counter()
    bbox_sizes = []

    for label_file in label_files[:1000]:  # Sample first 1000 for speed
        with open(label_file) as f:
            line = f.readline().strip()

        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            print(f"⚠️  Invalid label format: {label_file}")
            continue

        # Parse bounding box
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        bbox_sizes.append((width, height))

        # Parse keypoints
        keypoints = []
        for i in range(5, len(parts), 3):
            if i + 2 >= len(parts):
                break
            x = float(parts[i])
            y = float(parts[i + 1])
            v = int(parts[i + 2])

            keypoints.append((x, y, v))
            visibility_counts[v] += 1

            # Check coordinate ranges
            if not (0 <= x <= 1 and 0 <= y <= 1):
                print(f"⚠️  Out of range coordinates in {label_file}: x={x}, y={y}")

        all_keypoints.extend(keypoints)

    # Convert to array for analysis
    all_keypoints = np.array(all_keypoints)

    print("\n--- Keypoint Visibility Distribution ---")
    total_kps = len(all_keypoints)
    for v, count in sorted(visibility_counts.items()):
        pct = 100 * count / total_kps
        print(f"  Visibility {v}: {count:6d} ({pct:5.2f}%)")

    print("\n--- Bounding Box Statistics ---")
    bbox_sizes = np.array(bbox_sizes)
    print(f"  Width:  mean={bbox_sizes[:, 0].mean():.4f}, std={bbox_sizes[:, 0].std():.4f}")
    print(f"  Height: mean={bbox_sizes[:, 1].mean():.4f}, std={bbox_sizes[:, 1].std():.4f}")

    print("\n--- Keypoint Coordinate Statistics ---")
    print(f"  X: min={all_keypoints[:, 0].min():.4f}, max={all_keypoints[:, 0].max():.4f}")
    print(f"  Y: min={all_keypoints[:, 1].min():.4f}, max={all_keypoints[:, 1].max():.4f}")

    # Sample label
    print("\n--- Sample Label ---")
    sample_label = label_files[0]
    print(f"File: {sample_label.name}")
    with open(sample_label) as f:
        content = f.readline().strip()
        print(f"Content (first 200 chars): {content[:200]}...")


def main():
    """Main validation function."""
    print("=" * 60)
    print("FineFS YOLO Conversion Validation")
    print("=" * 60)

    # Analyze both splits
    analyze_split(TRAIN_DIR, "train")
    analyze_split(VAL_DIR, "val")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    train_labels = len(list((TRAIN_DIR / "labels").glob("*.txt")))
    val_labels = len(list((VAL_DIR / "labels").glob("*.txt")))

    print(f"Total train frames: {train_labels}")
    print(f"Total val frames: {val_labels}")
    print(f"Total frames: {train_labels + val_labels}")
    print(f"Train/Val ratio: {train_labels / val_labels:.2f}")

    # Expected vs actual
    expected_train = 8904
    expected_val = 2007

    if train_labels == expected_train:
        print(f"✅ Train frame count matches expected ({expected_train})")
    else:
        print(f"⚠️  Train frame count mismatch: expected {expected_train}, got {train_labels}")

    if val_labels == expected_val:
        print(f"✅ Val frame count matches expected ({expected_val})")
    else:
        print(f"⚠️  Val frame count mismatch: expected {expected_val}, got {val_labels}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
