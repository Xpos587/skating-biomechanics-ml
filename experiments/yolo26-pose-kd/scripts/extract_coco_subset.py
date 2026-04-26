#!/usr/bin/env python3
"""
Extract 10% random subset from COCO 2017 dataset for catastrophic forgetting prevention.

Usage:
    python extract_coco_subset.py /path/to/coco2017/train2017 /path/to/coco2017/annotations \
        --output-dir experiments/yolo26-pose-kd/data/coco_10pct --pct 0.1 --seed 42

Requirements:
    - pycocotools: pip install pycocotools
    - COCO 2017 train images and annotations

Output structure:
    coco_10pct/
        train/
            images/  # ~11,828 images (10% of 118,287)
            labels/  # YOLO format labels
"""

import argparse
import random
from pathlib import Path

try:
    from pycocotools.coco import COCO
except ImportError:
    print("Error: pycocotools not installed. Run: pip install pycocotools")
    exit(1)


def coco_to_yolo_bbox(bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """Convert COCO bbox [x, y, w, h] to YOLO format [x_center, y_center, width, height] (normalized)."""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return [x_center, y_center, width, height]


def coco_kp_to_yolo(kps: list[float], img_width: int, img_height: int) -> list[list[float]]:
    """
    Convert COCO keypoints to YOLO pose format.

    COCO format: [x1, y1, v1, x2, y2, v2, ...] where v=0 (not labeled), 1 (labeled but not visible), 2 (visible)
    YOLO format: [[x1, y1, v1], [x2, y2, v2], ...] normalized to [0, 1]

    Mapping: COCO 17kp → H3.6M 17kp (same order, verified compatible)
    """
    yolo_kps = []
    for i in range(0, len(kps), 3):
        x = kps[i] / img_width
        y = kps[i + 1] / img_height
        v = kps[i + 2]
        # Convert COCO visibility to YOLO: 0→0 (occluded), 1→1 (visible), 2→1 (visible)
        yolo_v = 1 if v >= 1 else 0
        yolo_kps.append([x, y, yolo_v])
    return yolo_kps


def extract_subset(
    images_dir: Path, annotations_file: Path, output_dir: Path, pct: float = 0.1, seed: int = 42
):
    """Extract random subset of COCO dataset and convert to YOLO format."""

    # Validate inputs
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not annotations_file.exists():
        raise ValueError(f"Annotations file not found: {annotations_file}")

    # Load COCO annotations
    print(f"Loading COCO annotations from {annotations_file}...")
    coco = COCO(str(annotations_file))

    # Get all image IDs
    all_img_ids = list(coco.imgs.keys())
    print(f"Total images in COCO train: {len(all_img_ids)}")

    # Random sample
    random.seed(seed)
    sample_size = int(len(all_img_ids) * pct)
    sampled_ids = random.sample(all_img_ids, sample_size)
    print(f"Sampled {sample_size} images ({pct * 100:.1f}%)")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_output = output_dir / "train" / "images"
    labels_output = output_dir / "train" / "labels"
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    # Process each sampled image
    processed = 0
    skipped = 0

    for img_id in sampled_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        # Source and destination paths
        src_img = images_dir / file_name
        dst_img = images_output / file_name
        label_file = labels_output / f"{src_img.stem}.txt"

        # Check if image exists
        if not src_img.exists():
            print(f"Warning: Image not found: {src_img}")
            skipped += 1
            continue

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Filter for person annotations with keypoints
        person_anns = [ann for ann in anns if ann["category_id"] == 1 and "keypoints" in ann]

        if not person_anns:
            skipped += 1
            continue

        # Copy image (symlink to save space)
        try:
            dst_img.symlink_to(src_img.resolve())
        except FileExistsError:
            pass  # Already exists

        # Write YOLO label file
        with open(label_file, "w") as f:
            for ann in person_anns:
                # Get bbox
                bbox = coco_to_yolo_bbox(ann["bbox"], img_width, img_height)

                # Get keypoints
                kps = coco_kp_to_yolo(ann["keypoints"], img_width, img_height)

                # Write YOLO pose line: class_id x_center y_center width height kps...
                kp_flat = [coord for kp in kps for coord in kp]
                line = f"0 {' '.join(f'{v:.6f}' for v in bbox + kp_flat)}\n"
                f.write(line)

        processed += 1

        if processed % 1000 == 0:
            print(f"Processed {processed}/{sample_size} images...")

    print("\nExtraction complete!")
    print(f"  Processed: {processed} images")
    print(f"  Skipped: {skipped} images")
    print(f"  Output directory: {output_dir}")
    print(f"  Images: {images_output}")
    print(f"  Labels: {labels_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 10% random subset from COCO 2017 for YOLO pose training"
    )
    parser.add_argument(
        "images_dir", type=Path, help="Path to COCO 2017 train2017 images directory"
    )
    parser.add_argument(
        "annotations_dir",
        type=Path,
        help="Path to COCO 2017 annotations directory (containing person_keypoints_train2017.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/yolo26-pose-kd/data/coco_10pct"),
        help="Output directory for YOLO format subset",
    )
    parser.add_argument(
        "--pct",
        type=float,
        default=0.1,
        help="Percentage of dataset to extract (default: 0.1 = 10%%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Find annotations file
    ann_file = args.annotations_dir / "person_keypoints_train2017.json"
    if not ann_file.exists():
        print(f"Error: Annotations file not found: {ann_file}")
        print("Expected: person_keypoints_train2017.json in annotations directory")
        exit(1)

    extract_subset(
        images_dir=args.images_dir,
        annotations_file=ann_file,
        output_dir=args.output_dir,
        pct=args.pct,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
