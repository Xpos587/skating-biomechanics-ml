#!/usr/bin/env python3
"""
Extract 10% random subset of COCO train2017 for knowledge distillation.

Purpose: Prevent catastrophic forgetting during skating domain fine-tuning.

Usage:
    python get_coco_10pct.py --coco-root /path/to/coco --output configs/coco_10pct.txt

Output:
    - configs/coco_10pct.txt: List of image paths (one per line)
    - Can be used directly in Ultralytics data.yaml or passed to training script

Requirements:
    - pycocotools: pip install pycocotools
    - COCO dataset: Download from https://cocodataset.org/#download
"""

import argparse
import random
from pathlib import Path

try:
    from pycocotools.coco import COCO
except ImportError:
    print("Error: pycocotools not installed. Run: pip install pycocotools")
    exit(1)


def extract_coco_10pct(
    coco_root: Path, output_path: Path, seed: int = 42, subset_pct: float = 0.10
) -> int:
    """
    Extract random subset of COCO train2017 image paths.

    Args:
        coco_root: Root directory containing COCO dataset
                   (expected: train2017/images/ and annotations/instances_train2017.json)
        output_path: Output text file path (one image path per line)
        seed: Random seed for reproducibility
        subset_pct: Percentage of images to extract (default: 10%)

    Returns:
        Number of images extracted
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Validate paths
    coco_root = Path(coco_root)
    if not coco_root.exists():
        raise FileNotFoundError(f"COCO root not found: {coco_root}")

    train_images_dir = coco_root / "train2017"
    annotations_file = coco_root / "annotations" / "instances_train2017.json"

    if not train_images_dir.exists():
        raise FileNotFoundError(f"COCO train2017 images not found: {train_images_dir}")

    if not annotations_file.exists():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_file}")

    # Load COCO annotations
    print(f"Loading COCO annotations from {annotations_file}...")
    coco = COCO(str(annotations_file))

    # Get all image IDs
    all_image_ids = list(coco.imgs.keys())
    total_images = len(all_image_ids)
    print(f"Total COCO train2017 images: {total_images}")

    # Calculate subset size
    subset_size = int(total_images * subset_pct)
    print(f"Extracting {subset_size} images ({subset_pct * 100:.1f}%)...")

    # Random sample
    sampled_ids = random.sample(all_image_ids, subset_size)

    # Convert to image paths
    image_paths: list[str] = []
    for img_id in sampled_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = str(train_images_dir / img_info["file_name"])
        image_paths.append(img_path)

    # Write to output file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for path in image_paths:
            f.write(f"{path}\n")

    print(f"✅ Saved {len(image_paths)} image paths to {output_path}")
    print(f"   Random seed: {seed}")
    print("   Sample paths:")
    for path in image_paths[:3]:
        print(f"     - {path}")

    return len(image_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Extract 10% random subset of COCO train2017 for KD training"
    )
    parser.add_argument(
        "--coco-root",
        type=str,
        required=True,
        help="Path to COCO dataset root (containing train2017/ and annotations/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/coco_10pct.txt",
        help="Output text file path (default: configs/coco_10pct.txt)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--subset-pct",
        type=float,
        default=0.10,
        help="Percentage of images to extract (default: 0.10 = 10%%)",
    )

    args = parser.parse_args()

    try:
        count = extract_coco_10pct(
            coco_root=Path(args.coco_root),
            output_path=Path(args.output),
            seed=args.seed,
            subset_pct=args.subset_pct,
        )
        print(f"\n✅ Success! Extracted {count} COCO images.")
        print("   Add to data.yaml train list:")
        print(f"   - {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
