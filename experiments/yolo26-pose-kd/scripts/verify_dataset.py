#!/usr/bin/env python3
"""
Verify dataset paths and counts for Knowledge Distillation training.

Checks:
- FineFS train/val images and labels
- AP3D-FS integration status
- COCO 10% subset status
- Total counts

Usage:
    python verify_dataset.py
"""

from pathlib import Path


def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files matching pattern in directory."""
    if not directory.exists():
        return 0
    try:
        return len(list(directory.glob(pattern)))
    except PermissionError:
        return 0


def verify_dataset():
    """Verify all dataset paths and counts."""

    # Auto-detect project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent
    data_root = project_root / "experiments/yolo26-pose-kd/data"

    print("=" * 60)
    print("Dataset Verification Report")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Data root: {data_root}")

    # FineFS
    print("\n1. FineFS (Primary - Figure Skating)")
    print("-" * 60)
    finefs_train_img = data_root / "finefs/train/images"
    finefs_train_lbl = data_root / "finefs/train/labels"
    finefs_val_img = data_root / "finefs/val/images"
    finefs_val_lbl = data_root / "finefs/val/labels"

    finefs_train_imgs = count_files(finefs_train_img, "*.jpg")
    finefs_train_lbls = count_files(finefs_train_lbl, "*.txt")
    finefs_val_imgs = count_files(finefs_val_img, "*.jpg")
    finefs_val_lbls = count_files(finefs_val_lbl, "*.txt")

    print(f"Train images: {finefs_train_imgs}")
    print(f"Train labels: {finefs_train_lbls}")
    print(f"Val images: {finefs_val_imgs}")
    print(f"Val labels: {finefs_val_lbls}")

    finefs_status = (
        "✓ OK"
        if finefs_train_imgs == finefs_train_lbls and finefs_val_imgs == finefs_val_lbls
        else "✗ MISMATCH"
    )
    print(f"Status: {finefs_status}")

    # AP3D-FS
    print("\n2. AP3D-FS (AthletePose3D - Figure Skating)")
    print("-" * 60)
    ap3d_train_img = data_root / "ap3d-fs/train/images"
    ap3d_train_lbl = data_root / "ap3d-fs/train/labels"

    if ap3d_train_img.exists():
        ap3d_train_imgs = count_files(ap3d_train_img, "*.jpg")
        ap3d_train_lbls = count_files(ap3d_train_lbl, "*.txt")
        print(f"Train images: {ap3d_train_imgs}")
        print(f"Train labels: {ap3d_train_lbls}")
        ap3d_status = "✓ OK" if ap3d_train_imgs == ap3d_train_lbls else "✗ MISMATCH"
        print(f"Status: {ap3d_status}")
    else:
        print("Status: ✗ NOT INTEGRATED (run integrate_ap3d_fs.py)")
        ap3d_train_imgs = 0

    # COCO 10%
    print("\n3. COCO 10% (Catastrophic Forgetting Prevention)")
    print("-" * 60)
    coco_train_img = data_root / "coco_10pct/train/images"
    coco_train_lbl = data_root / "coco_10pct/train/labels"

    if coco_train_img.exists():
        coco_train_imgs = count_files(coco_train_img, "*.jpg")
        coco_train_lbls = count_files(coco_train_lbl, "*.txt")
        print(f"Train images: {coco_train_imgs}")
        print(f"Train labels: {coco_train_lbls}")
        coco_status = "✓ OK" if coco_train_imgs == coco_train_lbls else "✗ MISMATCH"
        print(f"Status: {coco_status}")
    else:
        print("Status: ✗ NOT INTEGRATED (run extract_coco_subset.py)")
        coco_train_imgs = 0

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_train = finefs_train_imgs + ap3d_train_imgs + coco_train_imgs
    total_val = finefs_val_imgs

    print(f"Total train images: {total_train:,}")
    print(f"  - FineFS: {finefs_train_imgs:,} ({finefs_train_imgs / total_train * 100:.1f}%)")
    if ap3d_train_imgs > 0:
        print(f"  - AP3D-FS: {ap3d_train_imgs:,} ({ap3d_train_imgs / total_train * 100:.1f}%)")
    if coco_train_imgs > 0:
        print(f"  - COCO 10%: {coco_train_imgs:,} ({coco_train_imgs / total_train * 100:.1f}%)")
    print(f"Total val images: {total_val:,}")

    print("\nNext steps:")
    if not ap3d_train_img.exists():
        print("  1. Run: python scripts/integrate_ap3d_fs.py")
    if not coco_train_img.exists():
        print(
            "  2. Run: python scripts/extract_coco_subset.py /path/to/coco2017/train2017 /path/to/coco2017/annotations"
        )
    if ap3d_train_img.exists() and coco_train_img.exists():
        print("  ✓ All datasets integrated! Ready for training.")

    # Update data.yaml with actual paths
    print("\n4. Updating data.yaml with local paths...")
    data_yaml = project_root / "experiments/yolo26-pose-kd/configs/data.yaml"
    if data_yaml.exists():
        with open(data_yaml) as f:
            content = f.read()

        # Replace root path with actual project root
        content_updated = content.replace("/root/skating-biomechanics-ml", str(project_root))

        if content != content_updated:
            with open(data_yaml, "w") as f:
                f.write(content_updated)
            print(f"  ✓ Updated data.yaml with local path: {project_root}")
        else:
            print("  ✓ data.yaml already has correct path")


if __name__ == "__main__":
    verify_dataset()
