#!/usr/bin/env python3
"""
Integrate AthletePose3D figure skating dataset into YOLO training pipeline.

AP3D is already in COCO format with COCO 17kp (H3.6M compatible).
This script creates symlinks to AP3D images and converts annotations to YOLO format.

AP3D sport mapping (from data.zip JSON metadata):
  S1 = figure skating (jumps: Axel, Flip, Loop, Lutz, Salchow, Toeloop, Comb)
  S2 = figure skating (same jumps)
  S3 = track & field (tnf) -- EXCLUDED
  S4 = running (rm) -- EXCLUDED
  S5 = track & field (Discus, Javelin, Shot, Spin, Glide) -- EXCLUDED

Usage:
    python integrate_ap3d_fs.py

Input:
    /root/data/datasets/raw/athletepose3d/pose_2d/
      annotations/{split}.json
      {split}/  (images)

Output:
    /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/ap3d-fs/
      {split}/images/  (symlinks)
      {split}/labels/  (YOLO format, kpt_shape [17, 3])
"""

import json
from collections import Counter
from pathlib import Path

# Only S1 and S2 are figure skating
FS_SPORTS = {"S1", "S2"}


def coco_to_yolo_bbox(bbox: list, img_width: int, img_height: int) -> list:
    """Convert COCO bbox [x, y, w, h] to YOLO format [x_center, y_center, width, height] (normalized)."""
    x, y, w, h = bbox
    yolo_bbox = [
        (x + w / 2) / img_width,
        (y + h / 2) / img_height,
        w / img_width,
        h / img_height,
    ]
    return [max(0.0, min(1.0, v)) for v in yolo_bbox]


def coco_kp_to_yolo(kps: list, img_width: int, img_height: int) -> list:
    """Convert COCO 17kp to YOLO pose format.

    COCO v: 0=not labeled, 1=labeled invisible, 2=labeled visible
    YOLO v: 0=occluded, 1=visible
    """
    yolo_kps = []
    for i in range(0, len(kps), 3):
        x = max(0.0, min(1.0, kps[i] / img_width))
        y = max(0.0, min(1.0, kps[i + 1] / img_height))
        v = 1 if kps[i + 2] >= 1 else 0
        yolo_kps.extend([x, y, v])
    return yolo_kps


def process_split(ap3d_root: Path, split: str, output_root: Path) -> dict:
    """Process a single split (train/valid/test). Returns stats dict."""
    # AP3D uses train_set/valid_set/test_set naming
    ann_file = ap3d_root / "annotations" / f"{split}_set.json"
    images_dir = ap3d_root / f"{split}_set"

    if not ann_file.exists():
        print(f"  Skipping {split}: {ann_file} not found")
        return {"split": split, "total": 0, "fs": 0, "skipped": 0, "errors": 0}

    with open(ann_file) as f:
        coco_data = json.load(f)

    total = len(coco_data["annotations"])
    images_map = {img["id"]: img for img in coco_data["images"]}

    # Build sport index from info field
    # info[i] corresponds to annotations[i] (1:1 mapping)
    sport_map = {}
    for i, info_entry in enumerate(coco_data["info"]):
        vp = info_entry["video_path"]
        parts = vp.split("/")
        sport = parts[1] if len(parts) >= 2 else "unknown"
        sport_map[i] = sport

    images_output = output_root / split / "images"
    labels_output = output_root / split / "labels"
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0
    action_counter = Counter()

    for i, ann in enumerate(coco_data["annotations"]):
        sport = sport_map.get(i, "unknown")

        # Skip non-figure-skating
        if sport not in FS_SPORTS:
            skipped += 1
            continue

        # Skip non-person annotations
        if ann["category_id"] != 1:
            skipped += 1
            continue

        img_id = ann["image_id"]
        img_info = images_map.get(img_id)
        if not img_info:
            skipped += 1
            continue

        file_name = img_info["file_name"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        src_img = images_dir / file_name
        dst_img = images_output / file_name
        label_file = labels_output / f"{src_img.stem}.txt"

        if not src_img.exists():
            errors += 1
            continue

        # Create symlink to save space
        try:
            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())
        except FileExistsError:
            pass
        except OSError as e:
            print(f"  Symlink error: {e}")
            errors += 1
            continue

        # Convert annotation to YOLO format
        bbox = coco_to_yolo_bbox(ann["bbox"], img_width, img_height)
        kps = coco_kp_to_yolo(ann["keypoints"], img_width, img_height)

        # Write YOLO label: class x y w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
        with open(label_file, "w") as f:
            parts = [str(0)] + [f"{v:.6f}" for v in bbox] + [f"{v:.6f}" for v in kps]
            f.write(" ".join(parts) + "\n")

        # Track action type
        vp = coco_data["info"][i]["video_path"]
        vid_name = vp.split("/")[-1].replace(".mp4", "")
        action = vid_name.split("_")[0]
        action_counter[action] += 1

        processed += 1
        if processed % 10000 == 0:
            print(f"  [{split}] Processed {processed}...")

    return {
        "split": split,
        "total": total,
        "fs": processed,
        "skipped": skipped,
        "errors": errors,
        "actions": dict(action_counter),
    }


def main():
    project_root = Path("/root/skating-biomechanics-ml")
    ap3d_root = Path("/root/data/datasets/raw/athletepose3d/pose_2d")
    output_root = project_root / "experiments/yolo26-pose-kd/data/ap3d-fs"

    print(f"AP3D root: {ap3d_root}")
    print(f"Output root: {output_root}")
    print()

    total_fs = 0
    total_skipped = 0
    all_actions = Counter()

    for split in ["train", "valid", "test"]:
        print(f"Processing {split}...")
        stats = process_split(ap3d_root, split, output_root)
        print(f"  Total: {stats['total']}")
        print(f"  Figure skating: {stats['fs']}")
        print(f"  Skipped (non-FS): {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")
        if stats.get("actions"):
            for action, count in sorted(stats["actions"].items(), key=lambda x: -x[1]):
                print(f"    {action}: {count}")
                all_actions[action] += count
        print()

        total_fs += stats["fs"]
        total_skipped += stats["skipped"]

    # Verify output
    print("=" * 60)
    print("INTEGRATION COMPLETE")
    print(f"  Total figure skating frames: {total_fs}")
    print(f"  Total non-FS skipped: {total_skipped}")
    print()
    print("  Action distribution (all splits):")
    for action, count in sorted(all_actions.items(), key=lambda x: -x[1]):
        print(f"    {action}: {count}")
    print()

    # Count actual files
    for split in ["train", "valid", "test"]:
        img_dir = output_root / split / "images"
        lbl_dir = output_root / split / "labels"
        n_img = len(list(img_dir.glob("*.jpg")))
        n_lbl = len(list(lbl_dir.glob("*.txt")))
        print(f"  {split}: {n_img} images, {n_lbl} labels")
    print()
    print(f"  Output: {output_root}")


if __name__ == "__main__":
    main()
