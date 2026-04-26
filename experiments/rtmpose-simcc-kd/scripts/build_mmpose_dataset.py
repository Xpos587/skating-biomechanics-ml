#!/usr/bin/env python3
"""Build COCO-format dataset JSON for mmpose from existing image dirs + teacher coords.

Usage:
    python build_mmpose_dataset.py \
        --image-dirs data/finefs/train/images data/ap3d-fs/train/images \
        --coords data/teacher_coords.h5 \
        --output data/coco_skating_train.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from simcc_label_generator import transform_teacher_coords


def build_coco_json(image_dirs, coords_path, output_path):
    """Build COCO JSON with keypoints from teacher coords."""
    images = []
    annotations = []
    img_id = 0
    ann_id = 0

    with h5py.File(coords_path, "r") as f:
        all_coords = f["coords"][:]
        all_conf = f["confidence"][:]
        all_crop_params = f["crop_params"][:]
        index = json.loads(f.attrs["index"])
        index_map = {v: k for k, v in index.items()}

    for img_dir in image_dirs:
        img_dir = Path(img_dir)
        for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc=f"Processing {img_dir}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            rel_path = str(img_path.relative_to(img_dir.parent.parent))
            idx = index_map.get(rel_path)

            images.append(
                {
                    "id": img_id,
                    "file_name": str(img_path),
                    "height": h,
                    "width": w,
                }
            )

            if idx is not None:
                crop_coords = all_coords[idx]
                conf = all_conf[idx]
                cp = all_crop_params[idx]
                img_coords = transform_teacher_coords(crop_coords, cp)
                keypoints = []
                for k in range(17):
                    x = float(img_coords[k, 0] * w)
                    y = float(img_coords[k, 1] * h)
                    v = 2 if conf[k] > 0.5 else 1
                    keypoints.extend([x, y, v])
                num_kpts = int((conf > 0.5).sum())
            else:
                keypoints = [0.0] * 51
                num_kpts = 0

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "num_keypoints": num_kpts,
                    "bbox": [0, 0, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )

            img_id += 1
            ann_id += 1

    categories = [
        {"id": 1, "name": "person", "keypoints": [f"kp_{i}" for i in range(17)], "skeleton": []}
    ]

    coco = {"images": images, "annotations": annotations, "categories": categories}

    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(f"Wrote {len(images)} images, {len(annotations)} annotations to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dirs", nargs="+", required=True)
    parser.add_argument("--coords", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_coco_json(args.image_dirs, args.coords, args.output)


if __name__ == "__main__":
    main()
