#!/usr/bin/env python3
"""Build COCO-format dataset JSON for mmpose from existing image dirs + SimCC labels.

Usage:
    python build_mmpose_dataset.py \
        --image-dirs data/finefs/train/images data/ap3d-fs/train/images \
        --simcc data/teacher_simcc.npz \
        --output data/coco_skating_train.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def build_coco_json(image_dirs, simcc_path, output_path):
    """Build COCO JSON with SimCC labels in 'simcc' field."""
    images = []
    annotations = []
    img_id = 0
    ann_id = 0

    simcc_data = np.load(simcc_path) if simcc_path else None
    index_map = json.loads(simcc_data["index"]) if simcc_data else {}

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

            if idx is not None and simcc_data is not None:
                simcc_x = simcc_data["simcc_x"][idx]
                simcc_y = simcc_data["simcc_y"][idx]
            else:
                simcc_x = None
                simcc_y = None

            if simcc_x is not None:
                x_coords = simcc_x.argmax(axis=-1) / simcc_x.shape[-1]
                y_coords = simcc_y.argmax(axis=-1) / simcc_y.shape[-1]
                keypoints = []
                for k in range(17):
                    keypoints.extend(
                        [
                            float(x_coords[k] * w),
                            float(y_coords[k] * h),
                            2,
                        ]
                    )
            else:
                keypoints = [0.0] * 51

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "num_keypoints": 17,
                    "bbox": [0, 0, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "simcc_x": simcc_x.tolist() if simcc_x is not None else None,
                    "simcc_y": simcc_y.tolist() if simcc_y is not None else None,
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
    parser.add_argument("--simcc", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_coco_json(args.image_dirs, args.simcc, args.output)


if __name__ == "__main__":
    main()
