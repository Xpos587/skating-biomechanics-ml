#!/usr/bin/env python3
"""Convert AthletePose3D to HALPE26 COCO JSON for RTMPose fine-tuning.

Merges pre-computed _coco.npy (17kp) with projected foot keypoints (6kp)
and face duplicates (3kp) to produce HALPE26 26kp annotations.

Usage:
    uv run python scripts/prepare_athletepose3d.py
    uv run python scripts/prepare_athletepose3d.py --split train_set --sample-rate 3
    uv run python scripts/prepare_athletepose3d.py --max-sequences 10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skating_ml.datasets.coco_builder import (
    build_coco_json,
    format_keypoints,
    merge_coco_foot_keypoints,
    save_coco_json,
)
from skating_ml.datasets.projector import project_foot_frame, validate_foot_projection

DATA_ROOT = Path("data/datasets/athletepose3d")
OUTPUT_DIR = DATA_ROOT / "coco_annotations"


def discover_sequences(split: str) -> list[tuple[Path, Path, Path, Path]]:
    """Find all (npy, coco_npy, json, mp4) tuples in a split."""
    base = DATA_ROOT / "videos" / split
    if not base.exists():
        return []
    sequences = []
    for npy_path in sorted(base.glob("**/*.npy")):
        if "_coco" in npy_path.stem or "_h36m" in npy_path.stem:
            continue
        coco_path = npy_path.parent / f"{npy_path.stem}_coco.npy"
        json_path = npy_path.with_suffix(".json")
        mp4_path = npy_path.with_suffix(".mp4")
        if coco_path.exists() and json_path.exists() and mp4_path.exists():
            sequences.append((npy_path, coco_path, json_path, mp4_path))
    return sequences


def process_sequence(
    kp3d: np.ndarray,
    coco_kps: np.ndarray,
    cam: dict,
    sample_rate: int,
    image_width: int,
    image_height: int,
    video_stem: str,
    image_id_offset: int,
    ann_id_offset: int,
    scale_x: float,
    scale_y: float,
) -> tuple[list[dict], list[dict], int, int]:
    """Process a single video sequence.

    Returns:
        (images, annotations, next_image_id, next_ann_id)
    """
    n_frames = len(kp3d)
    images: list[dict] = []
    annotations: list[dict] = []
    img_id = image_id_offset
    ann_id = ann_id_offset

    for frame_idx in range(0, n_frames, sample_rate):
        coco_2d = coco_kps[frame_idx]
        foot_2d = project_foot_frame(kp3d[frame_idx], cam)
        validate_foot_projection(foot_2d, coco_2d)
        pts, vis = merge_coco_foot_keypoints(coco_2d, foot_2d)

        # Count visible keypoints (COCO 17 only for bbox calculation)
        valid_coco = vis[:17] > 0.1
        if valid_coco.sum() < 5:
            continue

        # Scale to target image resolution
        pts_scaled = pts.copy()
        pts_scaled[:, 0] *= scale_x
        pts_scaled[:, 1] *= scale_y

        # Compute bbox from visible COCO keypoints
        coco_pts_scaled = pts_scaled[:17]
        valid_pts = coco_pts_scaled[valid_coco]
        x_min = max(0, float(valid_pts[:, 0].min()) - 20)
        y_min = max(0, float(valid_pts[:, 1].min()) - 20)
        x_max = min(image_width, float(valid_pts[:, 0].max()) + 20)
        y_max = min(image_height, float(valid_pts[:, 1].max()) + 20)
        w, h = x_max - x_min, y_max - y_min

        if w <= 0 or h <= 0:
            continue

        n_visible = int((vis > 0.1).sum())
        if n_visible < 10:
            continue

        img_id += 1
        ann_id += 1

        images.append(
            {
                "file_name": f"{video_stem}/frame_{frame_idx:06d}.jpg",
                "id": img_id,
                "width": image_width,
                "height": image_height,
            }
        )

        annotations.append(
            {
                "image_id": img_id,
                "id": ann_id,
                "keypoints": format_keypoints(pts_scaled, vis),
                "num_keypoints": n_visible,
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )

    return images, annotations, img_id, ann_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert AthletePose3D to HALPE26 COCO JSON")
    parser.add_argument(
        "--split", default="train_set", choices=["train_set", "valid_set", "test_set"]
    )
    parser.add_argument("--sample-rate", type=int, default=3, help="Extract every Nth frame")
    parser.add_argument("--max-sequences", type=int, default=0, help="Max sequences (0=all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-width", type=int, default=1920, help="Target image width")
    parser.add_argument("--target-height", type=int, default=1088, help="Target image height")
    args = parser.parse_args()

    with (DATA_ROOT / "cam_param.json").open() as f:
        cam_params = json.load(f)

    sequences = discover_sequences(args.split)
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    print(f"Found {len(sequences)} sequences in {args.split}")
    print(f"Sample rate: {args.sample_rate}, target: {args.target_width}x{args.target_height}")
    print()

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    img_id = 0
    ann_id = 0
    skipped = 0

    for npy_path, coco_path, json_path, _mp4_path in tqdm(
        sequences, desc=f"Processing {args.split}"
    ):
        kp3d = np.load(npy_path)
        coco_kps = np.load(coco_path)

        if kp3d.shape[0] != coco_kps.shape[0]:
            skipped += 1
            continue

        # Skip sequences with fewer than 113 keypoints (missing foot markers)
        if kp3d.shape[1] < 113:
            skipped += 1
            continue

        with json_path.open() as f:
            meta = json.load(f)
        cam_key = meta["cam"]
        if cam_key not in cam_params:
            skipped += 1
            continue

        cam = cam_params[cam_key]
        scale_x = args.target_width / meta.get("video_width", 1920)
        scale_y = args.target_height / meta.get("video_height", 1088)

        images, annotations, img_id, ann_id = process_sequence(
            kp3d,
            coco_kps,
            cam,
            args.sample_rate,
            args.target_width,
            args.target_height,
            npy_path.stem,
            img_id,
            ann_id,
            scale_x,
            scale_y,
        )

        all_images.extend(images)
        all_annotations.extend(annotations)

    if skipped:
        print(f"\nSkipped {skipped} sequences (frame mismatch or missing camera)")

    print(f"\nTotal images: {len(all_images)}")
    print(f"Total annotations: {len(all_annotations)}")

    if args.dry_run:
        print("Dry run — nothing written.")
        return 0

    output_path = OUTPUT_DIR / f"athletepose3d_{args.split}.json"
    coco_data = build_coco_json(all_images, all_annotations)
    save_coco_json(coco_data, str(output_path))
    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
