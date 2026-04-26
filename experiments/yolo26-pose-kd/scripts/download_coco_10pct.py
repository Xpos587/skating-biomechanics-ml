#!/usr/bin/env python3
"""
Download 10% subset of COCO train2017 (person keypoints only) for catastrophic forgetting prevention.

Strategy:
1. Download person_keypoints_train2017.json annotations (~200MB)
2. Sample 10% of images that have person keypoints annotations
3. Download only those images (not the full 118K)
4. Convert to YOLO pose format

Usage:
    python download_coco_10pct.py --output-dir /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/coco-10pct

Requirements:
    pycocotools, requests
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from pycocotools.coco import COCO
except ImportError:
    print("Error: pycocotools not installed. Run: pip install pycocotools")
    exit(1)

import requests

# COCO 2017 train image base URL
COCO_IMAGE_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_BASE = "http://images.cocodataset.org/train2017/"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress display."""
    if desc:
        print(f"  Downloading {desc}...")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (10 * 1024 * 1024) < 8192:
                    pct = downloaded / total * 100
                    print(
                        f"    {downloaded / (1024 * 1024):.0f}/{total / (1024 * 1024):.0f} MB ({pct:.0f}%)"
                    )
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def download_image(args: tuple[str, Path, int]) -> tuple[str, bool]:
    """Download a single COCO image. Returns (filename, success)."""
    file_name, dest_dir, attempt = args
    url = COCO_IMG_BASE + file_name
    dest = dest_dir / file_name
    if dest.exists():
        return (file_name, True)

    for attempt_num in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return (file_name, False)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            return (file_name, True)
        except Exception:
            time.sleep(1 * (attempt_num + 1))
    return (file_name, False)


def coco_to_yolo_bbox(bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """Convert COCO bbox [x, y, w, h] to YOLO format [x_center, y_center, width, height]."""
    x, y, w, h = bbox
    return [(x + w / 2) / img_width, (y + h / 2) / img_height, w / img_width, h / img_height]


def coco_kp_to_yolo(kps: list[float], img_width: int, img_height: int) -> list[float]:
    """Convert COCO keypoints to YOLO pose format (normalized x, y, v)."""
    yolo_kps = []
    for i in range(0, len(kps), 3):
        x = kps[i] / img_width
        y = kps[i + 1] / img_height
        v = 1 if kps[i + 2] >= 1 else 0
        yolo_kps.extend([x, y, v])
    return yolo_kps


def main():
    parser = argparse.ArgumentParser(description="Download 10% COCO subset for YOLO pose training")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/coco-10pct"),
        help="Output directory for YOLO format dataset",
    )
    parser.add_argument(
        "--pct", type=float, default=0.10, help="Fraction of COCO to extract (default: 0.10)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=16, help="Parallel download threads")
    parser.add_argument(
        "--ann-only", action="store_true", help="Only download annotations, skip images"
    )
    args = parser.parse_args()

    coco_root = Path("/root/data/datasets/raw/coco2017")
    coco_root.mkdir(parents=True, exist_ok=True)
    ann_dir = coco_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    ann_file = ann_dir / "person_keypoints_train2017.json"

    # Step 1: Download annotations if not present
    if not ann_file.exists():
        print("Step 1: Downloading COCO annotations (~240MB)...")
        ann_zip = coco_root / "annotations_trainval2017.zip"
        if not ann_zip.exists():
            if not download_file(COCO_ANN_URL, ann_zip, "annotations_trainval2017.zip"):
                print("FAILED to download annotations")
                exit(1)
        # Extract
        print("  Extracting annotations...")
        import zipfile

        with zipfile.ZipFile(ann_zip, "r") as z:
            z.extractall(ann_dir.parent)
        print(f"  Annotations extracted to {ann_dir}")
    else:
        print(f"Step 1: Annotations already exist at {ann_file}")

    # Step 2: Load annotations and sample images with person keypoints
    print("\nStep 2: Loading annotations and sampling...")
    coco = COCO(str(ann_file))

    # Get all image IDs that have person keypoints annotations
    all_img_ids = list(coco.imgs.keys())
    print(f"  Total images in annotations: {len(all_img_ids)}")

    # Filter to images that have at least one person keypoint annotation
    # Use getAnnIds with catIds=1 for person annotations, then filter for keypoints
    imgs_with_kps = set()
    person_ann_ids = coco.getAnnIds(catIds=1)
    print(f"  Total person annotations: {len(person_ann_ids)}")
    person_anns = coco.loadAnns(person_ann_ids)
    for ann in person_anns:
        if "keypoints" in ann and sum(ann["keypoints"][2::3]) > 0:
            imgs_with_kps.add(ann["image_id"])

    print(f"  Images with person keypoints: {len(imgs_with_kps)}")

    random.seed(args.seed)
    sample_size = int(len(imgs_with_kps) * args.pct)
    sampled_ids = random.sample(list(imgs_with_kps), sample_size)
    print(f"  Sampled {sample_size} images ({args.pct * 100:.1f}%)")

    if args.ann_only:
        print("\n--ann-only: Skipping image download.")
        # Still write the list of needed images
        img_list_path = coco_root / "sampled_image_ids.json"
        with open(img_list_path, "w") as f:
            json.dump(sampled_ids, f)
        print(f"  Saved sampled image IDs to {img_list_path}")
        return

    # Step 3: Download images
    print(f"\nStep 3: Downloading {sample_size} images ({args.workers} threads)...")
    images_dir = coco_root / "train2017"
    images_dir.mkdir(parents=True, exist_ok=True)

    file_names = []
    for img_id in sampled_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_names.append(img_info["file_name"])

    # Download in parallel
    download_args = [(fn, images_dir, 0) for fn in file_names]
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_image, arg): arg[0] for arg in download_args}
        for i, future in enumerate(as_completed(futures)):
            fn, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            if (success + failed) % 500 == 0:
                print(f"  Progress: {success + failed}/{sample_size} (ok={success}, fail={failed})")

    print(f"  Download complete: {success} ok, {failed} failed")

    # Step 4: Convert to YOLO pose format
    print("\nStep 4: Converting to YOLO pose format...")
    output_dir = args.output_dir
    out_images = output_dir / "train" / "images"
    out_labels = output_dir / "train" / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    processed = 0
    for img_id in sampled_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img = images_dir / file_name
        if not src_img.exists():
            continue

        # Get person annotations with keypoints
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1)
        anns = coco.loadAnns(ann_ids)
        person_anns = [a for a in anns if "keypoints" in a and sum(a["keypoints"][2::3]) > 0]
        if not person_anns:
            continue

        # Symlink image
        dst_img = out_images / file_name
        if not dst_img.exists():
            try:
                dst_img.symlink_to(src_img.resolve())
            except FileExistsError:
                pass

        # Write label
        label_path = out_labels / f"{src_img.stem}.txt"
        with open(label_path, "w") as f:
            for ann in person_anns:
                bbox = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                kps = coco_kp_to_yolo(ann["keypoints"], img_w, img_h)
                vals = bbox + kps
                line = "0 " + " ".join(f"{v:.6f}" for v in vals) + "\n"
                f.write(line)

        processed += 1

    # Step 5: Write data.yaml
    print("\nStep 5: Writing data.yaml...")
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "",
        "test": "",
        "kpt_shape": [17, 3],
        "names": {0: "person"},
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        # Simple YAML writer (no pyyaml dependency needed)
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"test: {data_yaml['test']}\n")
        f.write(f"kpt_shape: {data_yaml['kpt_shape']}\n")
        f.write("names:\n")
        f.write("  0: person\n")

    print(f"\n{'=' * 60}")
    print("COCO 10% subset extraction complete!")
    print(f"  Images with labels: {processed}")
    print(f"  Output: {output_dir}")
    print(f"  data.yaml: {yaml_path}")
    print("  kpt_shape: [17, 3] (COCO 17kp)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
