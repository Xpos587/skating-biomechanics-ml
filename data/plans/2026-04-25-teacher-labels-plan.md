# Teacher Labels: MogaNet-B → YOLO26s via Pseudo-Labels

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace custom KD loss pipeline with simple pseudo-labeling — use MogaNet-B predictions as YOLO training labels, then fine-tune YOLO26s-pose with standard Ultralytics CLI.

**Architecture:** MogaNet-B (teacher) generates keypoints for all 270K training images offline. Keypoints are converted from crop space to YOLO label format. YOLO26s-pose fine-tunes on teacher labels using standard `yolo train` — zero custom code, zero coordinate transforms during training.

**Tech Stack:** h5py, numpy, Ultralytics YOLO

---

## Context

### Problem
Cross-architecture KD (MogaNet-B heatmap → YOLO26s regression) has failed after 35+ iterations due to coordinate system mismatches, epoch tracking bugs, and fundamental architecture incompatibility.

### Solution
Instead of matching teacher output during training (online KD), use teacher output AS labels (offline pseudo-labeling). One-time conversion, then standard fine-tuning.

### Available Data
- `teacher_coords.h5`: 291,901 entries with (coords, confidence, crop_params)
  - coords: (N, 17, 2) float32 — teacher keypoints in [0,1] crop space
  - confidence: (N, 17) float32 — **NaN for all entries** (raw DeconvHead output, not usable)
  - crop_params: (N, 6) float32 — (x1, y1, crop_w, crop_h, img_w, img_h) in original pixels
  - All 291,901 have valid crop_params
  - 612 NaN values in coords (0.2%)
- Training coverage: 229,169 FineFS + 35,705 AP3D-FS + 5,659 COCO = 270,533 (100%)
- 17,338 images have multiple teacher entries (multi-person)

### Coordinate Conversion
```
teacher (crop [0,1]) → pixel in crop → pixel in original → YOLO [0,1]

orig_x = teacher_x * crop_w + crop_x1
orig_y = teacher_y * crop_h + crop_y1
yolo_x = orig_x / img_w
yolo_y = orig_y / img_h
```

### Files
- Create: `experiments/yolo26-pose-kd/scripts/convert_teacher_labels.py` — conversion script
- Create: `experiments/yolo26-pose-kd/configs/data_teacher.yaml` — training config
- Modify: `experiments/yolo26-pose-kd/data/teacher-labels/` — output labels (generated on server)
- Read: `experiments/yolo26-pose-kd/scripts/extract_teacher_coords.py` — reference for HDF5 structure
- Read: `experiments/yolo26-pose-kd/configs/data.yaml` — reference for data config format

---

### Task 0: Stop v35d Training

**Files:** None (server operation)

- [ ] **Step 1: Stop v35d to save GPU budget**

```bash
ssh vastai "tmux send-keys -t kd C-c"
```

- [ ] **Step 2: Verify training stopped**

```bash
ssh vastai "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"
```

Expected: utilization drops to 0-5%

---

### Task 1: Write Conversion Script

**Files:**
- Create: `experiments/yolo26-pose-kd/scripts/convert_teacher_labels.py`

- [ ] **Step 1: Write the conversion script**

```python
#!/usr/bin/env python3
"""Convert MogaNet-B teacher coords from HDF5 to YOLO label format.

Reads teacher_coords.h5 (coords in crop [0,1] space + crop_params)
and writes YOLO pose labels (normalized by original image dimensions).

Usage:
    python3 convert_teacher_labels.py \
        --input data/teacher_coords.h5 \
        --output data/teacher-labels \
        --base /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import json
import numpy as np
from tqdm import tqdm


def convert_entry(
    teacher_xy: np.ndarray,  # (17, 2) in crop [0,1]
    crop_params: np.ndarray,  # (6,) [x1, y1, crop_w, crop_h, img_w, img_h]
) -> tuple[list[float], list[float]]:
    """Convert one teacher entry to YOLO bbox + keypoints.

    Returns:
        bbox: [cx, cy, w, h] normalized by original image
        kps: [x1, y1, v1, x2, y2, v2, ...] 17 keypoints, visibility 0 or 1
    """
    x1, y1, cw, ch, iw, ih = crop_params

    # Crop [0,1] -> original pixel
    orig_x = teacher_xy[:, 0] * cw + x1
    orig_y = teacher_xy[:, 1] * ch + y1

    # Original pixel -> YOLO [0,1] normalized
    yolo_x = orig_x / iw
    yolo_y = orig_y / ih

    # Clamp to [0, 1]
    yolo_x = np.clip(yolo_x, 0.0, 1.0)
    yolo_y = np.clip(yolo_y, 0.0, 1.0)

    # Bbox from crop_params (the crop IS the person detection bbox)
    cx = (x1 + cw / 2) / iw
    cy = (y1 + ch / 2) / ih
    w = cw / iw
    h = ch / ih
    bbox = [np.clip(cx, 0.0, 1.0), np.clip(cy, 0.0, 1.0), np.clip(w, 0.0, 1.0), np.clip(h, 0.0, 1.0)]

    # Keypoints: interleave x, y, v
    kps = []
    for i in range(17):
        nan_x = np.isnan(yolo_x[i])
        nan_y = np.isnan(yolo_y[i])
        kps.append(0.0 if nan_x else float(yolo_x[i]))
        kps.append(0.0 if nan_y else float(yolo_y[i]))
        kps.append(0 if (nan_x or nan_y) else 1)

    return bbox, kps


def main():
    parser = argparse.ArgumentParser(description="Convert teacher coords to YOLO labels")
    parser.add_argument("--input", required=True, help="Path to teacher_coords.h5")
    parser.add_argument("--output", required=True, help="Output root for teacher labels")
    parser.add_argument("--base", required=True, help="Base path for image directories")
    args = parser.parse_args()

    h5_path = Path(args.input)
    output_root = Path(args.output)
    base_path = Path(args.base)

    print(f"Loading {h5_path}...")
    with h5py.File(str(h5_path), "r") as f:
        coords = f["coords"][:]  # (N, 17, 2)
        crop_params = f["crop_params"][:]  # (N, 6)
        idx = json.loads(f.attrs["index"])  # {image_path: row}

    total = len(idx)
    print(f"Total entries: {total}")

    # Group entries by image path (multi-person -> multiple lines)
    img_entries = defaultdict(list)
    for img_path, row in idx.items():
        # Convert HDF5 key to relative path from base
        rel_path = img_path
        if rel_path.startswith("experiments/yolo26-pose-kd/data/"):
            rel_path = rel_path[len("experiments/yolo26-pose-kd/data/"):]
        img_entries[rel_path].append(row)

    print(f"Unique images: {len(img_entries)}")
    multi_person = sum(1 for v in img_entries.values() if len(v) > 1)
    print(f"Multi-person images: {multi_person}")

    # Process
    written = 0
    skipped = 0
    nan_entries = 0

    for rel_path, rows in tqdm(img_entries.items(), desc="Converting"):
        lines = []
        for row in rows:
            teacher_xy = coords[row]
            cp = crop_params[row]

            # Skip if crop_params are invalid (all -1)
            if cp[0] < 0:
                skipped += 1
                continue

            # Skip if ALL coords are NaN
            if np.isnan(teacher_xy).all():
                skipped += 1
                continue

            if np.isnan(teacher_xy).any():
                nan_entries += 1

            bbox, kps = convert_entry(teacher_xy, cp)

            # YOLO format: class cx cy w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
            parts = ["0"] + [f"{v:.6f}" for v in bbox] + [f"{v:.6f}" for v in kps]
            lines.append(" ".join(parts))

        if not lines:
            continue

        # Determine split from path
        parts = rel_path.split("/")
        if len(parts) >= 2:
            ds_name = parts[0]  # finefs/train, ap3d-fs/train, etc.
            # Normalize split name: finefs/train -> finefs/train, ap3d-fs/train -> ap3d-fs/train
            # For COCO: coco-10pct/train -> coco-10pct/train
            split_parts = parts[0].split("/")
            if len(split_parts) == 2:
                ds, split = split_parts
            else:
                ds = parts[0]
                split = parts[1] if len(parts) > 1 else "train"
        else:
            continue

        # Output label path
        label_dir = output_root / ds / split / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        img_stem = Path(rel_path).stem
        label_file = label_dir / f"{img_stem}.txt"

        with open(label_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        written += 1

    print(f"\nResults:")
    print(f"  Images written: {written}")
    print(f"  Skipped (invalid): {skipped}")
    print(f"  Entries with NaN coords: {nan_entries}")

    # Create symlinks for images
    print("\nCreating image symlinks...")
    for rel_path in tqdm(img_entries.keys(), desc="Symlinks"):
        parts = rel_path.split("/")
        if len(parts) >= 2:
            split_parts = parts[0].split("/")
            if len(split_parts) == 2:
                ds, split = split_parts
            else:
                continue
        else:
            continue

        src_img = base_path / rel_path
        dst_img = output_root / ds / split / "images"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_file = dst_img / src_img.name

        if not dst_file.exists() and src_img.exists():
            try:
                dst_file.symlink_to(src_img.resolve())
            except FileExistsError:
                pass

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit conversion script**

```bash
git add experiments/yolo26-pose-kd/scripts/convert_teacher_labels.py
git commit -m "feat(kd): add teacher coords to YOLO labels conversion script"
```

---

### Task 2: Write Training Config

**Files:**
- Create: `experiments/yolo26-pose-kd/configs/data_teacher.yaml`

- [ ] **Step 1: Create data config for teacher labels**

```yaml
# Teacher-labeled dataset: MogaNet-B predictions as YOLO training labels
# No custom KD loss needed — standard fine-tuning on teacher keypoints
#
# Data: 270,533 training images with MogaNet-B pseudo-labels
# All 3 datasets (FineFS, AP3D-FS, COCO) have teacher predictions

path: /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher-labels

train:
  - finefs/train/images
  - ap3d-fs/train/images
  - coco-10pct/train/images

val:
  - ap3d-fs/valid/images

kpt_shape: [17, 3]

names:
  0: person
```

- [ ] **Step 2: Commit config**

```bash
git add experiments/yolo26-pose-kd/configs/data_teacher.yaml
git commit -m "feat(kd): add teacher-labeled dataset config"
```

---

### Task 3: Run Conversion on Server

**Files:** None (server operation)

- [ ] **Step 1: Push changes to server**

```bash
git push origin fix/kd-distill-trainer-consolidation
ssh vastai "cd /root/skating-biomechanics-ml && git pull"
```

- [ ] **Step 2: Run conversion script**

```bash
ssh vastai "cd /root/skating-biomechanics-ml/experiments/yolo26-pose-kd && \
  python3 scripts/convert_teacher_labels.py \
    --input data/teacher_coords.h5 \
    --output data/teacher-labels \
    --base /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data"
```

Expected output:
- "Total entries: 291901"
- "Unique images: 274563"
- "Images written: ~274000"
- "Skipped (invalid): ~0"
- "Entries with NaN coords: ~few thousand"

- [ ] **Step 3: Verify output**

```bash
ssh vastai "ls /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher-labels/finefs/train/labels/ | wc -l"
ssh vastai "ls /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher-labels/ap3d-fs/train/labels/ | wc -l"
ssh vastai "head -1 /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher-labels/finefs/train/labels/\$(ls /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher-labels/finefs/train/labels/ | head -1)"
```

Expected:
- FineFS: ~229,169 label files
- AP3D-FS: ~35,705 label files
- Label format: `0 cx cy w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v` (all values in [0,1])

- [ ] **Step 4: Spot-check — verify no negative or >1 values**

```bash
ssh vastai "python3 -c \"
from pathlib import Path
import random
base = Path('/root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher-labels')
files = list(base.glob('*/train/labels/*.txt'))
sample = random.sample(files, min(1000, len(files)))
neg = over = 0
for f in sample:
    vals = f.read_text().split()
    for v in vals[1:]:  # skip class
        fv = float(v)
        if fv < 0: neg += 1
        if fv > 1: over += 1
print(f'Checked {len(sample)} files: negative={neg}, over1={over}')
\""
```

Expected: negative=0, over1=0

---

### Task 4: Launch Fine-Tuning

**Files:** None (server operation)

- [ ] **Step 1: Start YOLO26s-pose fine-tuning on teacher labels**

```bash
ssh vastai "cd /root/skating-biomechanics-ml/experiments/yolo26-pose-kd && \
  yolo train \
    model=yolo26s-pose.yaml \
    data=configs/data_teacher.yaml \
    epochs=100 \
    batch=128 \
    imgsz=384 \
    name=v36-teacher-labels \
    exist_ok=true \
    patience=20 \
    save_period=10"
```

Key differences from v35d:
- Standard `yolo train` — no custom trainer, no KD loss
- `patience=20` — early stop if no improvement for 20 epochs
- `save_period=10` — checkpoint every 10 epochs
- Mosaic enabled (default) — no need to disable for teacher labels
- No HDF5, no coordinate transforms, no epoch tracking

- [ ] **Step 2: Monitor first epoch completes without errors**

```bash
ssh vastai "tail -5 /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/runs/pose/v36-teacher-labels/results.csv"
```

Expected: Normal Ultralytics training output with box_loss, pose_loss, cls_loss, etc.

---

### Task 5: Monitor and Evaluate

**Files:** None (monitoring)

- [ ] **Step 1: Check progress at epoch 10**

```bash
ssh vastai "grep 'Epoch' /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/runs/pose/v36-teacher-labels/results.csv | tail -3"
```

Expected at epoch 10:
- Pose mAP50 > 0.10 (baseline YOLO26s zero-shot is ~0 on skating)
- No loss spikes (no cls_loss 142+ like v35c)
- Smooth loss curves

- [ ] **Step 2: Check progress at epoch 50**

```bash
ssh vastai "grep 'Epoch' /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/runs/pose/v36-teacher-labels/results.csv | tail -3"
```

Expected at epoch 50:
- Pose mAP50 > 0.30 (teacher labels should give significant boost)
- If < 0.15: teacher labels may have quality issues

- [ ] **Step 3: Final evaluation at epoch 100 (or early stop)**

```bash
ssh vastai "cat /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/runs/pose/v36-teacher-labels/results.csv | tail -1"
```

Expected at epoch 100:
- Pose mAP50 > 0.50
- Pose mAP50-95 > 0.15
- If AP > 0.50: proceed to InfoGCN training with this model

---

### Task 6: Export and Compare

**Files:** None (server operation)

- [ ] **Step 1: Export best model to ONNX**

```bash
ssh vastai "cd /root/skating-biomechanics-ml/experiments/yolo26-pose-kd && \
  yolo export \
    model=runs/pose/v36-teacher-labels/weights/best.pt \
    format=onnx \
    half=True \
    imgsz=384"
```

- [ ] **Step 2: Run zero-shot baseline comparison**

```bash
ssh vastai "cd /root/skating-biomechanics-ml/experiments/yolo26-pose-kd && \
  yolo val \
    model=yolo26s-pose.pt \
    data=configs/data_teacher.yaml \
    imgsz=384 \
    name=v36-zeroshot-baseline"
```

- [ ] **Step 3: Compare results**

Compare `v36-teacher-labels` vs `v36-zeroshot-baseline`:
- Teacher-labeled should be significantly better on skating val set
- Record metrics for InfoGCN planning
