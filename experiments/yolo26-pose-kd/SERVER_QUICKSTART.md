# Server Quickstart — Vast.ai RTX 5090

**Mission:** Set up Knowledge Distillation training environment on rented GPU

---

## Step 1: Initial SSH Connection

```bash
ssh root@<vast-ip>
cd /root/skating-biomechanics-ml
```

---

## Step 2: Verify Data Structure

```bash
ls experiments/yolo26-pose-kd/data/finefs/train/images/ | wc -l
# Expected: 8904

ls experiments/yolo26-pose-kd/data/finefs/val/images/ | wc -l
# Expected: 2007
```

---

## Step 3: Integrate AP3D-FS Dataset

```bash
python experiments/yolo26-pose-kd/scripts/integrate_ap3d_fs.py
```

**Expected output:**
```
Loading AP3D annotations from .../train_set.json...
Total images: 71375
Total annotations: ...
Processed 71375 annotations...
Integration complete!
  Processed: 71375 annotations
  Skipped: 0 annotations
Final counts:
  Images: 71375
  Labels: 71375
```

---

## Step 4: Verify All Datasets

```bash
python experiments/yolo26-pose-kd/scripts/verify_dataset.py
```

**Expected output:**
```
Total train images: 80,279
  - FineFS: 8,904 (11.1%)
  - AP3D-FS: 71,375 (88.9%)
Total val images: 2,007
```

---

## Step 5: Enable AP3D-FS in data.yaml

Edit `experiments/yolo26-pose-kd/configs/data.yaml`:

```yaml
train:
  - finefs/train/images
  - ap3d-fs/train/images  # UNCOMMENT THIS LINE
```

---

## Step 6: Calibration Run (5 epochs)

**Purpose:** Get real wall-clock time per epoch

```bash
yolo train \
    model=yolo26n-pose.pt \
    data=experiments/yolo26-pose-kd/configs/data.yaml \
    epochs=5 \
    batch=32 \
    imgsz=640 \
    device=0 \
    project=experiments/yolo26-pose-kd/results \
    name=calibration_run
```

**Record:**
- Total wall time
- Seconds per epoch
- VRAM usage

**Extrapolate:**
```
total_hours = (seconds_per_epoch / 3600) × 100_epochs
```

---

## Step 7: Pre-compute Teacher Heatmaps

**Purpose:** Offline MogaNet-B heatmaps for KD (eliminates 1.5× overhead)

```bash
# Create heatmaps generation script
cat > experiments/yolo26-pose-kd/scripts/generate_teacher_heatmaps.py << 'HEATMAP_SCRIPT'
#!/usr/bin/env python3
"""
Generate offline teacher heatmaps for DistilPose KD.
"""
import h5py
import json
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

# Load MogaNet-B
# TODO: Add MogaNet-B loading code
# TODO: Implement top-down inference with GT bboxes
# TODO: Save heatmaps to HDF5

print("Generating teacher heatmaps...")
print("Output: experiments/yolo26-pose-kd/data/teacher_heatmaps.h5")
HEATMAP_SCRIPT

chmod +x experiments/yolo26-pose-kd/scripts/generate_teacher_heatmaps.py

# Run heatmaps generation
python experiments/yolo26-pose-kd/scripts/generate_teacher_heatmaps.py
```

**Expected time:** ~1.5h on RTX 5090, ~3h on RTX 4090

---

## Step 8: Full KD Training (100 epochs)

```bash
yolo train \
    model=yolo26n-pose.pt \
    data=experiments/yolo26-pose-kd/configs/data.yaml \
    epochs=100 \
    batch=32 \
    imgsz=640 \
    device=0 \
    project=experiments/yolo26-pose-kd/results \
    name=kd_yolo26n_full \
    patience=20 \
    save_period=10
```

**Monitor:**
- GT loss
- KD loss
- Val AP (skating)
- VRAM usage

**Success criteria:**
- YOLO26n AP >= 0.85 on skating val
- If not → train YOLO26s (fallback)

---

## Step 9: Optional COCO Integration

**Purpose:** Prevent catastrophic forgetting

```bash
# If COCO 2017 is available
python experiments/yolo26-pose-kd/scripts/extract_coco_subset.py \
    /path/to/coco2017/train2017 \
    /path/to/coco2017/annotations \
    --output-dir experiments/yolo26-pose-kd/data/coco_10pct \
    --pct 0.1
```

**Then update data.yaml:**
```yaml
train:
  - finefs/train/images
  - ap3d-fs/train/images
  - coco_10pct/train/images  # ADD THIS LINE
```

**Dynamic mix strategy:**
- Start with 10% COCO
- Monitor val AP every 5 epochs
- If AP drops >5% → increase to 15-20%

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
yolo train ... batch=16  # or 8
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# If <90%, increase batch size or workers
yolo train ... batch=64 workers=8
```

### Instance Unstable
```bash
# Smoke test (15 min)
nvidia-smi

# Run short training
yolo train ... epochs=1

# Check temps
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
# If >85°C → destroy instance, rent another
```

---

## File Locations

```
/root/skating-biomechanics-ml/
├── experiments/yolo26-pose-kd/
│   ├── configs/
│   │   └── data.yaml                    # Dataset config
│   ├── data/
│   │   ├── finefs/
│   │   │   ├── train/                   # 8,904 images
│   │   │   └── val/                     # 2,007 images
│   │   ├── ap3d-fs/train/               # 71,375 images (after integration)
│   │   ├── coco_10pct/train/            # ~11,828 images (optional)
│   │   └── teacher_heatmaps.h5          # Pre-computed heatmaps
│   ├── scripts/
│   │   ├── integrate_ap3d_fs.py         # AP3D integration
│   │   ├── extract_coco_subset.py       # COCO extraction
│   │   ├── verify_dataset.py            # Verification
│   │   └── generate_teacher_heatmaps.py # Heatmaps generation
│   └── results/
│       ├── calibration_run/             # 5-epoch calibration
│       └── kd_yolo26n_full/             # Full training results
└── data/datasets/
    └── athletepose3d/
        └── pose_2d/pose_2d/
            ├── train_set/               # AP3D source images
            └── annotations/
                └── train_set.json       # AP3D annotations
```

---

## Time Estimates (After Calibration)

Replace these with actual measurements from calibration run:

| Stage | Epochs | Est. Time |
|-------|--------|-----------|
| Calibration | 5 | TBD |
| Teacher Heatmaps | — | ~1.5h |
| Full KD Training | 100 | TBD |

**Budget tracking:**
- RTX 5090: $0.305/hr
- RTX 4090: $0.295/hr
- Target: <$150 total

---

## Success Criteria

1. ✅ Data integrated (FineFS + AP3D-FS)
2. ✅ Calibration run complete
3. ✅ Teacher heatmaps generated
4. ✅ KD training converges
5. ✅ Student AP >= 0.85 on skating val
6. ✅ Total cost under $150

---

**Next:** Run Step 2 (verify data structure)
