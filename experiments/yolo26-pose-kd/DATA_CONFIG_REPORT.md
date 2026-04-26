# Data Configuration Report — Knowledge Distillation Dataset

**Date:** 2026-04-21  
**Project:** KD MogaNet-B → YOLO26-Pose  
**Location:** `/home/michael/Github/skating-biomechanics-ml/experiments/yolo26-pose-kd/`

---

## Summary

✅ **data.yaml created** at `configs/data.yaml`  
✅ **Helper scripts created** for AP3D-FS and COCO integration  
✅ **Verification script** for dataset validation  
✅ **FineFS dataset ready:** 8,904 train / 2,007 val images  

---

## Dataset Configuration

### File Location
`/home/michael/Github/skating-biomechanics-ml/experiments/yolo26-pose-kd/configs/data.yaml`

### Configuration Details

```yaml
# Dataset root (auto-updated to local path)
path: /home/michael/Github/skating-biomechanics-ml/experiments/yolo26-pose-kd/data

# Training images
train:
  - finefs/train/images  # 8,904 images (100% skating)
  # - ap3d-fs/train/images  # 71,375 images (to be added)
  # - coco_10pct/train/images  # ~11,828 images (to be added)

# Validation images
val: finefs/val/images  # 2,007 images (skating-only)

# Keypoint format
kpt_shape: [17, 3]  # H3.6M 17kp: [x, y, visibility]

# Classes
names:
  0: person
```

---

## Current Dataset Status

### ✅ FineFS (Primary)
- **Train:** 8,904 images / 8,904 labels
- **Val:** 2,007 images / 2,007 labels
- **Status:** Ready for training
- **Quality:** Tier 1 (validated excellent, 94.1% visible keypoints)

### ⏳ AP3D-FS (Secondary)
- **Train:** 71,375 images available
- **Status:** NOT INTEGRATED
- **Action required:** Run `python scripts/integrate_ap3d_fs.py`
- **Quality:** Tier 1 (human-verified 3D→2D projections)

### ⏳ COCO 10% (Catastrophic Forgetting Prevention)
- **Train:** ~11,828 images (10% of COCO 2017)
- **Status:** NOT INTEGRATED
- **Action required:** Run `python scripts/extract_coco_subset.py`
- **Purpose:** Prevent catastrophic forgetting during KD training

---

## Data Budget

### Current (FineFS only)
- **Train:** 8,904 images
- **Val:** 2,007 images
- **Total:** 10,911 images

### Target (Full Integration)
- **Train:** ~92,107 images
  - FineFS: 8,904 (9.7%)
  - AP3D-FS: 71,375 (77.5%)
  - COCO 10%: ~11,828 (12.8%)
- **Val:** 2,007 images (skating-only for quality metric)

---

## Helper Scripts

### 1. AP3D-FS Integration
**Location:** `scripts/integrate_ap3d_fs.py`

**Purpose:** Convert AP3D figure skating data to YOLO format

**Usage:**
```bash
cd /home/michael/Github/skating-biomechanics-ml
python experiments/yolo26-pose-kd/scripts/integrate_ap3d_fs.py
```

**Input:**
- `data/datasets/athletepose3d/pose_2d/pose_2d/train_set/` (images)
- `data/datasets/athletepose3d/pose_2d/pose_2d/annotations/train_set.json` (annotations)

**Output:**
- `experiments/yolo26-pose-kd/data/ap3d-fs/train/images/` (symlinks)
- `experiments/yolo26-pose-kd/data/ap3d-fs/train/labels/` (YOLO format)

---

### 2. COCO 10% Extraction
**Location:** `scripts/extract_coco_subset.py`

**Purpose:** Extract 10% random subset from COCO 2017

**Usage:**
```bash
cd /home/michael/Github/skating-biomechanics-ml
python experiments/yolo26-pose-kd/scripts/extract_coco_subset.py \
    /path/to/coco2017/train2017 \
    /path/to/coco2017/annotations \
    --output-dir experiments/yolo26-pose-kd/data/coco_10pct \
    --pct 0.1 \
    --seed 42
```

**Requirements:**
- `pip install pycocotools`
- COCO 2017 train images and annotations

**Output:**
- `experiments/yolo26-pose-kd/data/coco_10pct/train/images/` (~11,828 images)
- `experiments/yolo26-pose-kd/data/coco_10pct/train/labels/` (YOLO format)

---

### 3. Dataset Verification
**Location:** `scripts/verify_dataset.py`

**Purpose:** Verify all dataset paths and counts

**Usage:**
```bash
cd /home/michael/Github/skating-biomechanics-ml
python experiments/yolo26-pose-kd/scripts/verify_dataset.py
```

**Output:**
- Dataset counts (images/labels)
- Integration status
- Next steps

---

## Next Steps

### On Vast.ai Server (RTX 5090)

1. **Upload data.yaml:**
   ```bash
   rsync -avz experiments/yolo26-pose-kd/configs/data.yaml \
       root@<vast-ip>:/root/skating-biomechanics-ml/experiments/yolo26-pose-kd/configs/
   ```

2. **Integrate AP3D-FS:**
   ```bash
   ssh root@<vast-ip>
   cd /root/skating-biomechanics-ml
   python experiments/yolo26-pose-kd/scripts/integrate_ap3d_fs.py
   ```

3. **Extract COCO 10% (if COCO available):**
   ```bash
   python experiments/yolo26-pose-kd/scripts/extract_coco_subset.py \
       /path/to/coco2017/train2017 \
       /path/to/coco2017/annotations
   ```

4. **Verify datasets:**
   ```bash
   python experiments/yolo26-pose-kd/scripts/verify_dataset.py
   ```

5. **Update data.yaml to enable all datasets:**
   ```yaml
   train:
     - finefs/train/images
     - ap3d-fs/train/images  # Uncomment after integration
     - coco_10pct/train/images  # Uncomment after integration
   ```

6. **Start training:**
   ```bash
   yolo train model=yolo26n-pose.pt data=experiments/yolo26-pose-kd/configs/data.yaml \
       epochs=100 batch=32 imgsz=640 device=0
   ```

---

## Validation

Run verification before training:
```bash
python experiments/yolo26-pose-kd/scripts/verify_dataset.py
```

Expected output (after full integration):
```
============================================================
Dataset Verification Report
============================================================

1. FineFS (Primary - Figure Skating)
------------------------------------------------------------
Train images: 8904
Train labels: 8904
Val images: 2007
Val labels: 2007
Status: ✓ OK

2. AP3D-FS (AthletePose3D - Figure Skating)
------------------------------------------------------------
Train images: 71375
Train labels: 71375
Status: ✓ OK

3. COCO 10% (Catastrophic Forgetting Prevention)
------------------------------------------------------------
Train images: 11828
Train labels: 11828
Status: ✓ OK

============================================================
Summary
============================================================
Total train images: 92,107
  - FineFS: 8,904 (9.7%)
  - AP3D-FS: 71,375 (77.5%)
  - COCO 10%: 11,828 (12.8%)
Total val images: 2,007

✓ All datasets integrated! Ready for training.
```

---

## Notes

### COCO Dynamic Mix Strategy
- Start with 10% COCO mix
- Monitor val AP every 5 epochs
- If val AP drops >5% relative → increase COCO mix to 15-20%
- This prevents catastrophic forgetting while maintaining skating domain focus

### H3.6M 17-Keypoint Format
- Compatible with COCO 17kp (same order)
- Format: `[x, y, visibility]` per keypoint
- Visibility: 0=occluded, 1=visible
- 17 keypoints cover full body (head, torso, arms, legs)

### Path Handling
- `data.yaml` uses absolute paths (required by Ultralytics)
- Verification script auto-updates paths to local machine
- On Vast.ai, paths will be `/root/skating-biomechanics-ml/...`

---

## References

- **Plan:** `data/plans/2026-04-18-kd-moganet-yolo26-plan.md`
- **Design Spec:** `data/specs/2026-04-18-kd-moganet-yolo26-design.md`
- **FineFS Conversion:** `experiments/yolo26-pose-kd/FINEFS_CONVERSION_REPORT.md`

---

**Status:** ✅ Data configuration complete. Ready for AP3D-FS and COCO integration on Vast.ai server.
