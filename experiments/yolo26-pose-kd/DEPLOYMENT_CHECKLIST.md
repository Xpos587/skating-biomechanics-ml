# DWPose Feature Distillation - Deployment Checklist

**Date:** 2026-04-22
**Target:** Vast.ai GPU Server (RTX 5090)

---

## Pre-Deployment Checks

### ✅ Code Readiness

- [x] `generate_teacher_features.py` — Feature extraction script
- [x] `distill_trainer.py` — Updated with feature distillation
- [x] `test_feature_generation.sh` — Test script
- [x] `run_feature_generation.sh` — Full generation pipeline
- [x] `run_distill_training.sh` — Full training pipeline
- [x] `validate_feature_distillation.py` — Validation tests
- [x] `FEATURE_DISTILLATION_REPORT.md` — Detailed documentation
- [x] `IMPLEMENTATION_SUMMARY.md` — Implementation summary

### ✅ Validation Tests

- [x] TeacherFeatureLoader HDF5 reading — PASS
- [x] MSE Loss Computation — PASS
- [x] Integration with distill_trainer — PASS
- [ ] FeatureExtractorMogaNet Hooks — Requires timm (will test on server)

---

## Deployment Steps

### Step 1: Connect to Vast.ai Server

```bash
# Find server IP
vast show instances

# SSH into server
ssh root@<server-ip>

# Activate virtual environment
source /root/venv/bin/activate

# Change to working directory
cd /root/experiments/yolo26-pose-kd/scripts
```

**Expected:**
- Server is running
- Virtual environment exists
- Dependencies installed (timm, h5py, torch, etc.)

---

### Step 2: Copy Scripts to Server

```bash
# From local machine
scp experiments/yolo26-pose-kd/scripts/*.py \
    experiments/yolo26-pose-kd/scripts/*.sh \
    root@<server-ip>:/root/experiments/yolo26-pose-kd/scripts/

# Or use rsync
rsync -av experiments/yolo26-pose-kd/scripts/ \
    root@<server-ip>:/root/experiments/yolo26-pose-kd/scripts/
```

**Expected:**
- All scripts copied successfully
- Scripts are executable (`chmod +x *.sh`)

---

### Step 3: Verify Dependencies

```bash
# On server
python -c "
import torch
import h5py
import timm
import numpy as np
from PIL import Image
print('✓ All dependencies available')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
"
```

**Expected:**
- All imports succeed
- CUDA is available
- GPU detected (RTX 5090 or similar)

---

### Step 4: Test Feature Generation (10 Images)

```bash
# Start tmux session
tmux new -s feature_gen

# Run test
bash test_feature_generation.sh

# Detach: Ctrl+B, D
# Reattach: tmux attach -t feature_gen
```

**Expected Output:**
```
Running test forward pass to determine feature shapes...
Feature shapes:
  Layer 4: (1, 64, 72, 96) (C=64, H=72, W=96)
  Layer 6: (1, 160, 36, 48) (C=160, H=36, W=48)
  Layer 8: (1, 160, 36, 48) (C=160, H=36, W=48)

Writing features to /root/data/teacher_features_test.h5...
Done. 10 feature sets written to teacher_features_test.h5
Throughput: ~120 images/sec
Total storage: 0.01 GB
```

**Verification:**
```bash
h5ls -r /root/data/teacher_features_test.h5
```

**Expected:**
```
/                        Group
/layer4                  Dataset {10, 64, 72, 96}
/layer6                  Dataset {10, 160, 36, 48}
/layer8                  Dataset {10, 160, 36, 48}
```

---

### Step 5: Run Full Feature Generation

```bash
# Start tmux session
tmux new -s feature_gen_full

# Run full generation
bash run_feature_generation.sh

# Detach: Ctrl+B, D
# Monitor: tmux attach -t feature_gen_full
```

**Expected Progress:**
```
Found 15234 images with labels across 2 data dirs
  ap3d-fs: 5234 images
  coco-10pct: 10000 images

Loading MogaNet-B from /root/data/datasets/raw/athletepose3d/model_params/moganet_b_ap2d_384x288.pth...
Model on cuda
Parameters: 32,456,789 (32.5M)

Writing features to /root/data/teacher_features.h5...
Generating features: 100%|████████| 15234/15234 [02:10<00:00, 117.52img/s, feat=118/s, total=15234]

Done. 15234 feature sets written to /root/data/teacher_features.h5
Throughput: 118 images/sec
Total time: 129.0s

  Layer 4: 15234 x (64, 72, 96) = 0.67 GB
  Layer 6: 15234 x (160, 36, 48) = 1.78 GB
  Layer 8: 15234 x (160, 36, 48) = 1.78 GB
Total storage: 4.23 GB
Storage dtype: float16
```

**Verification:**
```bash
h5ls -r /root/data/teacher_features.h5
python -c "
import h5py
with h5py.File('/root/data/teacher_features.h5', 'r') as f:
    for name in sorted(f.keys()):
        ds = f[name]
        print(f'{name}: {ds.shape}, {ds.dtype}')
"
```

---

### Step 6: Verify Teacher Heatmaps Exist

```bash
# Check that teacher heatmaps are already generated
ls -lh /root/data/teacher_heatmaps.h5

# Verify structure
h5ls -r /root/data/teacher_heatmaps.h5
```

**Expected:**
```
/                        Group
/heatmaps                Dataset {15234, 17, 72, 96}
```

**If missing, generate:**
```bash
python generate_teacher_heatmaps.py \
    --data-dirs /root/data/datasets/ap3d-fs/train /root/data/datasets/coco-10pct/train \
    --output /root/data/teacher_heatmaps.h5 \
    --batch-size 128
```

---

### Step 7: Start DWPose Training

```bash
# Start tmux session
tmux new -s distill_training

# Run training
bash run_distill_training.sh

# Detach: Ctrl+B, D
# Monitor: tmux attach -t distill_training
```

**Expected Output:**
```
Configuration:
  Model: /root/data/models/yolo26n-pose.pt
  Data: /root/data/datasets/data.yaml
  Teacher heatmaps: /root/data/teacher_heatmaps.h5
  Teacher features: /root/data/teacher_features.h5
  Epochs: 210
  Batch size: 128
  Name: distil_pose_full

Starting Stage 1 training (feature + logit distillation)...

Ultralytics YOLOv8.0.0 🚀 Python-3.10.0 torch-2.0.0 CUDA:0 (NVIDIA RTX 5090, 24564MiB)
yolo: pose=detect.yaml, pretrained=yolo26n-pose.pt, epochs=210, batch=128

          Epoch    GPU_mem   box_loss   pose_loss   kobj_loss   cls_loss   dfl_loss  kd_logit_loss  kd_feat_loss  kd_weight       Instances       Size
      1/210        12.4G      2.345      1.234      0.567      0.123      0.456        0.123        0.001      1.000         1280          640
      2/210        12.4G      2.123      1.123      0.523      0.112      0.434        0.112        0.001      0.995         1280          640
     ...
    105/210        12.4G      1.234      0.567      0.234      0.056      0.234        0.056        0.000      0.505         1280          640
     ...
    210/210        12.4G      0.987      0.456      0.123      0.034      0.123        0.034        0.000      0.005         1280          640

✅ Stage 1 Training Complete!
Best model saved to: runs/detect/distil_pose_full/weights/best.pt
```

**Monitoring Tips:**
- `box_loss`, `pose_loss`, etc. should decrease (GT loss)
- `kd_logit_loss` should decrease (heatmap alignment)
- `kd_feat_loss` should decrease (feature alignment)
- `kd_weight` should decay from 1.0 to ~0.0
- GPU memory should be stable (~12-14 GB)

---

### Step 8: Evaluate Results

```bash
# After training completes
python distill_trainer.py \
    --model runs/detect/distil_pose_full/weights/best.pt \
    --data /root/data/datasets/data.yaml \
    --epochs 1 \
    --batch 64 \
    --val
```

**Expected Metrics:**
- Compare against baseline (no KD)
- Compare against logit-only KD
- Target: +2-5% AP improvement (per DWPose paper)

---

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'timm'

**Solution:**
```bash
pip install timm==0.9.12
```

### Issue: CUDA out of memory

**Solution:**
```bash
# Reduce batch size
python generate_teacher_features.py --batch-size 64

# Or reduce image size in training
--imgsz 512  # instead of 640
```

### Issue: Slow feature loading

**Solution:**
```bash
# Move HDF5 to tmpfs (RAM disk)
mkdir /tmp/hdf5_cache
mount -t tmpfs -o size=32G tmpfs /tmp/hdf5_cache
cp /root/data/teacher_features.h5 /tmp/hdf5_cache/

# Update training script to use cached path
--teacher-feat /tmp/hdf5_cache/teacher_features.h5
```

### Issue: Training loss is NaN

**Solution:**
1. Check learning rate (should be 0.002 for AdamW)
2. Check feature values (should be in reasonable range)
3. Check gradient clipping (add if needed)
4. Reduce alpha/beta weights

---

## Post-Deployment

### Monitoring

**Check GPU utilization:**
```bash
nvidia-smi -l 1  # Update every 1 second
```

**Check training progress:**
```bash
tail -f runs/detect/distil_pose_full/train.log
```

**Check HDF5 files:**
```bash
du -sh /root/data/teacher_*.h5
```

### Backup

**Copy trained model to local machine:**
```bash
scp root@<server-ip>:/root/experiments/yolo26-pose-kd/runs/detect/distil_pose_full/weights/best.pt \
    experiments/yolo26-pose-kd/checkpoints/
```

**Copy HDF5 files to local machine:**
```bash
scp root@<server-ip>:/root/data/teacher_*.h5* \
    experiments/yolo26-pose-kd/data/
```

---

## Success Criteria

✅ **Feature Generation:**
- [ ] Test on 10 images succeeds
- [ ] Shapes match expected dimensions
- [ ] Full generation completes in <5 minutes
- [ ] HDF5 files verified with h5ls

✅ **Training:**
- [ ] Stage 1 training starts without errors
- [ ] All loss components are logged
- [ ] Losses decrease over epochs
- [ ] Training completes in 2-3 days

✅ **Evaluation:**
- [ ] Validation AP measured
- [ ] Improvement over baseline observed
- [ ] Model checkpoint saved

---

## Next Steps After Deployment

1. **Run Stage 2 self-KD** (optional, +0.1% AP expected)
2. **Ablation studies:**
   - Feature-only vs Logit-only vs Both
   - Different layer combinations
   - Different α, β weights
3. **Production deployment:**
   - Export to ONNX
   - Integrate with main pipeline
   - Benchmark inference speed

---

**Generated:** 2026-04-22
**Status:** Ready for deployment
**Estimated Time to Complete:** 3-4 days (including training)
