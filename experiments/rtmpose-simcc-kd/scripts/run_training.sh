#!/bin/bash
# Training launcher for RTMPose-s distillation.
set -e

# Baseline: pseudo-label only (no KD)
echo "=== Training RTMPose-s baseline (pseudo-labels only) ==="
python scripts/train_rtmpose_kd.py \
    --config configs/rtmpose_s_coco17_skating.py \
    --work-dir work_dirs/rtmpose_s_baseline \
    --amp

# KD: DWPose-style distillation
echo "=== Training RTMPose-s with DWPose KD ==="
python scripts/train_rtmpose_kd.py \
    --config configs/rtmpose_s_kd.py \
    --work-dir work_dirs/rtmpose_s_kd \
    --teacher-simcc data/teacher_simcc.npz \
    --amp
