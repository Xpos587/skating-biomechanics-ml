#!/bin/bash
# DWPose KD Training - Stage 1
# Run this in tmux on the remote server

set -e

echo "==================================="
echo "DWPose KD Training - Stage 1"
echo "==================================="

# Activate virtual environment
source /root/venv/bin/activate

# Change to project directory
cd /root/skating-biomechanics-ml/experiments/yolo26-pose-kd

echo ""
echo "Configuration:"
echo "  Model: yolo26s-pose.pt (Stage 1 student)"
echo "  Teacher: MogaNet-B (pre-computed heatmaps + features)"
echo "  Epochs: 210"
echo "  Batch: 128"
echo "  Image size: 384"
echo "  Mosaic: 0.0 (DWPose Stage 1 OFF)"
echo "  KD weight decay: 1.0 -> 0.0 over training"
echo "  Alpha (feat): 0.00005"
echo "  Beta (logit): 0.1"
echo ""

# Verify teacher data exists
echo "Checking teacher data..."
if [ ! -f "/root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher_heatmaps.h5" ]; then
    echo "ERROR: teacher_heatmaps.h5 not found!"
    exit 1
fi

if [ ! -f "/root/data/teacher_features.h5" ]; then
    echo "WARNING: teacher_features.h5 not found!"
    echo "Training will proceed without feature distillation (logit-only KD)."
    TEACHER_FEAT=""
else
    echo "✓ teacher_features.h5 found"
    TEACHER_FEAT="--teacher-feat /root/data/teacher_features.h5"
fi

echo ""
echo "Starting training..."
echo "Log directory: runs/train/stage1_distill/"
echo ""

# Launch training with DWPose Stage 1 config
python scripts/distill_trainer.py train \
    --model checkpoints/yolo26s-pose.pt \
    --data configs/data.yaml \
    --teacher-hm data/teacher_heatmaps.h5 \
    $TEACHER_FEAT \
    --epochs 210 \
    --batch 128 \
    --imgsz 384 \
    --alpha 0.00005 \
    --beta 0.1 \
    --warmup-epochs 5 \
    --freeze-backbone \
    --feature-layers 4,6,8 \
    --name stage1_distill \
    --device 0

echo ""
echo "==================================="
echo "Training Complete!"
echo "==================================="
echo ""
echo "Best model saved to: runs/train/stage1_distill/weights/best.pt"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python scripts/evaluate_kd.py --model runs/train/stage1_distill/weights/best.pt"
echo "  2. Stage 2 (optional): bash scripts/run_stage2.sh"
echo ""
