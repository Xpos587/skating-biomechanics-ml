#!/bin/bash
# DWPose Two-Stage KD training with feature + logit distillation
# Run this in tmux on the Vast.ai server

set -e

echo "==================================="
echo "DWPose Two-Stage KD Training"
echo "==================================="

# Activate virtual environment
source /root/venv/bin/activate

# Change to script directory
cd /root/experiments/yolo26-pose-kd/scripts

# Configuration
MODEL="/root/data/models/yolo26n-pose.pt"
DATA="/root/data/datasets/data.yaml"
TEACHER_HM="/root/data/teacher_heatmaps.h5"
TEACHER_FEAT="/root/data/teacher_features.h5"
EPOCHS=210
BATCH=128
NAME="distil_pose_full"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA"
echo "  Teacher heatmaps: $TEACHER_HM"
echo "  Teacher features: $TEACHER_FEAT"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH"
echo "  Name: $NAME"
echo ""

# Verify teacher data exists
if [ ! -f "$TEACHER_HM" ]; then
    echo "ERROR: Teacher heatmaps not found at $TEACHER_HM"
    echo "Please run generate_teacher_heatmaps.py first"
    exit 1
fi

if [ ! -f "$TEACHER_FEAT" ]; then
    echo "ERROR: Teacher features not found at $TEACHER_FEAT"
    echo "Please run generate_teacher_features.py first"
    exit 1
fi

echo "Starting Stage 1 training (feature + logit distillation)..."
echo ""

# Stage 1: Feature + Logit distillation
python distill_trainer.py train \
    --model "$MODEL" \
    --data "$DATA" \
    --teacher-hm "$TEACHER_HM" \
    --teacher-feat "$TEACHER_FEAT" \
    --epochs $EPOCHS \
    --batch $BATCH \
    --alpha 0.00005 \
    --beta 0.1 \
    --feature-layers 4,6,8 \
    --name "$NAME" \
    --device 0

echo ""
echo "==================================="
echo "Stage 1 Training Complete!"
echo "==================================="
echo ""
echo "Best model saved to: runs/detect/${NAME}/weights/best.pt"
echo ""
echo "To run Stage 2 (self-KD), use:"
echo "  python distill_trainer.py train \\"
echo "    --model runs/detect/${NAME}/weights/best.pt \\"
echo "    --data $DATA \\"
echo "    --epochs 42 \\"
echo "    --stage2 \\"
echo "    --batch 64 \\"
echo "    --name ${NAME}_stage2"
echo ""
