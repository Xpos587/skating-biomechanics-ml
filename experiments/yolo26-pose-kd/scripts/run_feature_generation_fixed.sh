#!/bin/bash
# Generate teacher features for DWPose Stage 1 KD
# Run on Vast.ai server in tmux session: feature-cache

set -e

echo "========================================="
echo "Teacher Feature Generation"
echo "Started: $(date)"
echo "========================================="

cd /root/skating-biomechanics-ml/experiments/yolo26-pose-kd

# Create logs directory
mkdir -p logs

# Configuration
TEACHER_MODEL="/root/data/datasets/raw/athletepose3d/model_params/moganet_b_ap2d_384x288.pth"
DATA_DIRS="data/ap3d-fs/train data/finefs/train"
OUTPUT="data/teacher_features.h5"
BATCH_SIZE=128

echo "Configuration:"
echo "  Teacher model: $TEACHER_MODEL"
echo "  Data dirs: $DATA_DIRS"
echo "  Output: $OUTPUT"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Check if teacher model exists
if [ ! -f "$TEACHER_MODEL" ]; then
    echo "❌ Teacher model not found: $TEACHER_MODEL"
    echo "Looking for MogaNet-B weights..."
    find /root -name '*moganet*b_ap2d*.pth' 2>/dev/null || echo "Not found"
    exit 1
fi

echo "✅ Teacher model found"

# Check if output already exists
if [ -f "$OUTPUT" ]; then
    echo "⚠️  Output already exists: $OUTPUT"
    backup_name="${OUTPUT}.$(date +%Y%m%d_%H%M%S).bak"
    echo "Backing up to: $backup_name"
    mv "$OUTPUT" "$backup_name"
fi

# Run feature generation
echo "🚀 Generating teacher features..."
echo "Dataset: FineFS (229,169) + AP3D-FS (35,705) = 264,874 images"
echo "Layers: [4, 6, 8] (end-of-stage features)"
echo "ETA: ~2-3 hours"
echo ""

python scripts/generate_teacher_features.py \
    --model "$TEACHER_MODEL" \
    --data-dirs $DATA_DIRS \
    --output "$OUTPUT" \
    --batch-size $BATCH_SIZE \
    2>&1 | tee logs/feature_generation.log

# Check if output was created
if [ -f "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo ""
    echo "✅ Feature generation complete!"
    echo "Output: $OUTPUT ($SIZE)"
    echo "Completed: $(date)"

    # Verify output
    echo ""
    echo "Verifying output..."
    python -c "
import h5py
f = h5py.File('$OUTPUT', 'r')
print('Shape:', f['features'].shape)
print('Dtype:', f['features'].dtype)
print('Layers:', list(f.keys()))
f.close()
"
else
    echo "❌ Feature generation failed - output not found"
    exit 1
fi

echo ""
echo "========================================="
echo "Next: Run Stage 1 training with feature + logit distillation"
echo "Command: bash scripts/run_stage1_distill.sh"
echo "========================================="
