#!/bin/bash
# Full teacher feature generation on Vast.ai server
# Run this in tmux on the remote server

set -e

echo "==================================="
echo "Teacher Feature Generation - Full Run"
echo "==================================="

# Activate virtual environment
source /root/venv/bin/activate

# Change to script directory
cd /root/experiments/yolo26-pose-kd/scripts

# Data directories
DATA_DIRS="/root/data/datasets/ap3d-fs/train /root/data/datasets/coco-10pct/train"
OUTPUT="/root/data/teacher_features.h5"

echo ""
echo "Configuration:"
echo "  Data dirs: $DATA_DIRS"
echo "  Output: $OUTPUT"
echo "  Batch size: 128"
echo "  Layers: 4,6,8"
echo ""

# Run feature generation
python generate_teacher_features.py \
    --data-dirs $DATA_DIRS \
    --output $OUTPUT \
    --batch-size 128 \
    --extract-layers 4,6,8

echo ""
echo "==================================="
echo "Feature Generation Complete!"
echo "==================================="
echo ""
echo "Output file: $OUTPUT"
echo "Index file: ${OUTPUT}.h5.json"
echo ""
echo "Verifying output..."
h5ls -r $OUTPUT

echo ""
echo "Summary statistics:"
python -c "
import h5py
import json

with h5py.File('$OUTPUT', 'r') as f:
    with open('${OUTPUT}.h5.json', 'r') as idx_file:
        idx = json.load(idx_file)

    print(f'Total images: {len(idx)}')
    print('')
    print('Feature layers:')
    for name in sorted(f.keys()):
        ds = f[name]
        size_gb = ds.nbytes / 1e9
        print(f'  {name}: {ds.shape} = {size_gb:.2f} GB')
    print(f'Total storage: {f.file.filename}: {sum(ds.nbytes for ds in f.values()) / 1e9:.2f} GB')
"

echo ""
echo "==================================="
echo "Ready for DWPose KD training!"
echo "==================================="
