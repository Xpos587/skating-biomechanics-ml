#!/bin/bash
# Test script for teacher feature generation on Vast.ai server
# Run this in tmux on the remote server

set -e

echo "==================================="
echo "Testing Teacher Feature Generation"
echo "==================================="

# Activate virtual environment
source /root/venv/bin/activate

# Change to script directory
cd /root/experiments/yolo26-pose-kd/scripts

# Test on 10 images from ap3d-fs
echo ""
echo "Running test on 10 images..."
python generate_teacher_features.py \
    --data-dirs /root/data/datasets/ap3d-fs/train \
    --output /root/data/teacher_features_test.h5 \
    --batch-size 4 \
    --test

echo ""
echo "Test complete. Checking output..."
h5ls -r /root/data/teacher_features_test.h5

echo ""
echo "Verifying feature shapes..."
python -c "
import h5py
import torch

with h5py.File('/root/data/teacher_features_test.h5', 'r') as f:
    print('Datasets:', list(f.keys()))
    for name in f.keys():
        ds = f[name]
        print(f'{name}: shape={ds.shape}, dtype={ds.dtype}')
        if ds.shape[0] > 0:
            sample = torch.from_numpy(ds[0:1])
            print(f'  Sample range: [{sample.min():.3f}, {sample.max():.3f}]')
"

echo ""
echo "==================================="
echo "Test completed successfully!"
echo "==================================="
