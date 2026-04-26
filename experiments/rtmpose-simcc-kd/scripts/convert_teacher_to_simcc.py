#!/usr/bin/env python3
"""Convert teacher coords HDF5 → SimCC soft labels NPZ.

Usage:
    python convert_teacher_to_simcc.py \
        --input /path/to/teacher_coords.h5 \
        --output /path/to/teacher_simcc.npz \
        --num-x-bins 192 --num-y-bins 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from simcc_label_generator import batch_generate_simcc, transform_teacher_coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input teacher_coords.h5")
    parser.add_argument("--output", required=True, help="Output teacher_simcc.npz")
    parser.add_argument("--num-x-bins", type=int, default=192)
    parser.add_argument("--num-y-bins", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with h5py.File(args.input, "r") as f:
        coords = f["coords"][:]
        confidence = f["confidence"][:]
        crop_params = f["crop_params"][:]
        index = json.loads(f.attrs["index"])

    N = len(coords)
    print(f"Total entries: {N}")

    simcc_x_all = np.zeros((N, 17, args.num_x_bins), dtype=np.float16)
    simcc_y_all = np.zeros((N, 17, args.num_y_bins), dtype=np.float16)

    for start in tqdm(range(0, N, args.batch_size), desc="Converting"):
        end = min(start + args.batch_size, N)
        batch_coords = coords[start:end]
        batch_conf = confidence[start:end]
        batch_cp = crop_params[start:end]

        img_coords = np.zeros_like(batch_coords)
        for i in range(len(batch_coords)):
            img_coords[i] = transform_teacher_coords(batch_coords[i], batch_cp[i])

        sx, sy = batch_generate_simcc(img_coords, batch_conf, args.num_x_bins, args.num_y_bins)
        simcc_x_all[start:end] = sx.astype(np.float16)
        simcc_y_all[start:end] = sy.astype(np.float16)

    np.savez_compressed(
        args.output,
        simcc_x=simcc_x_all,
        simcc_y=simcc_y_all,
        confidence=confidence.astype(np.float16),
        index=json.dumps(index),
    )
    print(f"Saved to {args.output}")
    print(f"  simcc_x: {simcc_x_all.shape}, dtype={simcc_x_all.dtype}")
    print(f"  simcc_y: {simcc_y_all.shape}, dtype={simcc_y_all.dtype}")


if __name__ == "__main__":
    main()
