#!/usr/bin/env python3
"""Extract teacher keypoint coordinates from HDF5 heatmaps via soft argmax.

Outputs HDF5 with:
  - coords: (N, 17, 2) float32 — teacher keypoint (x, y) in normalized [0,1] crop space
  - confidence: (N, 17) float32 — max heatmap value per keypoint
  - crop_params: (N, 6) float32 — (x1, y1, crop_w, crop_h, img_w, img_h) in original image pixels

Usage:
    python3 extract_teacher_coords.py \
        --heatmaps data/teacher_heatmaps.h5 \
        --output data/teacher_coords.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def soft_argmax_heatmap(
    heatmap: np.ndarray, temperature: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sub-pixel coordinates via temperature-scaled spatial softmax.

    Args:
        heatmap: (K, H, W) float16 heatmap.
        temperature: softmax temperature. Higher = sharper peak.

    Returns:
        coords: (K, 2) float32 — (x, y) in [0, 1].
        confidence: (K,) float32 — max value per keypoint.
    """
    K, H, W = heatmap.shape
    hm = torch.from_numpy(heatmap.astype(np.float32))  # (K, H, W)

    flat = hm.reshape(K, -1)  # (K, H*W)
    probs = F.softmax(flat / temperature, dim=-1)  # (K, H*W)
    probs = probs.reshape(K, H, W)

    gx = torch.linspace(0, 1, W, device=hm.device)
    gy = torch.linspace(0, 1, H, device=hm.device)
    grid_x, grid_y = torch.meshgrid(gx, gy, indexing="xy")

    x_coords = (probs * grid_x.unsqueeze(0)).sum(dim=[1, 2])  # (K,)
    y_coords = (probs * grid_y.unsqueeze(0)).sum(dim=[1, 2])  # (K,)

    coords = torch.stack([x_coords, y_coords], dim=-1).numpy()  # (K, 2)
    confidence = hm.flatten(1).max(dim=1).values.numpy()  # (K,)

    return coords.astype(np.float32), confidence.astype(np.float32)


def soft_argmax_heatmap_batch(
    hm_batch: np.ndarray, temperature: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized soft argmax for batch of heatmaps.

    Uses temperature-scaled softmax so the peak dominates over 6912 spatial bins.
    temperature=0.01 sharpens the distribution so the Gaussian peak gets most
    probability mass (sub-pixel accuracy). For nearly-uniform heatmaps, returns
    the center of mass.

    Args:
        hm_batch: (B, K, H, W) float16 heatmap batch.
        temperature: softmax temperature. Higher = sharper peak.

    Returns:
        x_coords: (B, K) float32 — x coordinates in [0, 1].
        y_coords: (B, K) float32 — y coordinates in [0, 1].
        confidence: (B, K) float32 — max value per keypoint.
    """
    hm = torch.from_numpy(hm_batch.astype(np.float32))  # (B, K, H, W)
    B, K, H, W = hm.shape

    flat = hm.reshape(B, K, H * W)  # (B, K, H*W)
    probs = F.softmax(flat / temperature, dim=-1).reshape(B, K, H, W)

    gx = torch.linspace(0, 1, W, device=hm.device)
    gy = torch.linspace(0, 1, H, device=hm.device)
    grid_x, grid_y = torch.meshgrid(gx, gy, indexing="xy")

    x_coords = (probs * grid_x.view(1, 1, H, W)).sum(dim=[2, 3])
    y_coords = (probs * grid_y.view(1, 1, H, W)).sum(dim=[2, 3])

    confidence = hm.flatten(2).max(dim=2).values

    return (
        x_coords.numpy().astype(np.float32),
        y_coords.numpy().astype(np.float32),
        confidence.numpy().astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="Extract teacher coords from HDF5 heatmaps")
    parser.add_argument("--heatmaps", required=True, help="Path to teacher_heatmaps.h5")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--batch-size", type=int, default=4096, help="Processing batch size")
    args = parser.parse_args()

    h5_path = Path(args.heatmaps)
    json_path = h5_path.with_suffix(".h5.json")

    # Load index and crop_params from JSON sidecar
    print(f"Loading index from {json_path}")
    with json_path.open() as f:
        sidecar = json.load(f)
    idx_map = sidecar.get("index", {})
    crop_params_map = sidecar.get("crop_params", {})
    print(f"  Index entries: {len(idx_map)}")
    print(f"  Crop params: {len(crop_params_map)}")

    # Open HDF5
    print(f"Opening HDF5: {h5_path}")
    with h5py.File(str(h5_path), "r") as hf:
        dataset = hf["heatmaps"]
        total = dataset.shape[0]
        K, H, W = dataset.shape[1], dataset.shape[2], dataset.shape[3]
        print(f"  Total heatmaps: {total}, shape: ({K}, {H}, {W})")

        # Preallocate output arrays
        all_coords = np.zeros((total, K, 2), dtype=np.float32)
        all_confidence = np.zeros((total, K), dtype=np.float32)
        all_crop_params = np.full((total, 6), -1.0, dtype=np.float32)  # -1 = unknown

        # Build reverse index: row_idx -> image_path
        rev_idx = {v: k for k, v in idx_map.items()}

        # Process in batches
        for start in tqdm(range(0, total, args.batch_size), desc="Extracting coords"):
            end = min(start + args.batch_size, total)
            batch_hm = dataset[start:end]  # (B, K, H, W)

            # Vectorized soft argmax for entire batch
            batch_np = np.asarray(batch_hm)
            cx, cy, conf = soft_argmax_heatmap_batch(batch_np)
            batch_size = end - start

            # Stack coords into (B, K, 2)
            all_coords[start:end] = np.stack([cx, cy], axis=-1)
            all_confidence[start:end] = conf

            # Get crop params from sidecar (still per-image since it's JSON lookup)
            for i in range(batch_size):
                row = start + i
                img_path = rev_idx.get(row)
                if img_path and img_path in crop_params_map:
                    cp = crop_params_map[img_path]
                    all_crop_params[row] = [
                        cp["x1"],
                        cp["y1"],
                        cp["crop_w"],
                        cp["crop_h"],
                        cp["img_w"],
                        cp["img_h"],
                    ]

    valid_cp = np.sum(np.all(all_crop_params >= 0, axis=1))
    print("\nResults:")
    print(f"  Total: {total}")
    print(f"  Valid crop params: {valid_cp}/{total}")
    print(
        f"  Coord range: x=[{all_coords[:, :, 0].min():.4f}, {all_coords[:, :, 0].max():.4f}] "
        f"y=[{all_coords[:, :, 1].min():.4f}, {all_coords[:, :, 1].max():.4f}]"
    )
    print(f"  Confidence range: [{all_confidence.min():.4f}, {all_confidence.max():.4f}]")

    # Save to HDF5
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as hf:
        hf.create_dataset("coords", data=all_coords, compression="gzip", compression_opts=4)
        hf.create_dataset("confidence", data=all_confidence, compression="gzip", compression_opts=4)
        hf.create_dataset(
            "crop_params", data=all_crop_params, compression="gzip", compression_opts=4
        )
        hf.attrs["index"] = json.dumps(idx_map)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to: {output_path} ({size_mb:.1f} MB)")

    # Verify
    print("\nVerification:")
    with h5py.File(str(output_path), "r") as hf:
        print(f"  coords: {hf['coords'].shape} {hf['coords'].dtype}")
        print(f"  confidence: {hf['confidence'].shape} {hf['confidence'].dtype}")
        print(f"  crop_params: {hf['crop_params'].shape} {hf['crop_params'].dtype}")
        idx = json.loads(hf.attrs["index"])
        print(f"  index entries: {len(idx)}")


if __name__ == "__main__":
    main()
