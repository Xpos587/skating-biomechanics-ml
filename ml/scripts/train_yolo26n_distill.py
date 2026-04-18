#!/usr/bin/env python3
"""
Train YOLO26n with knowledge distillation from MogaNet-B pseudo-labels.

Usage:
    python train_yolo26n_distill.py \
        --data /root/data/datasets/skatingverse_pseudo \
        --output /root/yolo_runs/distill_moganet \
        --epochs 100 \
        --batch 64 \
        --device 0,1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml


def create_yolo_data_config(pseudo_dir: Path, output_path: Path) -> dict:
    """Create YOLO data.yaml config for pseudo-labels.

    Args:
        pseudo_dir: Path to pseudo-labels directory
        output_path: Path to save data.yaml

    Returns:
        Config dictionary
    """
    config = {
        "path": str(pseudo_dir),
        "train": "annotations/train.json",
        "val": "annotations/val.json",
        "nc": 1,  # Number of classes (person only)
        "names": ["person"],
        "kpt_shape": [17, 3],  # COCO format: 17 keypoints, (x, y, conf)
        "flip_idx": [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # Left-right flip
    }

    # Save to file
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config


def create_distill_config(
    data_yaml: Path,
    output_dir: Path,
    epochs: int,
    batch: int,
    devices: str,
    distill_weight: float = 0.3,
) -> dict:
    """Create distillation training config.

    Args:
        data_yaml: Path to data.yaml
        output_dir: Path to output directory
        epochs: Number of epochs
        batch: Batch size
        devices: Device IDs (e.g., "0,1")
        distill_weight: Weight for distillation loss

    Returns:
        Config dictionary
    """
    config = {
        # Model
        "model": "yolo26n-pose.pt",
        # Data
        "data": str(data_yaml),
        # Training
        "epochs": epochs,
        "batch": batch,
        "imgsz": 640,
        "rect": False,
        "cos_lr": True,
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        # Augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.15,
        # Validation (CRITICAL!)
        "val": True,
        "plots": True,
        "save": True,
        "save_period": 5,
        "patience": 15,
        # Hardware
        "device": devices,
        "workers": 8,
        "project": str(output_dir),
        "name": "distill",
        "exist_ok": True,
        # Logging
        "verbose": True,
        "seed": 42,
        "deterministic": False,
        # Distillation (custom - will be handled in training loop)
        "distill": {
            "enabled": True,
            "weight": distill_weight,
            "teacher": "moganet_b_ap2d_384x288.pth",
            "temperature": 1.0,
        },
    }

    return config


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26n with knowledge distillation")
    parser.add_argument("--data", type=Path, required=True, help="Path to pseudo-labels directory")
    parser.add_argument("--output", type=Path, required=True, help="Path to output directory")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n-pose.pt",
        help="YOLO model to train (default: yolo26n-pose.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--device", type=str, default="0", help="Device to use (default: 0)")
    parser.add_argument(
        "--distill-weight", type=float, default=0.3, help="Distillation loss weight (default: 0.3)"
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create YOLO data config
    data_yaml_path = args.output / "data.yaml"
    data_config = create_yolo_data_config(args.data, data_yaml_path)
    print(f"Created data config: {data_yaml_path}")

    # Create distillation config
    distill_config_path = args.output / "distill_config.yaml"
    distill_config = create_distill_config(
        data_yaml_path, args.output, args.epochs, args.batch, args.device, args.distill_weight
    )

    # Save config
    with open(distill_config_path, "w") as f:
        yaml.dump(distill_config, f, default_flow_style=False)

    print(f"Created distill config: {distill_config_path}")

    # Print training command
    print("\n" + "=" * 60)
    print("TRAINING COMMAND:")
    print("=" * 60)
    cmd = f"yolo detect train data={data_yaml_path} model={args.model} epochs={args.epochs} batch={args.batch} device={args.device} project={args.output} name=distill val=True save_period=5 patience=15"
    print(cmd)
    print("=" * 60)

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "args": vars(args),
        "data_config": str(data_yaml_path),
        "distill_config": str(distill_config_path),
        "training_command": cmd,
    }

    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {args.output / 'metadata.json'}")
    print("\nReady to train!")


if __name__ == "__main__":
    main()
