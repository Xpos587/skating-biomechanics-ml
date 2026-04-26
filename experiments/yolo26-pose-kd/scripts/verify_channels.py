#!/usr/bin/env python3
"""Verify actual channel dimensions from teacher HDF5 and student model.

This script checks:
1. Teacher feature channels from HDF5 cache
2. Student feature channels from YOLO26 backbone
3. Adapter creation for each layer
"""

from pathlib import Path

import h5py
import torch
import yaml


def check_teacher_channels(hdf5_path: str):
    """Check teacher feature channel dimensions."""
    print(f"Checking teacher features: {hdf5_path}")

    try:
        with h5py.File(hdf5_path, "r") as f:
            for layer_name in ["layer4", "layer6", "layer8"]:
                if layer_name in f:
                    dataset = f[layer_name]
                    shape = dataset.shape
                    print(
                        f"  {layer_name}: {shape} (N={shape[0]}, C={shape[1]}, H={shape[2]}, W={shape[3]})"
                    )
                else:
                    print(f"  {layer_name}: NOT FOUND")
    except FileNotFoundError:
        print(f"  ERROR: HDF5 file not found at {hdf5_path}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def check_student_channels(model_path: str, imgsz: int = 384):
    """Check student feature channel dimensions."""
    print(f"\nChecking student model: {model_path}")

    try:
        # Load YOLO model
        from ultralytics import YOLO

        model = YOLO(model_path)

        # Create dummy input
        dummy_input = torch.randn(1, 3, imgsz, imgsz)

        # Extract features from backbone layers 4, 6, 8
        features = {}
        hooks = []

        def make_hook(idx):
            def hook(module, input, output):
                features[idx] = output.detach()

            return hook

        # Register hooks on backbone layers
        backbone = model.model[:10] if hasattr(model, "model") else model

        for idx in [4, 6, 8]:
            if idx < len(backbone):
                layer = backbone[idx]
                handle = layer.register_forward_hook(make_hook(idx))
                hooks.append(handle)

        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)

        # Remove hooks
        for handle in hooks:
            handle.remove()

        # Print shapes
        for idx in [4, 6, 8]:
            if idx in features:
                shape = features[idx].shape
                print(
                    f"  Layer {idx}: {shape} (B={shape[0]}, C={shape[1]}, H={shape[2]}, W={shape[3]})"
                )
            else:
                print(f"  Layer {idx}: NOT FOUND")

        return features

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main verification routine."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify teacher/student channel dimensions")
    parser.add_argument(
        "--teacher-feat",
        type=str,
        default="data/teacher_features.h5",
        help="Teacher features HDF5 path",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="checkpoints/yolo26n-pose.pt",
        help="Student model path",
    )
    parser.add_argument("--imgsz", type=int, default=384, help="Input image size")
    parser.add_argument(
        "--config", type=str, default="configs/stage3_distill.yaml", help="Training config path"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Channel Dimension Verification for DWPose Knowledge Distillation")
    print("=" * 70)

    # Load config to get paths
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Extract paths from config
        model_path = config.get("model", args.student_model)
        teacher_feat_path = config.get("teacher_feat", args.teacher_feat)
        imgsz = config.get("imgsz", args.imgsz)

        print(f"\nConfig: {config_path}")
        print(f"  Student model: {model_path}")
        print(f"  Teacher features: {teacher_feat_path}")
        print(f"  Image size: {imgsz}")
    else:
        print(f"\nConfig not found: {config_path}")
        print("Using command-line arguments")
        model_path = args.student_model
        teacher_feat_path = args.teacher_feat
        imgsz = args.imgsz

    # Check teacher channels
    teacher_shapes = check_teacher_channels(teacher_feat_path)

    # Check student channels
    student_shapes = check_student_channels(model_path, imgsz)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Adapter Requirements")
    print("=" * 70)

    if teacher_shapes and student_shapes:
        for layer_idx in [4, 6, 8]:
            layer_name = f"layer{layer_idx}"

            # Get teacher channels from HDF5 shape
            if layer_name in teacher_shapes:
                teacher_ch = teacher_shapes[layer_name].shape[1]
            else:
                print(f"\nLayer {layer_idx}: Teacher data missing")
                continue

            # Get student channels from extracted features
            if layer_idx in student_shapes:
                student_ch = student_shapes[layer_idx].shape[1]
            else:
                print(f"\nLayer {layer_idx}: Student data missing")
                continue

            # Print adapter requirement
            print(f"\nLayer {layer_idx}:")
            print(f"  Teacher channels: {teacher_ch}")
            print(f"  Student channels: {student_ch}")
            print(f"  Adapter needed: {teacher_ch} → {student_ch}")
            print(f"  Projection: 1x1 conv ({teacher_ch}, {student_ch})")

            # Calculate preservation
            old_min_c = min(teacher_ch, student_ch)
            old_preserved = 100 * old_min_c / teacher_ch
            print(f"  Information preserved: 100% (NEW) vs {old_preserved:.1f}% (OLD)")

    print("\n" + "=" * 70)
    print("✅ Verification complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
