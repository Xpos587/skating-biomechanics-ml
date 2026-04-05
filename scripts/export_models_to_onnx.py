#!/usr/bin/env python3
"""Convert PyTorch pose estimation models to ONNX format.

Usage:
    uv run python scripts/export_models_to_onnx.py --model motionagformer-s
    uv run python scripts/export_models_to_onnx.py --model tcpformer
    uv run python scripts/export_models_to_onnx.py --all
"""

import argparse
import sys
from pathlib import Path

import torch


def _strip_prefix(state_dict: dict) -> dict:
    """Strip 'module.' prefix from DataParallel-wrapped checkpoints."""
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_state_dict(checkpoint_path: str) -> dict:
    """Load checkpoint and extract state dict, handling various formats."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    return _strip_prefix(state_dict)


def export_motionagformer_s(checkpoint_path: str, output_path: str) -> None:
    """Export MotionAGFormer-S to ONNX."""
    # Add project root to path so model imports work
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from models.motionagformer.MotionAGFormer import MotionAGFormer

    # Checkpoint has 26 MotionAGFormerBlock layers (indices 0-25)
    model = MotionAGFormer(
        n_layers=26,
        dim_in=3,
        dim_feat=64,
        dim_rep=512,
        dim_out=3,
        mlp_ratio=4,
        num_heads=4,
        num_joints=17,
        n_frames=81,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        use_adaptive_fusion=True,
        hierarchical=False,
        use_temporal_similarity=True,
        temporal_connection_len=1,
        use_tcn=False,
        graph_only=False,
        neighbour_num=4,
    )

    state_dict = _load_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(
            f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    model.eval()

    # Dummy input: [1, 81, 17, 3] (batch, frames, joints, channels)
    dummy_input = torch.randn(1, 81, 17, 3, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=18,
        input_names=["poses_2d"],
        output_names=["poses_3d"],
        dynamic_axes={
            "poses_2d": {0: "batch"},
            "poses_3d": {0: "batch"},
        },
        do_constant_folding=True,
    )
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Exported MotionAGFormer-S -> {output_path} ({file_size_mb:.1f} MB)")

    # Verify with ONNX Runtime
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    result = sess.run(None, {"poses_2d": dummy_input.numpy()})
    print(f"  Verification: output shape = {result[0].shape}, dtype = {result[0].dtype}")
    expected_shape = (1, 81, 17, 3)
    assert result[0].shape == expected_shape, (
        f"Shape mismatch: {result[0].shape} != {expected_shape}"
    )
    print("  ONNX verification PASSED")


def export_tcpformer(checkpoint_path: str, output_path: str) -> None:
    """Export TCPFormer (MemoryInducedTransformer) to ONNX."""
    # Add project root to path so model imports work
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from models.tcpformer.TCPFormer import MemoryInducedTransformer

    model = MemoryInducedTransformer(
        n_layers=16,
        dim_in=3,
        dim_feat=128,
        dim_rep=512,
        dim_out=3,
        mlp_ratio=4,
        num_heads=4,
        num_joints=17,
        n_frames=81,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        use_adaptive_fusion=True,
        hierarchical=False,
        use_temporal_similarity=True,
        temporal_connection_len=1,
        use_tcn=False,
        graph_only=False,
        neighbour_num=4,
    )

    state_dict = _load_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(
            f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    model.eval()

    dummy_input = torch.randn(1, 81, 17, 3, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=18,
        input_names=["poses_2d"],
        output_names=["poses_3d"],
        dynamic_axes={
            "poses_2d": {0: "batch"},
            "poses_3d": {0: "batch"},
        },
        do_constant_folding=True,
    )
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Exported TCPFormer -> {output_path} ({file_size_mb:.1f} MB)")

    import onnxruntime as ort

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    result = sess.run(None, {"poses_2d": dummy_input.numpy()})
    print(f"  Verification: output shape = {result[0].shape}, dtype = {result[0].dtype}")
    expected_shape = (1, 81, 17, 3)
    assert result[0].shape == expected_shape, (
        f"Shape mismatch: {result[0].shape} != {expected_shape}"
    )
    print("  ONNX verification PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument(
        "--model", choices=["motionagformer-s", "tcpformer"], help="Model to export"
    )
    parser.add_argument("--all", action="store_true", help="Export all models")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.print_help()
        return

    models_dir = Path("data/models")

    if args.all or args.model == "motionagformer-s":
        src = models_dir / "motionagformer-s-ap3d.pth.tr"
        dst = models_dir / "motionagformer-s-ap3d.onnx"
        if src.exists():
            print("\n=== Exporting MotionAGFormer-S ===")
            print(f"  Source: {src}")
            print(f"  Target: {dst}")
            export_motionagformer_s(str(src), str(dst))
        else:
            print(f"Skip: {src} not found")

    if args.all or args.model == "tcpformer":
        src = models_dir / "TCPFormer_ap3d_81.pth.tr"
        dst = models_dir / "TCPFormer_ap3d_81.onnx"
        if src.exists():
            print("\n=== Exporting TCPFormer ===")
            print(f"  Source: {src}")
            print(f"  Target: {dst}")
            export_tcpformer(str(src), str(dst))
        else:
            print(f"Skip: {src} not found")


if __name__ == "__main__":
    main()
