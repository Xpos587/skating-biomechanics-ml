#!/usr/bin/env python3
"""Download ML model weights for optional pipeline features.

Usage:
    uv run python scripts/download_ml_models.py --all
    uv run python scripts/download_ml_models.py --model depth_anything
    uv run python scripts/download_ml_models.py --list
"""

import argparse
from pathlib import Path

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "depth_anything": {
        "url": "https://huggingface.co/DepthAnything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.onnx",
        "filename": "depth_anything_v2_small.onnx",
        "size_mb": "~100MB",
        "description": "Monocular depth estimation (Depth Anything V2 Small)",
    },
    "optical_flow": {
        "url": "https://github.com/neufieldrobotics/NeuFlow_v2/releases/download/v2.0/neuflowv2_mixed.onnx",
        "filename": "neuflowv2_mixed.onnx",
        "size_mb": "~40MB",
        "description": "Dense optical flow (NeuFlowV2 mixed)",
    },
    "sam2_tiny": {
        "url": "https://huggingface.co/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.onnx",
        "filename": "sam2_tiny.onnx",
        "size_mb": "~160MB",
        "description": "Image segmentation (SAM 2 Tiny)",
    },
    "foot_tracker": {
        "url": "https://huggingface.co/qualcomm/Person-Foot-Detection/resolve/main/foot_detector.onnx",
        "filename": "foot_tracker.onnx",
        "size_mb": "~10MB",
        "description": "Person and foot detection (FootTrackNet)",
    },
    "video_matting": {
        "url": "https://huggingface.co/PINTO0309/RobustVideoMatting/resolve/main/rvm_mobilenetv3_fp32.onnx",
        "filename": "rvm_mobilenetv3.onnx",
        "size_mb": "~20MB",
        "description": "Video background removal (RobustVideoMatting MobileNetV3)",
    },
    "lama": {
        "url": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        "filename": "lama_fp32.onnx",
        "size_mb": "~174MB",
        "description": "Image inpainting (LAMA Dilated)",
    },
}


def download_model(model_id: str) -> None:
    """Download a single model."""
    import urllib.request

    info = MODELS[model_id]
    dest = MODELS_DIR / info["filename"]

    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading {info['description']} ({info['size_mb']})...")
    print(f"  URL: {info['url']}")
    urllib.request.urlretrieve(info["url"], dest)
    print(f"  Saved: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ML model weights")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument(
        "--model", type=str, choices=list(MODELS.keys()), help="Download specific model"
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for mid, info in MODELS.items():
            print(f"  {mid}: {info['description']} ({info['size_mb']})")
        return

    if args.all:
        print("Downloading all models...")
        for model_id in MODELS:
            download_model(model_id)
        print("Done!")
    elif args.model:
        download_model(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
