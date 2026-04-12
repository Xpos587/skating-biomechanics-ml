#!/usr/bin/env python3
"""Download ML model weights for optional pipeline features.

Usage:
    uv run python scripts/download_ml_models.py --all
    uv run python scripts/download_ml_models.py --model depth_anything
    uv run python scripts/download_ml_models.py --list

Uses huggingface_hub for HuggingFace models (auto-reads HF_TOKEN env var),
urllib for GitHub releases.
"""

import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS: dict[str, dict] = {
    "depth_anything": {
        "source": "hf",
        "repo_id": "onnx-community/depth-anything-v2-small",
        "filename": "onnx/model.onnx",
        "local_filename": "depth_anything_v2_small.onnx",
        "size_mb": "~99MB",
        "description": "Monocular depth estimation (Depth Anything V2 Small)",
    },
    "optical_flow": {
        "source": "url",
        "url": "https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow/releases/download/0.1.0/neuflow_mixed.onnx",
        "local_filename": "neuflowv2_mixed.onnx",
        "size_mb": "~40MB",
        "description": "Dense optical flow (NeuFlowV2 mixed)",
    },
    "sam2_tiny": {
        "source": "hf_multi",
        "repo_id": "onnx-community/sam2.1-hiera-tiny-ONNX",
        "files": [
            ("onnx/vision_encoder.onnx", "sam2/vision_encoder.onnx"),
            ("onnx/vision_encoder.onnx_data", "sam2/vision_encoder.onnx_data"),
            ("onnx/prompt_encoder_mask_decoder.onnx", "sam2/prompt_encoder_mask_decoder.onnx"),
            (
                "onnx/prompt_encoder_mask_decoder.onnx_data",
                "sam2/prompt_encoder_mask_decoder.onnx_data",
            ),
        ],
        "local_filename": "sam2/vision_encoder.onnx",
        "size_mb": "~155MB (4 files in sam2/)",
        "description": "Image segmentation (SAM 2.1 Tiny)",
    },
    "foot_tracker": {
        "source": "manual",
        "local_filename": "foot_tracker.onnx",
        "size_mb": "~10MB",
        "description": "Person and foot detection (FootTrackNet) — requires manual export",
    },
    "video_matting": {
        "source": "hf",
        "repo_id": "LPDoctor/video_matting",
        "filename": "rvm_mobilenetv3_fp32.onnx",
        "local_filename": "rvm_mobilenetv3.onnx",
        "size_mb": "~33MB",
        "description": "Video background removal (RobustVideoMatting MobileNetV3)",
    },
    "lama": {
        "source": "hf",
        "repo_id": "Carve/LaMa-ONNX",
        "filename": "lama_fp32.onnx",
        "local_filename": "lama_fp32.onnx",
        "size_mb": "~208MB",
        "description": "Image inpainting (LAMA Dilated)",
    },
}


def download_model(model_id: str) -> None:
    """Download a single model."""
    info = MODELS[model_id]
    source = info["source"]

    if source == "manual":
        print(f"  [MANUAL] {info['description']}")
        print("    Export from: pip install 'qai-hub-models[foot-track-net]'")
        print(
            '    Then: python -c "from qai_hub_models.models.foot_track_net import FootTrackNet; \\'
        )
        print("      m = FootTrackNet.from_pretrained(); m.eval(); \\")
        print(
            '      import torch; torch.onnx.export(m, torch.randn(1,3,480,640), \\"data/models/foot_tracker.onnx\\")"'
        )
        return

    if source == "hf":
        dest = MODELS_DIR / info["local_filename"]
        if dest.exists():
            print(f"  Already exists: {dest}")
            return
        print(f"  Downloading {info['description']} ({info['size_mb']})...")
        path = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=MODELS_DIR,
        )
        downloaded = Path(path)
        if downloaded != dest and downloaded.exists():
            downloaded.rename(dest)
        print(f"  Saved: {dest}")

    elif source == "hf_multi":
        all_exist = all((MODELS_DIR / local).exists() for _, local in info["files"])
        if all_exist:
            print(f"  Already exists: {[MODELS_DIR / loc for _, loc in info['files']]}")
            return
        print(f"  Downloading {info['description']} ({info['size_mb']})...")
        for hf_file, local_name in info["files"]:
            dest = MODELS_DIR / local_name
            if dest.exists():
                print(f"    Already exists: {dest}")
                continue
            path = hf_hub_download(
                repo_id=info["repo_id"],
                filename=hf_file,
                local_dir=MODELS_DIR,
            )
            downloaded = Path(path)
            if downloaded != dest and downloaded.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                downloaded.rename(dest)
            print(f"    Saved: {dest}")
        print(f"  Done: {len(info['files'])} files")

    elif source == "url":
        dest = MODELS_DIR / info["local_filename"]
        if dest.exists():
            print(f"  Already exists: {dest}")
            return
        print(f"  Downloading {info['description']} ({info['size_mb']})...")
        req = urllib.request.Request(info["url"])
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  Saved: {dest} ({len(data) / 1024 / 1024:.1f}MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ML model weights")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        help="Download specific model",
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for mid, info in MODELS.items():
            src = info["source"]
            tag = {
                "hf": "[HuggingFace]",
                "hf_multi": "[HuggingFace, multi-file]",
                "url": "[GitHub Release]",
                "manual": "[Manual export required]",
            }[src]
            print(f"  {mid}: {info['description']} ({info['size_mb']}) {tag}")
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
