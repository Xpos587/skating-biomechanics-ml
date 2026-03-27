#!/usr/bin/env python3
"""Download ML models required for the skating analysis system.

This script downloads:
- YOLOv11n weights for person detection
- BlazePose models (auto-downloaded by MediaPipe on first use)

Run this script once before using the system.
"""

import sys
from pathlib import Path


def main() -> int:
    """Download required models."""
    print("Downloading required ML models...")
    print("=" * 60)

    # Download YOLOv11n
    print("\n1. Downloading YOLOv11n for person detection...")
    try:
        from ultralytics import YOLO

        model = YOLO("yolov11n.pt")
        print(f"   ✓ YOLOv11n downloaded successfully")
        print(f"   Model size: ~6MB")
        print(f"   Params: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"   ✗ Failed to download YOLOv11n: {e}")
        return 1

    # MediaPipe BlazePose is auto-downloaded on first use
    print("\n2. BlazePose models:")
    print("   BlazePose (MediaPipe) will be auto-downloaded on first use")
    print("   Model size: ~10-20MB depending on complexity")
    print("   No manual download required")

    print("\n" + "=" * 60)
    print("✓ All models ready!")
    print("\nYou can now run:")
    print("  uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element three_turn")

    return 0


if __name__ == "__main__":
    sys.exit(main())
