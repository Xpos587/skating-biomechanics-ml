#!/usr/bin/env python3
"""Download ML models required for the skating analysis system.

This script downloads:
- YOLO26n-Pose for 2D pose estimation (H3.6M 17-keypoint format)

Run this script once before using the system.
"""

import sys


def main() -> int:
    """Download required models."""
    print("Downloading required ML models...")
    print("=" * 60)

    # Download YOLO26n-Pose
    print("\n1. Downloading YOLO26n-Pose for 2D pose estimation...")
    try:
        from ultralytics import YOLO

        model = YOLO("yolo26n-pose.pt")
        print("   ✓ YOLO26n-Pose downloaded successfully")
        print("   Model size: ~6MB")
        print(f"   Params: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"   ✗ Failed to download YOLO26n-Pose: {e}")
        return 1

    print("\n" + "=" * 60)
    print("✓ All models ready!")
    print("\nYou can now run:")
    print("  uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element three_turn")

    return 0


if __name__ == "__main__":
    sys.exit(main())
