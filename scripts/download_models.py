#!/usr/bin/env python3
"""Download ML models required for the skating analysis system.

This script downloads:
- YOLOv8n-Pose for 2D pose estimation (H3.6M 17-keypoint format)

Run this script once before using the system.
"""

import sys


def main() -> int:
    """Download required models."""
    print("Downloading required ML models...")
    print("=" * 60)

    # Download YOLOv8n-Pose
    print("\n1. Downloading YOLOv8n-Pose for 2D pose estimation...")
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n-pose.pt")
        print("   ✓ YOLOv8n-Pose downloaded successfully")
        print("   Model size: ~6MB")
        print(f"   Params: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"   ✗ Failed to download YOLOv8n-Pose: {e}")
        return 1

    print("\n" + "=" * 60)
    print("✓ All models ready!")
    print("\nYou can now run:")
    print("  uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element three_turn")
    print("\nOr start the Streamlit UI:")
    print("  uv run streamlit run scripts/streamlit_ui.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
