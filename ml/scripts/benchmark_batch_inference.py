"""Benchmark batch vs per-frame RTMO inference."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose_estimation.rtmo_batch import BatchRTMO


def benchmark_batch(video_path: str, batch_sizes: list[int] | None = None):
    """Compare batch inference times."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16]
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Total frames: {len(frames)}")

    for bs in batch_sizes:
        rtmo = BatchRTMO(mode="balanced", device="cuda", score_thr=0.3, nms_thr=0.45)

        start = time.perf_counter()
        for i in range(0, len(frames), bs):
            batch = frames[i : i + bs]
            rtmo.infer_batch(batch)
        elapsed = time.perf_counter() - start

        fps = len(frames) / elapsed
        print(f"batch_size={bs:2d}: {elapsed:.3f}s ({fps:.1f} fps)")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/test_video.mp4"
    benchmark_batch(video)
