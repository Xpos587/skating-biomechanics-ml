"""Benchmark pose extraction performance.

Compares sequential vs batched pose extraction to measure speedup.
"""

import time
from pathlib import Path

import numpy as np


def benchmark_sequential_extraction(
    video_path: Path,
    mode: str = "balanced",
) -> tuple[float, float]:
    """Benchmark sequential (current) pose extraction.

    Args:
        video_path: Path to video file.
        mode: RTMO model mode.

    Returns:
        (elapsed_time, fps) tuple.
    """
    from skating_ml.pose_estimation import PoseExtractor

    print("=" * 60)
    print("SEQUENTIAL EXTRACTION (BASELINE)")
    print("=" * 60)

    extractor = PoseExtractor(mode=mode, device="cuda")

    start = time.perf_counter()
    result = extractor.extract_video_tracked(video_path)
    elapsed = time.perf_counter() - start

    num_frames = result.poses.shape[0]
    fps = num_frames / elapsed

    print("\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Frames: {num_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Mode: {mode}")
    print("  Device: cuda")

    return elapsed, fps


def benchmark_batched_extraction(
    video_path: Path,
    batch_size: int = 8,
    mode: str = "balanced",
) -> tuple[float, float]:
    """Benchmark batched pose extraction.

    Args:
        video_path: Path to video file.
        batch_size: Number of frames per batch.
        mode: RTMO model mode.

    Returns:
        (elapsed_time, fps) tuple.
    """
    from skating_ml.pose_estimation import BatchPoseExtractor

    print("=" * 60)
    print(f"BATCHED EXTRACTION (batch_size={batch_size})")
    print("=" * 60)

    extractor = BatchPoseExtractor(batch_size=batch_size, mode=mode, device="cuda")

    start = time.perf_counter()
    result = extractor.extract_video_tracked(video_path)
    elapsed = time.perf_counter() - start

    num_frames = result.poses.shape[0]
    fps = num_frames / elapsed

    print("\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Frames: {num_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Batch size: {batch_size}")
    print(f"  Mode: {mode}")
    print("  Device: cuda")

    return elapsed, fps


def benchmark_batch_sizes(
    video_path: Path,
    batch_sizes: list[int] = [1, 2, 4, 8, 16],
    mode: str = "balanced",
) -> dict[int, tuple[float, float]]:
    """Benchmark multiple batch sizes.

    Args:
        video_path: Path to video file.
        batch_sizes: List of batch sizes to test.
        mode: RTMO model mode.

    Returns:
        Dict mapping batch_size to (elapsed_time, fps).
    """
    results = {}

    print("\n" + "=" * 60)
    print("BATCH SIZE COMPARISON")
    print("=" * 60)

    for batch_size in batch_sizes:
        elapsed, fps = benchmark_batched_extraction(video_path, batch_size, mode)
        results[batch_size] = (elapsed, fps)
        print()

    return results


def print_comparison_table(
    baseline: tuple[float, float],
    results: dict[int, tuple[float, float]],
) -> None:
    """Print comparison table of batch sizes.

    Args:
        baseline: (elapsed_time, fps) for sequential extraction.
        results: Dict mapping batch_size to (elapsed_time, fps).
    """
    baseline_elapsed, baseline_fps = baseline

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Batch Size':>12} | {'Time (s)':>10} | {'FPS':>8} | {'Speedup':>10}")
    print("-" * 60)

    # Print baseline
    print(f"{'Sequential':>12} | {baseline_elapsed:10.2f} | {baseline_fps:8.2f} | {'1.00x':>10}")

    # Print batched results
    for batch_size, (elapsed, fps) in sorted(results.items()):
        speedup = baseline_elapsed / elapsed
        print(f"{batch_size:12d} | {elapsed:10.2f} | {fps:8.2f} | {speedup:9.2f}x")

    print("=" * 60)

    # Find best batch size
    best_batch_size = max(results.keys(), key=lambda bs: results[bs][1])
    best_elapsed, best_fps = results[best_batch_size]
    best_speedup = baseline_elapsed / best_elapsed

    print(f"\n🏆 Best batch size: {best_batch_size}")
    print(f"   Speedup: {best_speedup:.2f}x")
    print(f"   FPS: {best_fps:.2f}")
    print(f"   Time: {best_elapsed:.2f}s")


def main():
    """Run benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark pose extraction performance")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["lightweight", "balanced", "performance"],
        help="RTMO model mode",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--sequential-only",
        action="store_true",
        help="Only benchmark sequential extraction",
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        return

    print(f"\nBenchmarking video: {args.video}")
    print("Video metadata:")

    from skating_ml.utils.video import get_video_meta

    meta = get_video_meta(args.video)
    print(f"  Resolution: {meta.width}x{meta.height}")
    print(f"  FPS: {meta.fps}")
    print(f"  Frames: {meta.num_frames}")
    print(f"  Duration: {meta.num_frames / meta.fps:.2f}s")

    # Benchmark sequential extraction
    baseline = benchmark_sequential_extraction(args.video, args.mode)

    if args.sequential_only:
        return

    # Benchmark batched extraction
    results = benchmark_batch_sizes(args.video, args.batch_sizes, args.mode)

    # Print comparison table
    print_comparison_table(baseline, results)

    # Print recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    best_batch_size = max(results.keys(), key=lambda bs: results[bs][1])
    best_speedup = baseline[0] / results[best_batch_size][0]

    if best_speedup > 3.0:
        print("✅ Excellent speedup! Frame batching is highly effective.")
    elif best_speedup > 2.0:
        print("✅ Good speedup! Frame batching provides significant improvement.")
    elif best_speedup > 1.5:
        print("⚠️ Moderate speedup. Consider optimizing batch size or pipeline.")
    else:
        print("❌ Low speedup. Current implementation may not benefit from batching.")

    print(f"\nRecommended batch size: {best_batch_size}")
    print(f"Expected speedup: {best_speedup:.2f}x")

    # GPU utilization advice
    print("\n" + "=" * 60)
    print("GPU UTILIZATION")
    print("=" * 60)
    print("To measure GPU utilization during benchmarking:")
    print("  1. Open a terminal and run: watch -n 0.1 nvidia-smi")
    print("  2. Run this benchmark in another terminal")
    print("  3. Observe GPU utilization percentage")
    print("\nTarget: > 80% GPU utilization for optimal performance")


if __name__ == "__main__":
    main()
