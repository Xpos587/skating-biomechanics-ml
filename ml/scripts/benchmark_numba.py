"""Benchmark Numba JIT performance."""

import time

import numpy as np


def benchmark_geometry():
    """Benchmark geometry functions."""
    from skating_ml.utils.geometry import angle_3pt, angle_3pt_batch

    print("Benchmarking geometry functions...")

    # Setup
    n = 10000
    triplets = np.random.randn(n, 3, 2).astype(np.float64)

    # Warmup (compile)
    _ = angle_3pt_batch(triplets[:10])

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = angle_3pt_batch(triplets)
    elapsed = time.perf_counter() - start

    total_ops = n * 100
    print(f"  {total_ops:,} angle calculations in {elapsed:.3f}s")
    print(f"  {total_ops / elapsed:,.0f} ops/sec")


def benchmark_smoothing():
    """Benchmark smoothing."""
    from skating_ml.utils.smoothing import smooth_trajectory_2d_numba

    print("\nBenchmarking smoothing...")

    trajectory = np.random.randn(1000, 2).astype(np.float64)

    # Warmup
    _ = smooth_trajectory_2d_numba(trajectory, fps=30.0, min_cutoff=0.004, beta=0.7, d_cutoff=1.0)

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = smooth_trajectory_2d_numba(trajectory, fps=30.0, min_cutoff=0.004, beta=0.7, d_cutoff=1.0)
    elapsed = time.perf_counter() - start

    print(f"  100,000 frames smoothed in {elapsed:.3f}s")
    print(f"  {100000 / elapsed:,.0f} frames/sec")


def benchmark_metrics():
    """Benchmark metrics."""
    from skating_ml.analysis.metrics import _compute_knee_angle_series_numba
    from skating_ml.types import H36Key

    print("\nBenchmarking metrics...")

    poses = np.random.randn(100, 17, 2).astype(np.float32)

    # Warmup
    _ = _compute_knee_angle_series_numba(poses, int(H36Key.LHIP), int(H36Key.LKNEE), int(H36Key.LFOOT))

    # Benchmark
    start = time.perf_counter()
    for _ in range(1000):
        for i in range(100):
            _ = _compute_knee_angle_series_numba(
                poses,
                int(H36Key.LHIP),
                int(H36Key.LKNEE),
                int(H36Key.LFOOT),
            )
    elapsed = time.perf_counter() - start

    print(f"  100,000 CoM calculations in {elapsed:.3f}s")
    print(f"  {100000 / elapsed:,.0f} ops/sec")


if __name__ == "__main__":
    benchmark_geometry()
    benchmark_smoothing()
    benchmark_metrics()
    print("\n✅ All benchmarks complete!")
