"""Benchmark Numba JIT vs pure NumPy - before/after comparison."""

import time

import numpy as np


def benchmark_angle_before_after():
    """Compare pure NumPy vs Numba JIT for angle calculations."""
    from skating_ml.utils.geometry import angle_3pt, angle_3pt_batch

    print("=" * 60)
    print("ANGLE CALCULATION (1M operations)")
    print("=" * 60)

    # Setup
    n = 10000
    runs = 100
    triplets = np.random.randn(n, 3, 2).astype(np.float64)

    # BEFORE: Pure NumPy loop
    print("\n🔵 BEFORE: Pure NumPy loop (no JIT)")
    start = time.perf_counter()
    for _ in range(runs):
        for i in range(n):
            a, b, c = triplets[i]
            _ = angle_3pt(a, b, c)
    time_before = time.perf_counter() - start
    ops_before = n * runs
    print(f"  Time: {time_before:.3f}s")
    print(f"  Speed: {ops_before / time_before:,.0f} ops/sec")

    # AFTER: Numba JIT batch
    print("\n🟢 AFTER: Numba JIT batch")
    # Warmup
    _ = angle_3pt_batch(triplets[:10])

    start = time.perf_counter()
    for _ in range(runs):
        _ = angle_3pt_batch(triplets)
    time_after = time.perf_counter() - start
    ops_after = n * runs
    print(f"  Time: {time_after:.3f}s")
    print(f"  Speed: {ops_after / time_after:,.0f} ops/sec")

    speedup = time_before / time_after
    print(f"\n📊 SPEEDUP: {speedup:.1f}x faster")

    return speedup


def benchmark_smoothing_before_after():
    """Compare pure Python vs Numba JIT for smoothing."""
    from skating_ml.utils.smoothing import OneEuroFilter, smooth_trajectory_2d_numba

    print("\n" + "=" * 60)
    print("SMOOTHING (100K frames, 2D trajectory)")
    print("=" * 60)

    # Setup
    trajectory = np.random.randn(1000, 2).astype(np.float64)
    runs = 100

    # BEFORE: Pure Python OneEuroFilter
    print("\n🔵 BEFORE: Pure Python OneEuroFilter")
    filter_obj = OneEuroFilter(freq=30.0, min_cutoff=0.004, beta=0.7)

    start = time.perf_counter()
    for _ in range(runs):
        for i in range(100):
            series = trajectory[:, 0]
            _ = filter_obj.reset_and_filter(series)
    time_before = time.perf_counter() - start
    frames_before = 100 * runs
    print(f"  Time: {time_before:.3f}s")
    print(f"  Speed: {frames_before / time_before:,.0f} frames/sec")

    # AFTER: Numba JIT batch
    print("\n🟢 AFTER: Numba JIT smooth_trajectory_2d")
    # Warmup
    _ = smooth_trajectory_2d_numba(trajectory, fps=30.0, min_cutoff=0.004, beta=0.7, d_cutoff=1.0)

    start = time.perf_counter()
    for _ in range(runs):
        _ = smooth_trajectory_2d_numba(
            trajectory, fps=30.0, min_cutoff=0.004, beta=0.7, d_cutoff=1.0
        )
    time_after = time.perf_counter() - start
    frames_after = 1000 * runs
    print(f"  Time: {time_after:.3f}s")
    print(f"  Speed: {frames_after / time_after:,.0f} frames/sec")

    speedup = time_before / time_after
    print(f"\n📊 SPEEDUP: {speedup:.1f}x faster")

    return speedup


def benchmark_metrics_before_after():
    """Compare loop vs Numba JIT for metrics."""
    from skating_ml.analysis.metrics import _compute_knee_angle_series_numba
    from skating_ml.types import H36Key
    from skating_ml.utils.geometry import angle_3pt

    print("\n" + "=" * 60)
    print("KNEE ANGLE SERIES (100K calculations)")
    print("=" * 60)

    # Setup
    poses = np.random.randn(100, 17, 2).astype(np.float32)
    runs = 1000

    # BEFORE: Pure Python loop with angle_3pt
    print("\n🔵 BEFORE: Pure Python loop + angle_3pt")
    hip_idx, knee_idx, foot_idx = H36Key.LHIP, H36Key.LKNEE, H36Key.LFOOT

    start = time.perf_counter()
    for _ in range(runs):
        for i in range(100):
            pose = poses[i]
            _ = angle_3pt(pose[hip_idx], pose[knee_idx], pose[foot_idx])
    time_before = time.perf_counter() - start
    ops_before = 100 * runs
    print(f"  Time: {time_before:.3f}s")
    print(f"  Speed: {ops_before / time_before:,.0f} ops/sec")

    # AFTER: Numba JIT batch
    print("\n🟢 AFTER: Numba JIT _compute_knee_angle_series_numba")
    # Warmup
    _ = _compute_knee_angle_series_numba(
        poses, int(H36Key.LHIP), int(H36Key.LKNEE), int(H36Key.LFOOT)
    )

    start = time.perf_counter()
    for _ in range(runs):
        _ = _compute_knee_angle_series_numba(
            poses, int(H36Key.LHIP), int(H36Key.LKNEE), int(H36Key.LFOOT)
        )
    time_after = time.perf_counter() - start
    ops_after = 100 * runs
    print(f"  Time: {time_after:.3f}s")
    print(f"  Speed: {ops_after / time_after:,.0f} ops/sec")

    speedup = time_before / time_after
    print(f"\n📊 SPEEDUP: {speedup:.1f}x faster")

    return speedup


def print_complexity_explanation():
    """Explain computational complexity."""
    print("\n" + "=" * 60)
    print("COMPUTATIONAL COMPLEXITY EXPLAINED")
    print("=" * 60)

    print("""
Почему Python медленный для этих операций?

1. **angle_3pt loop (O(n))**:
   - Python loop overhead: ~50ns per iteration
   - NumPy array indexing: ~100ns per access
   - Итого: ~150-200ns per operation vs ~2ns with Numba JIT

2. **OneEuroFilter (O(n) per frame)**:
   - Python: for loop → function calls → NumPy operations
   - Numba: compiled loop → CPU instructions (no Python overhead)
   - Сглаживание 1000 кадров:
     * Python: 1000 × (loop overhead + computations)
     * Numba: векторизованный loop с near-C скоростью

3. **Knee angle series (O(n) where n=frames)**:
   - Python: для каждого кадра вызывается angle_3pt
   - Numba: весь цикл компилируется в одну функцию
   - Для 100 кадров × 1000 повторений:
     * Python: 100,000 Python function calls
     * Numba: 1 скомпилированная функция, 100,000 ops

Ключевой инсайт:
Python interpreter overhead → основной bottleneck для compute-intensive операций.
Numba JIT компилирует Python код в машинный код, устраняя overhead.

Результат: 10-100x ускорение для операций, которые вызываются многократно.
""")


if __name__ == "__main__":
    print_complexity_explanation()

    speedup_angle = benchmark_angle_before_after()
    speedup_smoothing = benchmark_smoothing_before_after()
    speedup_metrics = benchmark_metrics_before_after()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Angle calculation:   {speedup_angle:.1f}x faster")
    print(f"Smoothing:           {speedup_smoothing:.1f}x faster")
    print(f"Metrics (knee angle): {speedup_metrics:.1f}x faster")
    print("\n✅ Numba JIT дает существенное ускорение для compute-intensive операций!")
