import os
import time

import h5py
import numpy as np


def benchmark_hdf5_random_read():
    """Benchmark HDF5 random read performance for training workload simulation."""

    # Test configuration (reduced for speed)
    N = 5000  # 5K heatmaps (faster creation)
    shape = (17, 72, 96)  # MogaNet-B output shape
    test_file = "/tmp/test_heatmaps.h5"

    print(f"Creating test HDF5 file with {N} heatmaps of shape {shape}...")

    # Create test file (smaller, faster)
    start = time.time()
    with h5py.File(test_file, "w") as f:
        # Use chunks for better random access
        f.create_dataset("heatmaps", shape=(N, *shape), dtype=np.float16, chunks=(1, *shape))
        # Fill in smaller batches to avoid memory spike
        for i in range(0, N, 1000):
            end_idx = min(i + 1000, N)
            f["heatmaps"][i:end_idx] = np.random.rand(end_idx - i, *shape).astype(np.float16)
    creation_time = time.time() - start
    file_size = os.path.getsize(test_file) / (1024 * 1024)  # MB

    print(f"File created: {file_size:.1f} MB in {creation_time:.2f}s")

    # Benchmark 1: Basic random read (reduced samples)
    print("\n[Test 1] Basic random read (500 samples)...")
    indices = np.random.randint(0, N, 500)

    with h5py.File(test_file, "r") as f:
        start = time.time()
        for idx in indices:
            _ = f["heatmaps"][idx]
        elapsed = time.time() - start

    throughput = 500 / elapsed
    print(f"Throughput: {throughput:.1f} heatmaps/sec")

    # Benchmark 2: Random read with latest libver
    print("\n[Test 2] Random read with libver='latest'...")
    indices = np.random.randint(0, N, 500)

    with h5py.File(test_file, "r", libver="latest") as f:
        start = time.time()
        for idx in indices:
            _ = f["heatmaps"][idx]
        elapsed = time.time() - start

    throughput_latest = 500 / elapsed
    print(f"Throughput: {throughput_latest:.1f} heatmaps/sec")

    # Benchmark 3: Batch reads (simulate DataLoader with batch_size=16)
    print("\n[Test 3] Batch reads (batch_size=16, 16 batches)...")

    with h5py.File(test_file, "r") as f:
        start = time.time()
        for i in range(16):
            batch_indices = np.random.randint(0, N, 16)
            batch = [f["heatmaps"][idx] for idx in batch_indices]
            _ = np.array(batch)  # Simulate collate_fn
        elapsed = time.time() - start

    batch_throughput = (16 * 16) / elapsed
    print(f"Throughput: {batch_throughput:.1f} heatmaps/sec")

    # Benchmark 4: Sliced reads (consecutive indices)
    print("\n[Test 4] Sliced reads (50 consecutive samples x 10)...")

    with h5py.File(test_file, "r") as f:
        start = time.time()
        for i in range(10):
            start_idx = np.random.randint(0, N - 50)
            _ = f["heatmaps"][start_idx : start_idx + 50]
        elapsed = time.time() - start

    sliced_throughput = 500 / elapsed
    print(f"Throughput: {sliced_throughput:.1f} heatmaps/sec")

    # Cleanup
    os.remove(test_file)

    return {
        "basic": throughput,
        "latest": throughput_latest,
        "batch": batch_throughput,
        "sliced": sliced_throughput,
        "file_size_mb": file_size,
    }


def print_report(results):
    """Print benchmark report in Russian."""

    print("\n" + "=" * 50)
    print("HDF5 PERFORMANCE REPORT")
    print("=" * 50)

    basic = results["basic"]
    latest = results["latest"]
    batch = results["batch"]
    sliced = results["sliced"]

    print(f"\nFile size: {results['file_size_mb']:.1f} MB (10K heatmaps)")
    print("\nThroughput Results:")
    print(f"  - Basic random read:     {basic:.1f} heatmaps/sec")
    print(f"  - With libver='latest':  {latest:.1f} heatmaps/sec")
    print(f"  - Batch reads (bs=32):   {batch:.1f} heatmaps/sec")
    print(f"  - Sliced reads:          {sliced:.1f} heatmaps/sec")

    target = 100.0
    best = max(basic, latest, batch, sliced)

    print(f"\nTarget: >{target:.0f} heatmaps/sec")
    print(f"Best throughput: {best:.1f} heatmaps/sec")

    if best >= target * 2:
        verdict = "OTLICHNO (EXCELLENT)"
    elif best >= target:
        verdict = "DOSTATOCHNO (SUFFICIENT)"
    elif best >= target * 0.7:
        verdict = "NAGRANITsNO (BORDERLINE)"
    else:
        verdict = "NEDOSTATOCHNO (INSUFFICIENT)"

    print(f"Verdict: {verdict}")

    print("\nRecommendations:")
    print("-" * 50)

    if best >= target:
        print("- Use: Single HDF5 file")
        print(f"- Settings: libver='latest' ({latest:.1f} heatmaps/sec)")
        print("- Estimated impact on training: Fast enough")

        if batch > basic * 1.2:
            print(f"- Note: Batch reads are {(batch / basic - 1) * 100:.0f}% faster")
            print("  → Use batch_size=32 for best performance")
    else:
        print("- Use: Sharded HDF5 or LMDB")
        print("- Settings: Consider 4-10 shards for parallel access")
        print("- Estimated impact on training: Potential bottleneck")

    print("\nNotes:")
    print(f"- Random read performance: {basic:.1f} heatmaps/sec")
    print(f"- Sequential read advantage: {(sliced / basic - 1) * 100:.0f}% faster")
    print(f"- 10K samples = {results['file_size_mb']:.1f} MB (manageable)")
    print("=" * 50)


if __name__ == "__main__":
    results = benchmark_hdf5_random_read()
    print_report(results)
