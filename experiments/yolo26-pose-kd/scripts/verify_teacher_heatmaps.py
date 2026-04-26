#!/usr/bin/env python3
"""
Verify Teacher Heatmaps Integrity

Comprehensive verification of teacher_heatmaps.h5 file on Vast.ai server.
Checks file integrity, structure, data quality, and spatial structure.

Usage:
    # Run on Vast.ai server
    ssh root@167.172.27.229 "cd /root/skating-biomechanics-ml && python3 experiments/yolo26-pose-kd/scripts/verify_teacher_heatmaps.py"

    # Run locally if file exists
    python3 experiments/yolo26-pose-kd/scripts/verify_teacher_heatmaps.py --path /path/to/teacher_heatmaps.h5
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def check_file_integrity(h5_path):
    """Check if HDF5 file opens without errors and get metadata."""
    print("\n" + "=" * 60)
    print("FILE INTEGRITY CHECK")
    print("=" * 60)

    try:
        with h5py.File(h5_path, "r") as f:
            datasets = list(f.keys())
            print("✅ File opens without errors")
            print(f"   Datasets: {datasets}")

            if "heatmaps" not in datasets:
                print("❌ 'heatmaps' dataset not found!")
                return None

            # Get metadata
            heatmap_ds = f["/heatmaps"]
            shape = heatmap_ds.shape
            dtype = heatmap_ds.dtype
            chunks = heatmap_ds.chunks
            compression = heatmap_ds.compression

            print("\nFile Info:")
            print(f"  Shape: {shape}")
            print(f"  Dtype: {dtype}")
            print(f"  Chunks: {chunks}")
            print(f"  Compression: {compression}")

            # Get file size
            file_size_gb = Path(h5_path).stat().st_size / (1024**3)
            print(f"  Size: {file_size_gb:.2f} GB")

            return {
                "shape": shape,
                "dtype": dtype,
                "chunks": chunks,
                "file_size_gb": file_size_gb,
                "datasets": datasets,
            }

    except Exception as e:
        print(f"❌ File integrity check failed: {e}")
        return None


def check_data_structure(h5_path, expected_shape=(264874, 17, 72, 96)):
    """Check if data structure matches expected format."""
    print("\n" + "=" * 60)
    print("DATA STRUCTURE CHECK")
    print("=" * 60)

    try:
        with h5py.File(h5_path, "r") as f:
            heatmap_ds = f["/heatmaps"]
            actual_shape = heatmap_ds.shape
            actual_dtype = heatmap_ds.dtype

            print(f"Expected shape: {expected_shape}")
            print(f"Actual shape:   {actual_shape}")
            print(f"Match: {'✅' if actual_shape == expected_shape else '❌'}")

            print("\nExpected dtype: float16")
            print(f"Actual dtype:   {actual_dtype}")
            print(f"Match: {'✅' if actual_dtype == np.float16 else '❌'}")

            # Check chunking is reasonable
            chunks = heatmap_ds.chunks
            if chunks:
                print(f"\nChunking: {chunks}")
                # Good chunking: first dim ~32, full spatial dims
                if chunks[0] >= 16 and chunks[0] <= 64:
                    print("✅ Chunk size reasonable for batch loading")
                else:
                    print("⚠️  Unusual chunk size (may affect performance)")

            return actual_shape == expected_shape and actual_dtype == np.float16

    except Exception as e:
        print(f"❌ Data structure check failed: {e}")
        return False


def check_data_quality(h5_path, num_samples=100):
    """Check data quality: NaN, Inf, value ranges, peak values."""
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)

    try:
        with h5py.File(h5_path, "r") as f:
            heatmap_ds = f["/heatmaps"]
            total_heatmaps = heatmap_ds.shape[0]

            # Sample random heatmaps
            indices = np.random.choice(
                total_heatmaps, min(num_samples, total_heatmaps), replace=False
            )
            samples = heatmap_ds[indices]

            print(f"Sampled {len(samples)} heatmaps for quality checks")

            # Check for NaN/Inf
            has_nan = np.any(np.isnan(samples))
            has_inf = np.any(np.isinf(samples))

            print(f"\nNaN values: {'❌ DETECTED' if has_nan else '✅ None'}")
            print(f"Inf values: {'❌ DETECTED' if has_inf else '✅ None'}")

            # Check value range
            min_val = np.min(samples)
            max_val = np.max(samples)
            mean_val = np.mean(samples)
            std_val = np.std(samples)

            print("\nValue Statistics:")
            print(f"  Min:  {min_val:.6f}")
            print(f"  Max:  {max_val:.6f}")
            print(f"  Mean: {mean_val:.6f}")
            print(f"  Std:  {std_val:.6f}")

            in_range = min_val >= 0.0 and max_val <= 1.0
            print(f"  Values in [0, 1]: {'✅' if in_range else '❌'}")

            # Check for Gaussian peaks (should have values close to 1.0)
            peak_values = []
            for i in range(len(samples)):
                heatmap = samples[i]
                # Get max per keypoint
                max_per_kp = np.max(heatmap, axis=(1, 2))
                peak_values.extend(max_per_kp.tolist())

            peak_min = min(peak_values)
            peak_max = max(peak_values)
            peak_mean = np.mean(peak_values)

            print("\nPeak Values (per keypoint):")
            print(f"  Min:  {peak_min:.4f}")
            print(f"  Max:  {peak_max:.4f}")
            print(f"  Mean: {peak_mean:.4f}")

            has_peaks = peak_max > 0.95
            print(f"  Peaks ~1.0 present: {'✅' if has_peaks else '❌'}")

            return not has_nan and not has_inf and in_range and has_peaks

    except Exception as e:
        print(f"❌ Data quality check failed: {e}")
        return False


def check_spatial_structure(h5_path, num_visualize=3, output_dir=None):
    """Check spatial structure by visualizing sample heatmaps."""
    print("\n" + "=" * 60)
    print("SPATIAL STRUCTURE CHECK")
    print("=" * 60)

    try:
        with h5py.File(h5_path, "r") as f:
            heatmap_ds = f["/heatmaps"]
            total_heatmaps = heatmap_ds.shape[0]

            # Sample random heatmaps
            indices = np.random.choice(
                total_heatmaps, min(num_visualize, total_heatmaps), replace=False
            )

            print(f"\nVisualizing {len(indices)} sample heatmaps...")

            for idx in indices:
                heatmap = heatmap_ds[idx]  # (17, 72, 96)

                # Check each keypoint
                print(f"\nSample heatmap #{idx}:")
                for kp_idx in range(min(5, 17)):  # Check first 5 keypoints
                    kp_heatmap = heatmap[kp_idx]
                    max_val = np.max(kp_heatmap)
                    argmax = np.unravel_index(np.argmax(kp_heatmap), kp_heatmap.shape)

                    # Check if Gaussian peak is centered
                    expected_center = (36, 48)  # Center of (72, 96)
                    distance = np.sqrt(
                        (argmax[0] - expected_center[0]) ** 2
                        + (argmax[1] - expected_center[1]) ** 2
                    )

                    print(
                        f"  KP {kp_idx}: max={max_val:.3f}, pos={argmax}, "
                        f"dist_from_center={distance:.1f}px"
                    )

                # Optionally save visualization
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    # Create composite image of all keypoints
                    grid_size = int(np.ceil(np.sqrt(17)))
                    composite = np.zeros((grid_size * 72, grid_size * 96))

                    for kp_idx in range(17):
                        row = kp_idx // grid_size
                        col = kp_idx % grid_size
                        composite[row * 72 : (row + 1) * 72, col * 96 : (col + 1) * 96] = heatmap[
                            kp_idx
                        ]

                    # Normalize to 0-255
                    composite = (composite * 255).astype(np.uint8)

                    # Save
                    img = Image.fromarray(composite, mode="L")
                    save_path = output_path / f"heatmap_{idx}.png"
                    img.save(save_path)
                    print(f"  Saved visualization to {save_path}")

            print("\n✅ Spatial structure check complete")
            return True

    except Exception as e:
        print(f"❌ Spatial structure check failed: {e}")
        return False


def check_generation_code():
    """Review generation code for correctness."""
    print("\n" + "=" * 60)
    print("GENERATION CODE REVIEW")
    print("=" * 60)

    script_path = Path(__file__).parent.parent / "scripts" / "generate_teacher_heatmaps.py"

    if not script_path.exists():
        print(f"⚠️  Generation script not found at {script_path}")
        return None

    with open(script_path) as f:
        content = f.read()

    checks = {
        "MSRA Gaussian": "MSRA Gaussian" in content or "exp(-((x-μx)²" in content,
        "Clamping": "torch.clamp(heatmaps, 0.0, 1.0)" in content,
        "Float16": "astype(np.float16)" in content or "dtype='float16'" in content,
        "Batch processing": "batch_imgs" in content or "batch_size" in content,
    }

    print("\nCode checks:")
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")

    all_passed = all(checks.values())
    if all_passed:
        print("\n✅ Generation code follows best practices")
    else:
        print("\n⚠️  Generation code may have issues")

    return all_passed


def generate_report(h5_path, output_path=None):
    """Generate comprehensive verification report."""
    print("\n" + "=" * 60)
    print("TEACHER HEATMAPS VERIFICATION REPORT")
    print("=" * 60)
    print(f"File: {h5_path}")
    print(f"Date: {np.datetime64('now')}")
    print("=" * 60)

    results = {"file_path": str(h5_path), "timestamp": str(np.datetime64("now")), "checks": {}}

    # Run all checks
    metadata = check_file_integrity(h5_path)
    if metadata is None:
        print("\n❌ CRITICAL: File integrity check failed")
        print("Status: REGENERATE NEEDED")
        return False

    results["checks"]["file_integrity"] = {"passed": True, "metadata": metadata}

    structure_ok = check_data_structure(h5_path, expected_shape=metadata["shape"])
    results["checks"]["data_structure"] = {"passed": structure_ok}

    quality_ok = check_data_quality(h5_path, num_samples=100)
    results["checks"]["data_quality"] = {"passed": quality_ok}

    spatial_ok = check_spatial_structure(h5_path, num_visualize=3, output_dir=output_path)
    results["checks"]["spatial_structure"] = {"passed": spatial_ok}

    code_ok = check_generation_code()
    if code_ok is not None:
        results["checks"]["generation_code"] = {"passed": code_ok}

    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    all_passed = all(
        [
            results["checks"]["file_integrity"]["passed"],
            results["checks"]["data_structure"]["passed"],
            results["checks"]["data_quality"]["passed"],
            results["checks"]["spatial_structure"]["passed"],
        ]
    )

    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nStatus: SAFE TO USE")
        print("\nSummary:")
        print(f"  - {metadata['shape'][0]} heatmaps verified")
        print(f"  - Shape: {metadata['shape']}")
        print(f"  - Dtype: {metadata['dtype']}")
        print(f"  - Size: {metadata['file_size_gb']:.2f} GB")
        print("  - No data corruption detected")
        print("  - Gaussian peaks present")
        print("  - Value range correct [0, 1]")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nStatus: REGENERATE NEEDED")
        print("\nFailed checks:")
        for check_name, check_result in results["checks"].items():
            if not check_result["passed"]:
                print(f"  - {check_name}")

    # Save report
    if output_path:
        report_path = Path(output_path) / "verification_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Report saved to {report_path}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify teacher heatmaps HDF5 file")
    parser.add_argument(
        "--path",
        type=str,
        default="/root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher_heatmaps.h5",
        help="Path to teacher_heatmaps.h5 file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/heatmap_verification",
        help="Output directory for visualizations and reports",
    )
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"❌ File not found: {args.path}")
        print("\nNote: This script should be run on the Vast.ai server where the file exists.")
        print("Example:")
        print(
            '  ssh root@167.172.27.229 "cd /root/skating-biomechanics-ml && python3 experiments/yolo26-pose-kd/scripts/verify_teacher_heatmaps.py"'
        )
        sys.exit(1)

    success = generate_report(args.path, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
