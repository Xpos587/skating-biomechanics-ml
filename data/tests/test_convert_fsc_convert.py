"""Tests for FSC converter."""

from pathlib import Path

import numpy as np
import pytest


def test_convert_fsc_creates_output():
    """Test that convert_fsc creates expected output files."""
    from data_tools.convert_fsc import convert_fsc

    # Use the FSC dataset from the main repo (symlinked or copied)
    raw_dir = Path(
        "/home/michael/Github/skating-biomechanics-ml/data/datasets/figure-skating-classification"
    )
    if not raw_dir.exists():
        pytest.skip("FSC data not found")

    out = Path("data/datasets/unified/fsc-64")

    # Run conversion
    stats = convert_fsc(raw_dir, out)

    # Check output files exist
    assert (out / "train_poses.npy").exists(), "train_poses.npy not created"
    assert (out / "train_labels.npy").exists(), "train_labels.npy not created"
    assert (out / "test_poses.npy").exists(), "test_poses.npy not created"
    assert (out / "test_labels.npy").exists(), "test_labels.npy not created"
    assert (out / "meta.json").exists(), "meta.json not created"

    # Check stats (allowing for some validation failures)
    assert stats["train"] > 3500, f"Expected >3500 train samples, got {stats['train']}"
    assert stats["test"] > 500, f"Expected >500 test samples, got {stats['test']}"
    assert stats["classes"] == 64, f"Expected 64 classes, got {stats['classes']}"
    assert "created" in stats, "Missing 'created' timestamp in stats"


def test_convert_fsc_shapes():
    """Test that converted data has correct shapes."""
    from data_tools.convert_fsc import convert_fsc, load_unified

    raw_dir = Path(
        "/home/michael/Github/skating-biomechanics-ml/data/datasets/figure-skating-classification"
    )
    if not raw_dir.exists():
        pytest.skip("FSC data not found")

    out = Path("data/datasets/unified/fsc-64")

    # Run conversion
    convert_fsc(raw_dir, out)

    # Load train split
    poses, labels = load_unified(out, split="train")

    # Check poses
    assert isinstance(poses, list), "poses should be a list"
    assert len(poses) > 0, "poses list is empty"

    # Check first pose sequence
    first_pose = poses[0]
    assert first_pose.ndim == 3, (
        f"Expected 3D array (T, 17, 2), got {first_pose.ndim}D with shape {first_pose.shape}"
    )
    assert first_pose.shape[1] == 17, f"Expected 17 keypoints, got {first_pose.shape[1]}"
    assert first_pose.shape[2] == 2, f"Expected 2 channels (x,y), got {first_pose.shape[2]}"
    assert first_pose.shape[0] >= 5, f"Expected at least 5 frames, got {first_pose.shape[0]}"
    assert first_pose.dtype == np.float32, f"Expected float32, got {first_pose.dtype}"

    # Check labels
    assert labels.ndim == 1, f"Expected 1D labels array, got {labels.ndim}D"
    assert len(labels) == len(poses), f"Labels count {len(labels)} != poses count {len(poses)}"
    assert labels.dtype == np.int32, f"Expected int32 labels, got {labels.dtype}"

    # Check label range
    assert labels.min() >= 0, f"Min label {labels.min()} < 0"
    assert labels.max() < 64, f"Max label {labels.max()} >= 64"


def test_convert_fsc_normalization():
    """Test that poses are properly normalized."""
    from data_tools.convert_fsc import convert_fsc, load_unified

    raw_dir = Path(
        "/home/michael/Github/skating-biomechanics-ml/data/datasets/figure-skating-classification"
    )
    if not raw_dir.exists():
        pytest.skip("FSC data not found")

    out = Path("data/datasets/unified/fsc-64")

    # Run conversion
    convert_fsc(raw_dir, out)

    # Load train split
    poses, _ = load_unified(out, split="train")

    # Check that poses are root-centered (mid-hip near origin)
    # Mid-hip is average of keypoints 11 and 12 (H3.6M format)
    for i, pose in enumerate(poses[:10]):  # Check first 10
        mid_hip = pose[:, 11:13].mean(axis=(0, 1))  # Mean across frames and the 2 hip keypoints
        # Should be close to origin (allowing for numerical precision)
        assert np.allclose(mid_hip, 0.0, atol=0.1), (
            f"Sample {i} not root-centered: mid_hip={mid_hip}"
        )


def test_load_unified_train_test():
    """Test that load_unified works for both train and test splits."""
    from data_tools.convert_fsc import convert_fsc, load_unified

    raw_dir = Path(
        "/home/michael/Github/skating-biomechanics-ml/data/datasets/figure-skating-classification"
    )
    if not raw_dir.exists():
        pytest.skip("FSC data not found")

    out = Path("data/datasets/unified/fsc-64")

    # Run conversion
    convert_fsc(raw_dir, out)

    # Load both splits
    train_poses, train_labels = load_unified(out, split="train")
    test_poses, test_labels = load_unified(out, split="test")

    # Check that train has more samples than test
    assert len(train_poses) > len(test_poses), (
        f"Train {len(train_poses)} should be larger than test {len(test_poses)}"
    )

    # Check that both have valid data
    assert len(train_poses) > 0, "Train poses empty"
    assert len(test_poses) > 0, "Test poses empty"
    assert len(train_labels) == len(train_poses), "Train labels/poses count mismatch"
    assert len(test_labels) == len(test_poses), "Test labels/poses count mismatch"
