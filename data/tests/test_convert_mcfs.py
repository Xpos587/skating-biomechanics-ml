"""Tests for MCFS converter."""

from pathlib import Path

import numpy as np
import pytest

RAW_DIR = Path("data/datasets/raw/mcfs")
OUTPUT_DIR = Path("data/datasets/unified/mcfs-129")


def test_convert_mcfs_creates_output():
    """Test that convert_mcfs creates expected output files."""
    from data_tools.convert_mcfs import convert_mcfs

    if not (RAW_DIR / "segments.pkl").exists():
        pytest.skip("MCFS segments.pkl not found")

    # Run conversion
    stats = convert_mcfs(RAW_DIR, OUTPUT_DIR)

    # Check output files exist
    assert (OUTPUT_DIR / "all_poses.npy").exists(), "all_poses.npy not created"
    assert (OUTPUT_DIR / "all_labels.npy").exists(), "all_labels.npy not created"
    assert (OUTPUT_DIR / "meta.json").exists(), "meta.json not created"

    # Check stats
    assert stats["all"] > 2000, f"Expected >2000 samples, got {stats['all']}"
    assert stats["classes"] >= 100, f"Expected >=100 classes, got {stats['classes']}"
    assert "created" in stats, "Missing 'created' timestamp in stats"
    assert "label_mapping" in stats, "Missing 'label_mapping' in stats"


def test_convert_mcfs_shapes():
    """Test that converted data has correct shapes."""
    from data_tools.convert_mcfs import convert_mcfs, load_unified

    if not (RAW_DIR / "segments.pkl").exists():
        pytest.skip("MCFS segments.pkl not found")

    # Run conversion
    convert_mcfs(RAW_DIR, OUTPUT_DIR)

    # Load all split
    poses, labels = load_unified(OUTPUT_DIR, split="all")

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


def test_convert_mcfs_no_nan():
    """Test that poses contain no NaN values."""
    from data_tools.convert_mcfs import convert_mcfs, load_unified

    if not (RAW_DIR / "segments.pkl").exists():
        pytest.skip("MCFS segments.pkl not found")

    # Run conversion
    convert_mcfs(RAW_DIR, OUTPUT_DIR)

    # Load all split
    poses, _ = load_unified(OUTPUT_DIR, split="all")

    # Check no NaN in any sample
    for i, pose in enumerate(poses):
        assert not np.isnan(pose).any(), f"Sample {i} contains NaN values"
