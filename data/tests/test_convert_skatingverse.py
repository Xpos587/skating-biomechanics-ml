"""Tests for SkatingVerse converter."""

from pathlib import Path

import numpy as np
import pytest


def test_sv_metadata_loading():
    """Test that train.txt and mapping.txt parse correctly (no GPU needed)."""
    from data_tools.convert_skatingverse import (
        SKIP_CLASSES,
        _load_mapping,
        _load_split_index,
    )

    raw_dir = Path("data/datasets/raw/skatingverse")
    if not (raw_dir / "train.txt").exists():
        pytest.skip("SkatingVerse raw data not found")

    # Load train index
    entries = _load_split_index(raw_dir / "train.txt")
    assert len(entries) > 0, "train.txt should have entries"

    # Check first entry format: (name, int label)
    name, label = entries[0]
    assert isinstance(name, str), "Filename should be string"
    assert isinstance(label, int), "Label should be int"

    # Verify filenames do NOT have .mp4 extension
    assert not name.endswith(".mp4"), f"train.txt name should not have .mp4: {name}"

    # Load mapping
    mapping = _load_mapping(raw_dir / "mapping.txt")
    assert len(mapping) == 28, f"Expected 28 classes, got {len(mapping)}"

    # Check that skipped classes exist
    for cls_id in SKIP_CLASSES:
        assert cls_id in mapping, f"Skip class {cls_id} not in mapping"

    # Check known classes
    assert mapping.get(0) == "3Toeloop", f"Class 0 should be '3Toeloop', got {mapping.get(0)}"
    assert mapping.get(12) == "No Basic", f"Class 12 should be 'No Basic', got {mapping.get(12)}"
    assert mapping.get(27) == "Sequence", f"Class 27 should be 'Sequence', got {mapping.get(27)}"

    # Check that train entries reference valid class IDs
    all_labels = {label for _, label in entries}
    for label in all_labels:
        assert label in mapping, f"Label {label} in train.txt but not in mapping.txt"

    # Load answer (test) index — note: answer.txt has filenames only, no labels
    answer_entries = _load_split_index(raw_dir / "answer.txt")
    assert len(answer_entries) > 0, "answer.txt should have entries"

    # Verify test entries have None labels (unlabeled)
    for name, label in answer_entries[:5]:
        assert label is None, f"answer.txt entries should have no label, got {label} for {name}"

    # Verify train > test
    assert len(entries) > len(answer_entries), (
        f"Train ({len(entries)}) should be larger than test ({len(answer_entries)})"
    )


def test_sv_extract_single_video():
    """Test extraction from a single video (requires GPU)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    from data_tools.convert_skatingverse import extract_single_video
    from rtmlib import Wholebody

    raw_dir = Path("data/datasets/raw/skatingverse")
    videos_dir = raw_dir / "train_videos"
    if not videos_dir.exists():
        pytest.skip("SkatingVerse videos not found")

    # Find a non-skipped class video (not class 12 or 27)
    entries_path = raw_dir / "train.txt"
    if not entries_path.exists():
        pytest.skip("train.txt not found")

    from data_tools.convert_skatingverse import SKIP_CLASSES, _load_split_index

    entries = _load_split_index(entries_path)
    test_entry = None
    for name, label in entries:
        if label not in SKIP_CLASSES:
            video_file = videos_dir / f"{name}.mp4"
            if video_file.exists():
                test_entry = (video_file, name, label)
                break

    if test_entry is None:
        pytest.skip("No suitable test video found")

    video_file, video_name, _video_label = test_entry

    # Create model once
    wb = Wholebody(mode="balanced", backend="onnxruntime")

    # Extract
    result = extract_single_video(video_file, wb=wb, frame_skip=4)

    if result is None:
        pytest.skip(f"Video {video_name} returned no poses (too short or no detections)")

    # Check shape: (T, 17, 2)
    assert result.ndim == 3, f"Expected 3D (T, 17, 2), got {result.ndim}D shape {result.shape}"
    assert result.shape[1] == 17, f"Expected 17 keypoints, got {result.shape[1]}"
    assert result.shape[2] == 2, f"Expected 2 channels, got {result.shape[2]}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    # Should have reasonable number of frames for a short clip
    assert result.shape[0] >= 10, f"Expected >= 10 frames, got {result.shape[0]}"

    # Coordinates should be pixel values (positive, reasonable range)
    assert result.min() >= 0, f"Pixel coords should be non-negative, min={result.min()}"
    assert result.max() < 5000, f"Pixel coords too large, max={result.max()}"

    print(f"\nExtracted {video_name}: {result.shape[0]} frames, {result.shape[1]} keypoints")


@pytest.mark.timeout(300)
def test_sv_convert_batch_tiny(tmp_path):
    """Test batch conversion with max_per_class=1 (requires GPU)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    from data_tools.convert_skatingverse import convert_skatingverse, load_unified

    raw_dir = Path("data/datasets/raw/skatingverse")
    if not (raw_dir / "train.txt").exists():
        pytest.skip("SkatingVerse raw data not found")

    # Convert with max_per_class=1 to keep it fast
    output_dir = tmp_path / "sv-tiny"
    stats = convert_skatingverse(
        raw_dir,
        output_dir,
        max_per_class=1,
        frame_skip=16,  # Very aggressive skip for speed
    )

    # Check basic stats
    assert "train" in stats, "Stats missing 'train'"
    assert stats["train"] > 0, f"Expected some train samples, got {stats['train']}"
    # 28 classes - 2 skipped = 26 classes, max 1 each
    assert stats["train"] <= 26, f"Too many train samples with max_per_class=1: {stats['train']}"

    # Check output files
    assert (output_dir / "train_poses.npy").exists(), "train_poses.npy not created"
    assert (output_dir / "train_labels.npy").exists(), "train_labels.npy not created"
    assert (output_dir / "meta.json").exists(), "meta.json not created"

    # Load and verify shapes
    poses, labels = load_unified(output_dir, split="train")
    assert poses is not None, "load_unified returned None"

    assert isinstance(poses, list), "poses should be a list"
    assert len(poses) > 0, "poses list is empty"

    first_pose = poses[0]
    assert first_pose.ndim == 3, f"Expected 3D (T, 17, 2), got {first_pose.ndim}D"
    assert first_pose.shape[1] == 17, f"Expected 17 keypoints, got {first_pose.shape[1]}"
    assert first_pose.shape[2] == 2, f"Expected 2 channels, got {first_pose.shape[2]}"
    assert first_pose.dtype == np.float32, f"Expected float32, got {first_pose.dtype}"

    # Labels
    assert labels.ndim == 1, f"Expected 1D labels, got {labels.ndim}D"
    assert len(labels) == len(poses), f"Labels {len(labels)} != poses {len(poses)}"
    assert labels.dtype == np.int32, f"Expected int32, got {labels.dtype}"

    # No skipped class labels should be present
    for lbl in labels:
        assert lbl not in {12, 27}, f"Found skipped class label {lbl} in output"
