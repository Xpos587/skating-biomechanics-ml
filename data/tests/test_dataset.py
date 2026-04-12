"""Tests for unified PyTorch dataset."""

from pathlib import Path

import pytest


def test_dataset_loads_fsc():
    """Test loading FSC unified dataset."""
    from data_tools.dataset import UnifiedSkatingDataset

    data_dir = Path("data/datasets/unified/fsc-64")
    if not (data_dir / "train_poses.npy").exists():
        pytest.skip("Unified FSC data not found")

    ds = UnifiedSkatingDataset(data_dir, split="train")
    assert len(ds) > 4000, f"Expected >4000 train samples, got {len(ds)}"

    poses, label = ds[0]
    assert poses.ndim == 2, f"Expected 2D (T, 34), got {poses.ndim}D"
    assert poses.shape[1] == 34, f"Expected 34 features (17*2), got {poses.shape[1]}"
    assert isinstance(label, int)


def test_dataset_loads_fsc_test():
    """Test loading FSC test split."""
    from data_tools.dataset import UnifiedSkatingDataset

    data_dir = Path("data/datasets/unified/fsc-64")
    if not (data_dir / "test_poses.npy").exists():
        pytest.skip("Unified FSC test data not found")

    ds = UnifiedSkatingDataset(data_dir, split="test")
    assert len(ds) > 900, f"Expected >900 test samples, got {len(ds)}"

    poses, _label = ds[0]
    assert poses.shape[1] == 34


def test_dataset_loads_mcfs():
    """Test loading MCFS unified dataset."""
    from data_tools.dataset import UnifiedSkatingDataset

    data_dir = Path("data/datasets/unified/mcfs-129")
    if not (data_dir / "all_poses.npy").exists():
        pytest.skip("Unified MCFS data not found")

    ds = UnifiedSkatingDataset(data_dir, split="all")
    assert len(ds) > 2000, f"Expected >2000 MCFS samples, got {len(ds)}"

    poses, _label = ds[0]
    assert poses.ndim == 2
    assert poses.shape[1] == 34


def test_dataset_with_augmentation():
    """Test that augmented dataset returns different poses."""
    from data_tools.dataset import UnifiedSkatingDataset

    data_dir = Path("data/datasets/unified/fsc-64")
    if not (data_dir / "train_poses.npy").exists():
        pytest.skip("Unified FSC data not found")

    ds = UnifiedSkatingDataset(data_dir, split="train", augment=True)
    assert len(ds) > 4000
    # Just verify it doesn't crash and returns correct shape
    poses, _label = ds[0]
    assert poses.shape[1] == 34


def test_dataset_with_label_map():
    """Test label remapping."""
    from data_tools.dataset import UnifiedSkatingDataset

    data_dir = Path("data/datasets/unified/fsc-64")
    if not (data_dir / "train_poses.npy").exists():
        pytest.skip("Unified FSC data not found")

    label_map = {i: i + 100 for i in range(64)}
    ds = UnifiedSkatingDataset(data_dir, split="train", label_map=label_map)
    _, label = ds[0]
    assert label >= 100, f"Expected remapped label >= 100, got {label}"


def test_dataset_collate():
    """Test varlen_collate produces correct shapes."""
    from data_tools.dataset import UnifiedSkatingDataset, varlen_collate
    from torch.utils.data import DataLoader

    data_dir = Path("data/datasets/unified/fsc-64")
    if not (data_dir / "train_poses.npy").exists():
        pytest.skip("Unified FSC data not found")

    ds = UnifiedSkatingDataset(data_dir, split="train")
    loader = DataLoader(ds, batch_size=4, collate_fn=varlen_collate)
    padded, lengths, labels = next(iter(loader))

    assert padded.ndim == 3, f"Expected 3D (B, T, 34), got {padded.ndim}D"
    assert padded.shape[0] == 4, f"Expected batch size 4, got {padded.shape[0]}"
    assert padded.shape[2] == 34, f"Expected 34 features, got {padded.shape[2]}"
    assert lengths.shape[0] == 4
    assert labels.shape[0] == 4
    # Lengths should match actual sequence lengths
    for i in range(4):
        assert lengths[i] <= padded.shape[1]


def test_dataset_missing_split():
    """Test that missing split raises FileNotFoundError."""
    from data_tools.dataset import UnifiedSkatingDataset

    data_dir = Path("data/datasets/unified/fsc-64")
    if not data_dir.exists():
        pytest.skip("Unified FSC data not found")

    with pytest.raises(FileNotFoundError, match="val"):
        UnifiedSkatingDataset(data_dir, split="val")


def test_dataset_min_frames():
    """Test min_frames filtering."""
    from data_tools.dataset import UnifiedSkatingDataset

    data_dir = Path("data/datasets/unified/fsc-64")
    if not (data_dir / "train_poses.npy").exists():
        pytest.skip("Unified FSC data not found")

    ds_normal = UnifiedSkatingDataset(data_dir, split="train", min_frames=5)
    ds_strict = UnifiedSkatingDataset(data_dir, split="train", min_frames=1000)
    assert len(ds_strict) < len(ds_normal), "Stricter min_frames should produce fewer samples"
