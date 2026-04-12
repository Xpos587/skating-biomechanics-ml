"""Unified PyTorch Dataset for figure skating skeleton data.

Loads from the numpy npy format produced by the converters
(convert_fsc, convert_mcfs, convert_skatingverse).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# COCO 17-keypoint left-right swap indices
LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class UnifiedSkatingDataset(Dataset):
    """Variable-length skeleton dataset from unified npy files.

    Args:
        data_dir: Path to unified dataset directory (e.g. unified/fsc-64/).
            Must contain {split}_poses.npy and {split}_labels.npy.
        split: "train", "test", or "all" (MCFS uses "all").
        augment: Apply data augmentation (noise + mirror). Only effective for train.
        min_frames: Skip samples with fewer frames.
        label_map: Optional dict mapping source label IDs to different IDs.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        augment: bool = False,
        min_frames: int = 5,
        label_map: dict[int, int] | None = None,
    ):
        data_dir = Path(data_dir)

        poses_path = data_dir / f"{split}_poses.npy"
        labels_path = data_dir / f"{split}_labels.npy"

        if not poses_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Split '{split}' not found in {data_dir}. "
                f"Expected {poses_path.name} and {labels_path.name}"
            )

        poses_obj = np.load(poses_path, allow_pickle=True)
        labels_arr = np.load(labels_path)

        # Filter by min_frames and build valid indices
        self.poses: list[np.ndarray] = []
        self.labels: list[int] = []

        for i in range(len(poses_obj)):
            p = poses_obj[i]
            if p.shape[0] < min_frames:
                continue
            self.poses.append(p)
            self.labels.append(int(labels_arr[i]))

        if label_map:
            self.labels = [label_map.get(lbl, lbl) for lbl in self.labels]

        self.augment = augment and split == "train"
        self._poses_by_label: dict[int, list[np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        poses = self.poses[idx].astype(np.float32)  # (T, 17, 2)
        label = self.labels[idx]

        if self.augment:
            poses = self._augment(poses, label)

        # Flatten: (T, 17, 2) -> (T, 34)
        return torch.from_numpy(poses.reshape(len(poses), -1)), label

    def _augment(self, poses: np.ndarray, label: int) -> np.ndarray:
        """Random augmentation: noise + mirror + temporal scale + joint drop."""
        results = [poses]

        # Gaussian noise (80% chance)
        if random.random() < 0.8:
            noisy = poses + np.random.randn(*poses.shape).astype(np.float32) * 0.02
            results.append(noisy)

        if random.random() < 0.5:
            mirrored = poses.copy()
            mirrored[:, :, 0] = -mirrored[:, :, 0]
            mirrored = mirrored[:, LR_SWAP, :]
            results.append(mirrored)

        # Temporal scale (50% chance)
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            T = len(poses)
            new_T = max(10, int(T * scale))
            indices = np.linspace(0, T - 1, new_T).astype(int)
            results.append(poses[indices])

        # Joint drop (30% chance)
        if random.random() < 0.3:
            dropped = poses.copy()
            n_drop = random.randint(1, 2)
            joints = random.sample(range(17), n_drop)
            for j in joints:
                dropped[:, j, :] = 0.0
            results.append(dropped)

        return random.choice(results)

    def build_label_index(self) -> None:
        """Build index of poses grouped by label (needed for SkeletonMix)."""
        self._poses_by_label = {}
        for p, lbl in zip(self.poses, self.labels, strict=True):
            self._poses_by_label.setdefault(lbl, []).append(p)


def varlen_collate(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences with padding.

    Returns:
        padded: (B, max_T, C) float32
        lengths: (B,) int64
        labels: (B,) int64
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, lengths, labels
