"""
Experiments 2b, 2c, 2d: Follow-up classification experiments.

2b: Center-crop + top-10 classes (isolate truncation/class imbalance)
2c: BiGRU with variable-length sequences (proper temporal modeling)
2d: MMFS dataset baseline (63 classes, quality scores)

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    uv run python experiments/exp_2b_2c_2d.py
"""

import pickle
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

BASE = Path("data/datasets")


# ─── Normalize ───────────────────────────────────────────────────────────────

def normalize(p: np.ndarray) -> np.ndarray:
    """Root-center + scale normalize. p: (F, 17, 2) float32. COCO indices."""
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)  # LHIP=11, RHIP=12
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)  # LSHOULDER=5, RSHOULDER=6
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_fsc(split: str):
    path = BASE / "figure-skating-classification"
    data = pickle.load(open(path / f"{split}_data.pkl", "rb"))
    labels = pickle.load(open(path / f"{split}_label.pkl", "rb"))
    poses = [normalize(np.array(d[:, :, :2, 0], dtype=np.float32)) for d in data]
    return poses, labels


def load_mmfs(split: str):
    path = BASE / "mmfs/MMFS/skeleton"
    data = pickle.load(open(path / f"{split}_data.pkl", "rb"))
    labels = pickle.load(open(path / f"{split}_label.pkl", "rb"))
    poses = [normalize(np.array(d[:, :, :2, 0], dtype=np.float32)) for d in data]
    return poses, labels


# ─── Dataset Classes ─────────────────────────────────────────────────────────

class PaddedDataset(Dataset):
    def __init__(self, poses, labels, max_len=150, crop="start"):
        self.poses, self.labels = poses, labels
        self.max_len, self.crop = max_len, crop

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p = self.poses[idx]
        F = len(p)
        if self.crop == "center" and F > self.max_len:
            s = (F - self.max_len) // 2
            p = p[s:s + self.max_len]
        elif F > self.max_len:
            p = p[:self.max_len]
        if len(p) < self.max_len:
            pad = np.zeros((self.max_len - len(p), 17, 2), dtype=np.float32)
            p = np.concatenate([p, pad])
        return torch.tensor(p.reshape(self.max_len, -1)), self.labels[idx]


class VarLenDataset(Dataset):
    def __init__(self, poses, labels):
        self.poses, self.labels = poses, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p = self.poses[idx]
        return torch.tensor(p.reshape(len(p), -1)), self.labels[idx]


def varlen_collate(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch])
    lengths = torch.tensor([s.shape[0] for s in seqs])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, lengths, labels


# ─── Models ──────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    def __init__(self, in_channels=34, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc(self.conv(x.permute(0, 2, 1)))


class BiGRU(nn.Module):
    def __init__(self, in_features=34, hidden=128, num_layers=2, num_classes=64):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h = self.gru(packed)
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))


# ─── Training Loops ─────────────────────────────────────────────────────────

def train_eval_cnn(model, train_loader, test_loader, device, epochs=30, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        # eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        acc = correct / total
        best_acc = max(best_acc, acc)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d}: test_acc={acc:.3f}  best={best_acc:.3f}")
    return best_acc


def train_eval_gru(model, train_loader, test_loader, device, epochs=30, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x, lengths), y); loss.backward(); opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, lengths, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x, lengths).argmax(1) == y).sum().item()
                total += len(y)
        acc = correct / total
        best_acc = max(best_acc, acc)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d}: test_acc={acc:.3f}  best={best_acc:.3f}")
    return best_acc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load FSC
    fsc_train_p, fsc_train_l = load_fsc("train")
    fsc_test_p, fsc_test_l = load_fsc("test")

    # ═══════════════════════════════════════════════════════════════════════
    # 2b: Center-crop + Top-10 Classes (FSC)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXP 2b: Center-crop + Top-10 Classes (FSC)")
    print("=" * 60)

    counts = Counter(fsc_train_l)
    top10 = [c for c, _ in counts.most_common(10)]
    cmap = {c: i for i, c in enumerate(top10)}

    tr_p = [p for p, l in zip(fsc_train_p, fsc_train_l) if l in cmap]
    tr_l = [cmap[l] for l in fsc_train_l if l in cmap]
    te_p = [p for p, l in zip(fsc_test_p, fsc_test_l) if l in cmap]
    te_l = [cmap[l] for l in fsc_test_l if l in cmap]

    lens = [len(p) for p in tr_p + te_p]
    print(f"Top-10 classes: {len(tr_p)} train, {len(te_p)} test")
    print(f"Seq lengths: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.0f}")
    print(f"Random baseline: {1/10:.1%}\n")

    results_2b = {}
    for crop in ["start", "center"]:
        print(f"--- {crop} crop, 150 frames ---")
        tr_ds = PaddedDataset(tr_p, tr_l, 150, crop)
        te_ds = PaddedDataset(te_p, te_l, 150, crop)
        tr_dl = DataLoader(tr_ds, 64, shuffle=True)
        te_dl = DataLoader(te_ds, 64)
        model = CNN1D(num_classes=10).to(device)
        t0 = time.time()
        acc = train_eval_cnn(model, tr_dl, te_dl, device)
        print(f"  Result: {acc:.1%} ({time.time()-t0:.0f}s)\n")
        results_2b[crop] = acc

    # ═══════════════════════════════════════════════════════════════════════
    # 2c: BiGRU Variable-Length (FSC, all 64 classes)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXP 2c: BiGRU Variable-Length (FSC, 64 classes)")
    print("=" * 60)

    # Filter empty sequences
    fsc_tr_valid = [(p, l) for p, l in zip(fsc_train_p, fsc_train_l) if len(p) > 0]
    fsc_te_valid = [(p, l) for p, l in zip(fsc_test_p, fsc_test_l) if len(p) > 0]
    fsc_train_p_v, fsc_train_l_v = zip(*fsc_tr_valid) if fsc_tr_valid else ([], [])
    fsc_test_p_v, fsc_test_l_v = zip(*fsc_te_valid) if fsc_te_valid else ([], [])

    all_labels = sorted(set(fsc_train_l_v + fsc_test_l_v))
    lmap = {l: i for i, l in enumerate(all_labels)}
    tr_l_all = [lmap[l] for l in fsc_train_l_v]
    te_l_all = [lmap[l] for l in fsc_test_l_v]

    print(f"Classes: {len(all_labels)}, Train: {len(fsc_train_p_v)}, Test: {len(fsc_test_p_v)}")
    print(f"Random baseline: {1/len(all_labels):.3%}\n")

    tr_ds = VarLenDataset(list(fsc_train_p_v), tr_l_all)
    te_ds = VarLenDataset(list(fsc_test_p_v), te_l_all)
    tr_dl = DataLoader(tr_ds, 64, shuffle=True, collate_fn=varlen_collate)
    te_dl = DataLoader(te_ds, 64, shuffle=False, collate_fn=varlen_collate)

    model = BiGRU(num_classes=len(all_labels)).to(device)
    t0 = time.time()
    result_2c = train_eval_gru(model, tr_dl, te_dl, device)
    print(f"  Result: {result_2c:.1%} ({time.time()-t0:.0f}s)\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 2d: MMFS Dataset Baseline
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXP 2d: MMFS Dataset Baseline")
    print("=" * 60)

    mmfs_tr_p, mmfs_tr_l = load_mmfs("train")
    mmfs_te_p, mmfs_te_l = load_mmfs("test")
    mmfs_tr_s = pickle.load(open(BASE / "mmfs/MMFS/skeleton/train_score.pkl", "rb"))
    mmfs_te_s = pickle.load(open(BASE / "mmfs/MMFS/skeleton/test_score.pkl", "rb"))

    # Filter empty sequences
    mmfs_tr = [(p, l) for p, l in zip(mmfs_tr_p, mmfs_tr_l) if len(p) > 0]
    mmfs_te = [(p, l) for p, l in zip(mmfs_te_p, mmfs_te_l) if len(p) > 0]
    mmfs_tr_p, mmfs_tr_l = zip(*mmfs_tr) if mmfs_tr else ([], [])
    mmfs_te_p, mmfs_te_l = zip(*mmfs_te) if mmfs_te else ([], [])

    print(f"Classes: {len(set(mmfs_tr_l))}, Train: {len(mmfs_tr_p)}, Test: {len(mmfs_te_p)}")
    print(f"Score range: {min(mmfs_tr_s):.1f} - {max(mmfs_tr_s):.1f}")
    lens = [len(p) for p in mmfs_tr_p + mmfs_te_p]
    print(f"Seq lengths: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.0f}")
    print(f"Random baseline: {1/63:.3%}\n")

    # Top-10 classes CNN
    counts_mmfs = Counter(mmfs_tr_l)
    top10_mmfs = [c for c, _ in counts_mmfs.most_common(10)]
    cmap_mmfs = {c: i for i, c in enumerate(top10_mmfs)}

    m_tr_p = [p for p, l in zip(mmfs_tr_p, mmfs_tr_l) if l in cmap_mmfs]
    m_tr_l = [cmap_mmfs[l] for l in mmfs_tr_l if l in cmap_mmfs]
    m_te_p = [p for p, l in zip(mmfs_te_p, mmfs_te_l) if l in cmap_mmfs]
    m_te_l = [cmap_mmfs[l] for l in mmfs_te_l if l in cmap_mmfs]

    print(f"Top-10 MMFS: {len(m_tr_p)} train, {len(m_te_p)} test\n--- CNN center-crop ---")
    tr_ds = PaddedDataset(m_tr_p, m_tr_l, 150, "center")
    te_ds = PaddedDataset(m_te_p, m_te_l, 150, "center")
    model = CNN1D(num_classes=10).to(device)
    t0 = time.time()
    result_2d_top10 = train_eval_cnn(model, DataLoader(tr_ds, 64, True), DataLoader(te_ds, 64), device)
    print(f"  Result: {result_2d_top10:.1%} ({time.time()-t0:.0f}s)\n")

    # All 63 classes BiGRU
    all_mmfs = sorted(set(mmfs_tr_l + mmfs_te_l))
    mmfs_lmap = {l: i for i, l in enumerate(all_mmfs)}
    mmfs_tr_l2 = [mmfs_lmap[l] for l in mmfs_tr_l]
    mmfs_te_l2 = [mmfs_lmap[l] for l in mmfs_te_l]

    print(f"--- BiGRU all {len(all_mmfs)} classes ---")
    tr_ds = VarLenDataset(mmfs_tr_p, mmfs_tr_l2)
    te_ds = VarLenDataset(mmfs_te_p, mmfs_te_l2)
    model = BiGRU(num_classes=len(all_mmfs)).to(device)
    t0 = time.time()
    result_2d_all = train_eval_gru(model, DataLoader(tr_ds, 64, True, collate_fn=varlen_collate),
                                     DataLoader(te_ds, 64, False, collate_fn=varlen_collate), device)
    print(f"  Result: {result_2d_all:.1%} ({time.time()-t0:.0f}s)\n")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exp 1  cosine sim gap:       0.090 (noise)")
    print(f"Exp 2  CNN 64cls start:      21.8%")
    print(f"Exp 2b CNN 10cls start:      {results_2b['start']:.1%}")
    print(f"Exp 2b CNN 10cls center:     {results_2b['center']:.1%}")
    print(f"Exp 2c BiGRU 64cls:          {result_2c:.1%}")
    print(f"Exp 2d MMFS CNN 10cls:       {result_2d_top10:.1%}")
    print(f"Exp 2d MMFS BiGRU 63cls:     {result_2d_all:.1%}")


if __name__ == "__main__":
    main()
