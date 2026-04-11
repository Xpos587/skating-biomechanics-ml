"""
Experiment: Data Augmentation for Figure Skating Element Classification.

Tests which augmentations help prevent overfitting on 64-class FSC dataset.
Uses BiGRU (best model from exp 2c/2d) with early stopping.

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    uv run python data/experiments/exp_augmentation.py
"""

import pickle
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path

BASE = Path("data/datasets")

# COCO 17kp left-right swap for mirroring
# 0=Nose, 1=L_EYE, 2=R_EYE, 3=L_EAR, 4=R_EAR, 5=L_SHOULDER, 6=R_SHOULDER,
# 7=L_ELBOW, 8=R_ELBOW, 9=L_WRIST, 10=R_WRIST, 11=L_HIP, 12=R_HIP,
# 13=L_KNEE, 14=R_KNEE, 15=L_ANKLE, 16=R_ANKLE
LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


# ─── Normalize ───────────────────────────────────────────────────────────────

def normalize(p: np.ndarray) -> np.ndarray:
    """Root-center + scale normalize. p: (F, 17, 2) float32."""
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


# ─── Augmentations ───────────────────────────────────────────────────────────

def aug_joint_noise(p, sigma=0.02):
    """Gaussian noise on joint coordinates."""
    return p + np.random.randn(*p.shape).astype(np.float32) * sigma


def aug_mirror(p):
    """Horizontal flip with left-right joint swap."""
    p = p.copy()
    p[:, :, 0] = -p[:, :, 0]  # flip x
    p = p[:, LR_SWAP, :]      # swap left/right
    return p


def aug_temporal_scale(p, scale_range=(0.9, 1.1)):
    """Resample sequence to random speed."""
    F = len(p)
    scale = random.uniform(*scale_range)
    new_F = int(F * scale)
    if new_F < 10:
        new_F = 10
    indices = np.linspace(0, F - 1, new_F).astype(int)
    return p[indices]


def aug_joint_drop(p, max_drop=2):
    """Zero out 1-2 random joints per frame (simulates occlusion)."""
    p = p.copy()
    n_drop = random.randint(1, max_drop)
    joints = random.sample(range(17), n_drop)
    for j in joints:
        p[:, j, :] = 0.0
    return p


def aug_skeleton_mix(p1, p2, alpha=0.3):
    """Blend two sequences from same class."""
    F = min(len(p1), len(p2))
    mixed = alpha * p1[:F] + (1 - alpha) * p2[:F]
    return mixed.astype(np.float32)


def augment(p, label, all_poses_by_label, p_mirror=0.5, p_noise=0.8, p_tscale=0.5, p_drop=0.3, p_mix=0.2):
    """Apply random augmentations. Returns list of (pose, label) tuples."""
    results = [(p, label)]

    # Mirror (with label-aware warning: some elements are direction-dependent)
    if random.random() < p_mirror:
        results.append((aug_mirror(p), label))

    # Joint noise
    if random.random() < p_noise:
        results.append((aug_joint_noise(p), label))

    # Temporal scaling
    if random.random() < p_tscale:
        results.append((aug_temporal_scale(p), label))

    # Joint dropping
    if random.random() < p_drop:
        results.append((aug_joint_drop(p), label))

    # SkeletonMix (only if same-class samples available)
    if random.random() < p_mix and label in all_poses_by_label and len(all_poses_by_label[label]) > 1:
        other = random.choice(all_poses_by_label[label])
        results.append((aug_skeleton_mix(p, other), label))

    return results


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_fsc(split: str):
    path = BASE / "figure-skating-classification"
    data = pickle.load(open(path / f"{split}_data.pkl", "rb"))
    labels = pickle.load(open(path / f"{split}_label.pkl", "rb"))
    poses = [normalize(np.array(d[:, :, :2, 0], dtype=np.float32)) for d in data]
    # Filter empty
    valid = [(p, l) for p, l in zip(poses, labels) if len(p) > 0]
    if not valid:
        return [], []
    poses, labels = zip(*valid)
    return list(poses), list(labels)


# ─── Dataset ─────────────────────────────────────────────────────────────────

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


# ─── Model ───────────────────────────────────────────────────────────────────

class BiGRU(nn.Module):
    def __init__(self, in_features=34, hidden=128, num_layers=2, num_classes=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden, num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h = self.gru(packed)
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))


# ─── Training with Early Stopping ────────────────────────────────────────────

def train_eval(model, train_loader, val_loader, test_loader, device,
               epochs=50, lr=1e-3, weight_decay=1e-4, patience=10):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float("inf")
    best_test_acc = 0.0
    best_state = None
    wait = 0
    history = []

    for ep in range(epochs):
        # Train
        model.train()
        train_loss, train_n = 0.0, 0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(y)
            train_n += len(y)
        sched.step()

        # Validate
        model.eval()
        val_loss, val_n, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                val_loss += crit(logits, y).item() * len(y)
                val_n += len(y)
                val_correct += (logits.argmax(1) == y).sum().item()
        val_acc = val_correct / val_n
        val_loss_avg = val_loss / val_n

        # Test
        test_correct, test_n = 0, 0
        with torch.no_grad():
            for x, lengths, y in test_loader:
                x, y = x.to(device), y.to(device)
                test_correct += (model(x, lengths).argmax(1) == y).sum().item()
                test_n += len(y)
        test_acc = test_correct / test_n

        history.append((ep + 1, train_loss / train_n, val_loss_avg, val_acc, test_acc))

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            gap = (train_loss / train_n) - val_loss_avg
            print(f"  Ep {ep+1:3d}: train={train_loss/train_n:.3f} val={val_loss_avg:.3f} "
                  f"gap={gap:+.3f} val_acc={val_acc:.3f} test_acc={test_acc:.3f} "
                  f"best_test={best_test_acc:.3f} lr={sched.get_last_lr()[0]:.5f}")

        if wait >= patience:
            print(f"  Early stop at epoch {ep+1} (no val improvement for {patience} epochs)")
            break

    # Restore best
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_test_acc, history


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda")
    print(f"Device: {device}\n")

    # Load data
    train_p, train_l = load_fsc("train")
    test_p, test_l = load_fsc("test")

    all_labels = sorted(set(train_l + test_l))
    lmap = {l: i for i, l in enumerate(all_labels)}
    train_l = [lmap[l] for l in train_l]
    test_l = [lmap[l] for l in test_l]

    # Split train into train/val (90/10, stratified)
    by_class = {}
    for p, l in zip(train_p, train_l):
        by_class.setdefault(l, []).append(p)
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for cls, samples in by_class.items():
        random.shuffle(samples)
        split = max(1, int(len(samples) * 0.1))
        va_p.extend(samples[:split])
        va_l.extend([cls] * split)
        tr_p.extend(samples[split:])
        tr_l.extend([cls] * (len(samples) - split))

    print(f"Classes: {len(all_labels)}")
    print(f"Train: {len(tr_p)}, Val: {len(va_p)}, Test: {len(test_p)}")
    print(f"Random baseline: {1/len(all_labels):.3%}\n")

    # Build index for SkeletonMix
    poses_by_label = {}
    for p, l in zip(tr_p, tr_l):
        poses_by_label.setdefault(l, []).append(p)

    def make_loaders(train_poses, train_labels, batch=64):
        tr_dl = DataLoader(VarLenDataset(train_poses, train_labels), batch, True, collate_fn=varlen_collate)
        va_dl = DataLoader(VarLenDataset(va_p, va_l), batch, False, collate_fn=varlen_collate)
        te_dl = DataLoader(VarLenDataset(test_p, test_l), batch, False, collate_fn=varlen_collate)
        return tr_dl, va_dl, te_dl

    results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # Baseline: No augmentation
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("BASELINE: No augmentation (BiGRU, early stopping)")
    print("=" * 65)
    model = BiGRU(num_classes=len(all_labels)).to(device)
    tr_dl, va_dl, te_dl = make_loaders(tr_p, tr_l)
    t0 = time.time()
    acc, hist = train_eval(model, tr_dl, va_dl, te_dl, device)
    results["baseline"] = (acc, hist)
    print(f"  >>> Best test: {acc:.1%} ({time.time()-t0:.0f}s)\n")

    # ═══════════════════════════════════════════════════════════════════════
    # Aug A: Noise + Mirror only (safe augmentations)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("AUG A: Noise (0.02) + Mirror")
    print("=" * 65)
    aug_p, aug_l = [], []
    for p, l in zip(tr_p, tr_l):
        aug_p.append(p)
        aug_l.append(l)
        if random.random() < 0.8:
            aug_p.append(aug_joint_noise(p))
            aug_l.append(l)
        if random.random() < 0.5:
            aug_p.append(aug_mirror(p))
            aug_l.append(l)
    print(f"  Augmented train: {len(aug_p)} ({len(aug_p)/len(tr_p):.1f}x)")
    model = BiGRU(num_classes=len(all_labels)).to(device)
    tr_dl, va_dl, te_dl = make_loaders(aug_p, aug_l)
    t0 = time.time()
    acc, hist = train_eval(model, tr_dl, va_dl, te_dl, device)
    results["noise+mirror"] = (acc, hist)
    print(f"  >>> Best test: {acc:.1%} ({time.time()-t0:.0f}s)\n")

    # ═══════════════════════════════════════════════════════════════════════
    # Aug B: Full pipeline (noise + mirror + tscale + drop + mix)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("AUG B: Full (noise + mirror + tscale + drop + SkeletonMix)")
    print("=" * 65)
    aug_p, aug_l = [], []
    for p, l in zip(tr_p, tr_l):
        augmented = augment(p, l, poses_by_label,
                            p_mirror=0.5, p_noise=0.8, p_tscale=0.5, p_drop=0.3, p_mix=0.2)
        for ap, al in augmented:
            aug_p.append(ap)
            aug_l.append(al)
    print(f"  Augmented train: {len(aug_p)} ({len(aug_p)/len(tr_p):.1f}x)")
    model = BiGRU(num_classes=len(all_labels)).to(device)
    tr_dl, va_dl, te_dl = make_loaders(aug_p, aug_l)
    t0 = time.time()
    acc, hist = train_eval(model, tr_dl, va_dl, te_dl, device)
    results["full"] = (acc, hist)
    print(f"  >>> Best test: {acc:.1%} ({time.time()-t0:.0f}s)\n")

    # ═══════════════════════════════════════════════════════════════════════
    # Aug C: Conservative (noise + mirror only, lower lr, more dropout)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("AUG C: Conservative (noise+mirror, dropout=0.4, wd=1e-3)")
    print("=" * 65)
    aug_p, aug_l = [], []
    for p, l in zip(tr_p, tr_l):
        aug_p.append(p); aug_l.append(l)
        aug_p.append(aug_joint_noise(p, sigma=0.015)); aug_l.append(l)
        aug_p.append(aug_mirror(p)); aug_l.append(l)
    print(f"  Augmented train: {len(aug_p)} ({len(aug_p)/len(tr_p):.1f}x)")
    model = BiGRU(num_classes=len(all_labels), dropout=0.4).to(device)
    tr_dl, va_dl, te_dl = make_loaders(aug_p, aug_l)
    t0 = time.time()
    acc, hist = train_eval(model, tr_dl, va_dl, te_dl, device, lr=5e-4, weight_decay=1e-3)
    results["conservative"] = (acc, hist)
    print(f"  >>> Best test: {acc:.1%} ({time.time()-t0:.0f}s)\n")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for name, (acc, hist) in results.items():
        last = hist[-1]
        overfit_gap = last[1] - last[2]  # train_loss - val_loss
        print(f"  {name:20s}: test={acc:.1%}  epochs={len(hist)}  final_train_val_gap={overfit_gap:+.3f}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
