# Dataset Registry

Single source of truth for all datasets. See `research/RESEARCH.md` for research context.

## Directory Structure

```
data/datasets/
├── unified/                   # Canonical format (converted, ready for training)
│   ├── fsc-64/               # FSC 64 classes (4046 train, 985 test)
│   ├── mcfs-129/             # MCFS 129 classes (2617 samples, no split)
│   └── skatingverse-28/      # SkatingVerse 28 classes (pending extraction)
├── raw/                       # Original files (read-only, do not modify)
│   ├── figure-skating-classification/  # FSC source (pkl)
│   ├── mmfs/                           # MMFS (redundant, subset of FSC)
│   ├── mcfs/                           # MCFS (segments.pkl + splits)
│   ├── skatingverse/                   # SkatingVerse (mp4 videos, 46GB)
│   ├── athletepose3d/                  # AthletePose3D (71GB, 12 sports)
│   └── finefs/                         # FineFS (quality scores)
└── checkpoints/               # Model checkpoints
```

## Unified Format

All datasets converted to numpy npy format by converters in `data/data_tools/`.

**Skeleton:** `(T, 17, 2)` float32 — COCO/H3.6M 17 keypoints, xy only.
**Normalization:** Root-centered (mid-hip), spine-length scaled.
**Storage:** `{split}_poses.npy` (object array of variable-length), `{split}_labels.npy` (int32), `meta.json`.

| Dataset | Classes | Samples | Split | Converter |
|---------|---------|---------|-------|-----------|
| FSC | 64 | 4,046 train + 985 test | train/test | `convert_fsc` |
| MCFS | 129 | 2,617 | all | `convert_mcfs` |
| SkatingVerse | 28 | ~28K (pending) | train/test | `convert_skatingverse` |

## Relationships

```
MMFS (4,915 seq, 63 cls) + mocap (253 seq) ──merged into──> FSC (5,168 seq, 64 cls)
SkatingVerse (28 cls) ──~90% overlap──> FSC (64 cls)
MCFS (129 cls) ──different granularity──> not directly mappable to FSC
AthletePose3D (12 sports) ──contains──> FS-Jump3D (figure skating subset)
```

**FSC = MMFS + mocap.** MMFS stored in `raw/mmfs/` for reference but is redundant.

## PyTorch Dataset

```python
from data_tools.dataset import UnifiedSkatingDataset, varlen_collate
from torch.utils.data import DataLoader

# FSC
ds = UnifiedSkatingDataset("data/datasets/unified/fsc-64", split="train", augment=True)
loader = DataLoader(ds, batch_size=64, collate_fn=varlen_collate, shuffle=True)

# MCFS
ds = UnifiedSkatingDataset("data/datasets/unified/mcfs-129", split="all")
```

Augmentation: Gaussian noise (80%), mirror flip (50%), temporal scale (50%), joint drop (30%).

## Label Ontology

See `data/data_tools/label_ontology.py` for unified label mappings across FSC (64), MCFS (129), SkatingVerse (28).

## Preprocessing Pipeline

```bash
PYTHONPATH=data uv run python -c "
from data_tools.convert_fsc import convert_fsc
from pathlib import Path
convert_fsc(Path('data/datasets/raw/figure-skating-classification'), Path('data/datasets/unified/fsc-64'))
"

PYTHONPATH=data uv run python -c "
from data_tools.convert_mcfs import convert_mcfs
from pathlib import Path
convert_mcfs(Path('data/datasets/raw/mcfs'), Path('data/datasets/unified/mcfs-129'))
"

# SkatingVerse requires GPU + ~40h (run on vast.ai)
PYTHONPATH=data uv run python -c "
from data_tools.convert_skatingverse import convert_skatingverse
from pathlib import Path
convert_skatingverse(Path('data/datasets/raw/skatingverse'), Path('data/datasets/unified/skatingverse-28'))
"
```

## Raw Dataset Details

### Figure-Skating-Classification (FSC)

| Field | Value |
|-------|-------|
| Source | [HuggingFace](https://huggingface.co/datasets/Mercity/Figure-Skating-Classification-Data) |
| Size | 340MB |
| Format | pkl: `(150, 17, 3, 1)` float32, normalized [-1,1] |
| Classes | 64: single jumps (0-20), combos (21-30), spins (31-58), steps (59-63) |

### MCFS

| Field | Value |
|-------|-------|
| Paper | AAAI 2021 — "Temporal Segmentation of Fine-grained Semantic Action" |
| Size | 103MB (segments.pkl) |
| Format | pkl: list of `(poses, label_str)` — poses `(T, 17, 2)` float32 |
| Classes | 129 unique string labels (132 in mapping.txt, 3 = NONE) |
| Splits | 5-fold CV bundles in `splits/` (not mapped to segments) |

### SkatingVerse

| Field | Value |
|-------|-------|
| Paper | ECCV 2024 Workshop — IET Computer Vision |
| Size | 46GB (mp4 videos) |
| Format | mp4 → RTMPose extraction (HALPE26 → COCO 17kp) |
| Classes | 28: skip class 12 (NoBasic), 27 (Sequence, ~25s videos) |
| Note | train.txt filenames lack .mp4 extension; answer.txt has no labels |

### AthletePose3D

| Field | Value |
|-------|-------|
| Paper | CVPR 2025 Workshop (CVSports) — [arXiv:2503.07499](https://arxiv.org/abs/2503.07499) |
| Size | 71GB (5154 videos, 12 sports) |
| Format | 3D/2D annotations, 24-camera multi-view |
| Note | Contains FS-Jump3D as subset. Not yet converted to unified format. |

## Keypoint Convention

All datasets use COCO 17kp (equivalent to H3.6M body keypoints). Our pipeline's `H36Key` enum matches this.

COCO 17: nose, L/R eye, L/R ear, L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle.

## Not Downloaded / Pending

| Dataset | Status | Notes |
|---------|--------|-------|
| YourSkatingCoach (2024) | Not downloaded | BIOES annotations, supplementary only |
| FSBench (CVPR 2025) | Temporarily closed | 783 videos, 76+ hours |
| FineFS | Downloaded | Quality scores + boundaries |
