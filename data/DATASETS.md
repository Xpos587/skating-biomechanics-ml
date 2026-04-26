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
│   ├── figure-skating-classification/  # FSC source (pkl) + merged MMFS quality scores
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

**FSC = MMFS + mocap.** MMFS quality scores merged into FSC (`train_scores.npy`, `test_scores.npy`). MMFS directory deleted — data is redundant (verified: first sample bit-identical, labels match exactly for first 3959 train samples).

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
| Quality scores | `train_scores.npy`, `test_scores.npy` — merged from MMFS (NaN for 253 mocap samples) |

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
| **FineFS** | ✅ Downloaded (Vast.ai) | 1167 samples, 3D poses + GOE scores + boundaries |

---

## FineFS Dataset Details

**Location:** `/root/data/datasets/raw/FineFS/data/` (Vast.ai GPU server)

**Source:** [arXiv:2305.14372](https://arxiv.org/abs/2305.14372) — "Fine-grained Figure Skating Dataset" (GitHub link dead, downloaded via alternative source)

### Structure

```
FineFS/data/
├── skeleton/        - 1167 NPZ files (3D poses, 829MB)
├── annotation/      - 1167 JSON files (element labels, 1.1MB)
├── video_features/  - 1167 PKL files (848MB)
└── video.zip        - 41GB (raw videos, not extracted)
```

### Skeleton Format (NPZ)

**File:** `skeleton/{id}.npz`
- **Key:** `reconstruction`
- **Shape:** `(T, 17, 3)` where T = num frames (~4300)
- **Dtype:** `float32`
- **Format:** H3.6M 17-keypoint 3D skeleton ✅
- **Coordinates:** 3D world coordinates (meters), NOT normalized
  - X range: `[-0.817, 0.887]` meters
  - Y range: `[-0.704, 0.708]` meters
  - Z range: `[0.000, 1.802]` meters (vertical)

**Keypoint mapping (H3.6M 17kp):**
```
0: Hip (center)
1-3: Right leg (hip, knee, ankle)
4-6: Left leg (hip, knee, ankle)
7-10: Spine + neck + head
11-13: Left arm (shoulder, elbow, wrist)
14-16: Right arm (shoulder, elbow, wrist)
```

**Occlusion handling:**
- `Z=0` indicates occlusion/missing keypoints
- Keypoints 3, 6 (ankles) have ~50% Z-zero (often occluded in skating)

### Annotation Format (JSON)

**File:** `annotation/{id}.json`

**Fields:**
- `competition`: str (e.g., "WC2019_Men")
- `total_segment_score`: float
- `element_score`: list[float]
- `executed_element`: dict[element_data]

**Element structure:**
```json
{
  "element1": {
    "element": "3A",
    "time": "0-22,0-25",
    "coarse_class": "jump",
    "goe": 1.6,
    "score_of_pannel": 9.6,
    "bv": 8.0,
    "judge_score": [2, 2, 2, 2, 1, 2, 2, 2, 2],
    "info": ["<<"]
  }
}
```

**Timing format:** `MM-SS,MM-SS` (start_time, end_time)
- Example: `"0-22,0-25"` = 0:22 to 0:25 (3 seconds)
- FPS: ~30 FPS (4350 frames ≈ 145 seconds)

**Element types:**
- `jump`: 3A (Axel), 4T (Toe loop), 3Lz+3T (Lutz+Toe combo)
- `spin`: FCSp3 (sit spin), CSSp4 (camel spin), CCoSp3 (spin)
- `sequence`: StSq2 (step sequence)

### Conversion to YOLO

**Requirements:**
1. ✅ **Already H3.6M 17kp** — no keypoint remapping needed!
2. ❌ **Normalize coordinates** to `[0,1]`
   - Current: world coords in meters
   - Strategy: divide by max(X,Y) range or use bounding box normalization
3. ❌ **Generate bounding boxes** around skeleton
4. ❌ **Convert Z=0 to visibility flags** (confidence)
5. ❌ **Extract element segments** from full sequence using timing

**YOLO-pose format:**
```
class_id x_center y_center width height k1x k1y k1v ... k17x k17y k17v
```

### Dataset Size

- **Samples:** 1167
- **Frames per sample:** ~4300 (≈ 143 seconds at 30 FPS)
- **Elements per sample:** 5-8
- **Estimated total elements:** ~7000

### Usage Example

```python
import numpy as np
import json

# Load skeleton
skeleton_data = np.load("skeleton/0.npz")
poses = skeleton_data["reconstruction"]  # (T, 17, 3)

# Load annotation
with open("annotation/0.json") as f:
    annotation = json.load(f)

# Parse element timing
def parse_time(time_str):
    """Parse 'MM-SS,MM-SS' to frame indices"""
    start_str, end_str = time_str.split(",")
    start_min, start_sec = map(int, start_str.split("-"))
    end_min, end_sec = map(int, end_str.split("-"))
    start_frame = start_min * 60 + start_sec
    end_frame = end_min * 60 + end_sec
    return start_frame, end_frame

# Extract element segment
for elem_key, elem_data in annotation["executed_element"].items():
    start, end = parse_time(elem_data["time"])
    element_poses = poses[start:end]  # Segment poses
    element_type = elem_data["element"]
    goe_score = elem_data["goe"]
``` |
