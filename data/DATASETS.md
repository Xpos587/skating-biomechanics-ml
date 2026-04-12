# Dataset Registry

Single source of truth for all datasets in `data/datasets/`. See `research/RESEARCH.md` for research context.

## Relationships

```
FS-Jump3D (2024) ──subset of──> AthletePose3D (2025, CVSports)
                                     │
                                     ├── MMFS (2023) ──skeleton data used in──> Figure-Skating-Classification (HuggingFace)
                                     │
                                     └── YourSkatingCoach (2024) — BIOES annotations (not downloaded)
```

**Key:** FS-Jump3D is integrated into AthletePose3D by the same research group (Tanaka, Fujii, Nagoya University). Do NOT download both — AthletePose3D is the superset.

---

## Active Datasets

### 1. AthletePose3D (CVSports 2025)

| Field | Value |
|-------|-------|
| **Paper** | "AthletePose3D: A Benchmark Dataset for 3D Human Pose Estimation and Kinematic Validation in Athletic Movements" |
| **arXiv** | [2503.07499](https://arxiv.org/abs/2503.07499) (v3, Jul 2025) |
| **Venue** | CVPR 2025 Workshop — 11th IEEE Int'l Workshop on Computer Vision in Sports (CVSports) |
| **PDF** | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Yeung_AthletePose3D_A_Benchmark_Dataset_for_3D_Human_Pose_Estimation_and_CVPRW_2025_paper.pdf) |
| **Repo** | https://github.com/calvinyeungck/AthletePose3D |
| **Authors** | Calvin Yeung, Tomohiro Suzuki, Ryota Tanaka, Zhuoer Yin, Keisuke Fujii (Nagoya University / RIKEN) |
| **Size** | 71GB (5154 videos, 1.3M frames, 997K annotations) |
| **License** | Non-commercial research only |
| **Status** | Downloaded |
| **Local path** | `data/datasets/athletepose3d/` |

```bibtex
@misc{yeung2025athletepose3d,
    title={AthletePose3D: A Benchmark Dataset for 3D Human Pose Estimation and Kinematic Validation in Athletic Movements},
    author={Yeung, Calvin and Suzuki, Tomohiro and Tanaka, Ryota and Yin, Zhuoer and Fujii, Keisuke},
    year={2025},
    eprint={2503.07499},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

**What it contains:**
- 12 sports: Axel, Flip, Lutz, Loop, Salchow, Toeloop, Spin, Running, Discus, Javelin, Shot, Glide, Comb
- 5 athletes (S1-S5), multiple moves per athlete, 12 cameras per move
- Videos: 1920x1088, 60fps, `.mp4`
- 3D annotations in H3.6M 17kp format (camera + image space)
- 2D annotations in COCO 17kp format (clean)
- 24 camera intrinsic/extrinsic matrices

**Directory structure:**
```
athletepose3d/
├── cam_param.json                    # 24 cameras, intrinsic + extrinsic
├── annotations_3d/
│   └── pose_3d_v3/
│       ├── train.pkl                 # 844MB, 703,720 annotations
│       ├── valid.pkl                 # 352MB, 293,941 annotations
│       └── frame_81/                 # Per-frame splits (81-frame windows)
│           ├── train/
│           └── test/
├── pose_2d/
│   └── pose_2d/
│       ├── annotations/              # COCO 17kp 2D annotations
│       ├── det_result/               # YOLOv8 detection results
│       ├── train_set/                # 2D pose images
│       ├── valid_set/
│       └── test_set/
└── videos/
    ├── train_set/                    # S1-S5 athletes
    ├── valid_set/                    # S1-S3 athletes
    └── test_set/                     # S1-S3 athletes
```

**Annotation format (per sample):**
```python
{
    "videoid": int,                    # Video ID
    "cameraid": str,                   # e.g. "fs_camera_1"
    "imageid": int,                    # Frame ID
    "image_path": str,                 # e.g. "pose_3d/train_img/S1_Salchow_4_cam_1_0.jpg"
    "subject": str,                    # e.g. "S1"
    "action": str,                     # "fs" (figure skating)
    "subaction": str,                  # e.g. "Salchow_4"
    "video_width": 1920,
    "video_height": 1088,
    "fps": 60.0,
    "joint_3d_image": ndarray(17, 3), # 3D joints in image space
    "joint_3d_camera": ndarray(17, 3), # 3D joints in camera space
    "center": tuple,                   # Bounding box center
    "scale": tuple,                    # Bounding box scale
    "box": ndarray(4),                 # [x1, y1, x2, y2]
    "root_depth": float,               # Root joint depth
    "camera_param": dict,              # Camera calibration for this frame
}
```

**Subsumes:** FS-Jump3D (Tanaka et al. 2024) — same data, extended to 12 sports.

**Caveat:** `coco_annotations/` was deleted because it contained a failed attempt to extend 17 COCO kp to 26 HALPE26 kp. Only 18/26 were valid, rest were zeros. Use `pose_2d/pose_2d/annotations/` for clean 17kp data.

**Usage in our project:**
- VIFSS contrastive pre-training (virtual camera projections from 3D poses)
- 3D pose normalization pipeline development
- Camera parameter reference for CorrectiveLens

---

### 2. MMFS (Multi-modality Multi-task Figure Skating, 2023)

| Field | Value |
|-------|-------|
| **Paper** | "Fine-grained Action Analysis: A Multi-modality and Multi-task Dataset of Figure Skating" |
| **arXiv** | [2307.02730](https://arxiv.org/abs/2307.02730) (v3, Apr 2024) |
| **Venue** | arXiv preprint (not formally published at a conference) |
| **Repo** | https://github.com/dingyn-Reno/MMFS |
| **Authors** | Sheng-Lan Liu, Yu-Ning Ding, Gang Yan, Si-Fan Zhang, Jin-Rong Zhang, Wen-Yue Chen, Xue-Hai Xu (Dalian University of Technology) |
| **Contact** | liusl@dlut.edu.cn |
| **Size** | 2.1GB (skeleton data only) |
| **License** | MIT |
| **Status** | Downloaded and extracted |
| **Local path** | `data/datasets/mmfs/MMFS/` |

```bibtex
@misc{liu2023finegrained,
    title={Fine-grained Action Analysis: A Multi-modality and Multi-task Dataset of Figure Skating},
    author={Liu, Sheng-Lan and Ding, Yu-Ning and Yan, Gang and Zhang, Si-Fan and Zhang, Jin-Rong and Chen, Wen-Yue and Xu, Xue-Hai},
    year={2023},
    eprint={2307.02730},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

**What it contains:**
- 4,915 skeleton sequences (3,959 train, 956 test)
- 63 unique action categories (MMFS63)
- COCO 17kp skeleton data, raw pixel coordinates
- Quality scores per action
- Full competition routines (variable length: 4,929-8,219 frames)
- 99 unique performers

**Directory structure:**
```
mmfs/
├── MMFS/
│   ├── readme.txt                     # Chinese README
│   ├── readme_en.txt                  # English README
│   └── skeleton/
│       ├── n01_p02.npy                # Per-person per-performance skeleton
│       ├── n01_p03.npy                #   (nXX = person, pYY = performance)
│       ├── ...                        # 1,178 .npy files total
│       ├── train_data.pkl             # 3,959 training sequences (list of arrays)
│       ├── train_label.pkl            # Category labels (0-62)
│       ├── train_name.pkl             # Action names
│       ├── train_score.pkl            # Quality scores
│       ├── test_data.pkl              # 956 test sequences
│       ├── test_label.pkl             # Category labels
│       ├── test_name.pkl              # Action names
│       └── test_score.pkl             # Quality scores
```

**Data format:**
```python
# Individual .npy file:
shape = (num_frames, 17, 3)   # frames x keypoints x (x, y, confidence)
dtype = float32
# x, y: pixel coordinates (not normalized)
# confidence: 0.0-1.0

# train_data.pkl: list of numpy arrays, variable length
# train_label.pkl: numpy array of ints (0-62)
# train_name.pkl: list of action name strings
# train_score.pkl: list of float scores
```

**Action categories (63 classes):**
Jumps (with rotation count): 1-4 Axel, Flip, Lutz, Loop, Salchow, Toeloop
Spins: CSp, USp, SSp, LSp, CoSp, CCSp
Sequences: ChoreoSequence, StepSequence
Combinations: 3Lz+3T, 3F+2T+2Lo, etc.

**Usage in our project:**
- VIFSS fine-tuning for element classification (replaces SkatingVerse)
- Quality score prediction (action assessment)
- COCO 17kp → H3.6M 17kp mapping via `src/ml/vifss/keypoint_map.py`

---

### 3. Figure-Skating-Classification (HuggingFace, 2026)

| Field | Value |
|-------|-------|
| **Paper** | No dedicated paper — compiled/derived dataset |
| **Source paper** | MMFS: [arXiv:2307.02730](https://arxiv.org/abs/2307.02730) (Liu et al., 2023) |
| **HF Link** | https://huggingface.co/datasets/Mercity/Figure-Skating-Classification-Data |
| **Author** | Mercity (mercity.ai, HuggingFace user) |
| **Size** | 340MB |
| **License** | MIT |
| **Status** | Downloaded |
| **Local path** | `data/datasets/figure-skating-classification/` |

> **Note:** This is a compiled dataset (MMFS + proprietary mocap), not an original research contribution. Cite the MMFS paper (arXiv:2307.02730) when using this data. The 253 mocap sequences have no public paper or source attribution.

**What it contains:**
- 5,168 skeleton sequences (4,161 train, 1,007 test)
- 64 element classes (single jumps, combinations, spins, step sequences)
- COCO 17kp, 150 frames per sequence, normalized coordinates
- Sources: MMFS (4,915 seq) + professional mocap (253 seq)
- Multi-jump combinations preserved (2A+3T, 3Lz+3T, etc.)

**Directory structure:**
```
figure-skating-classification/
├── README.md                        # Full documentation
├── dataset_info.json                # Metadata
├── label_mapping.json               # Class ID → name mapping
├── train_data.pkl                   # 4,161 sequences (list of arrays)
├── train_label.pkl                  # Class labels (0-63)
├── test_data.pkl                    # 1,007 sequences
└── test_label.pkl                   # Class labels
```

**Data format:**
```python
# train_data.pkl: list of numpy arrays
shape = (150, 17, 3)                # frames x keypoints x (x, y, confidence)
dtype = float32
# Coordinates normalized to [-1, 1] range (unlike raw MMFS)

# train_label.pkl: numpy array of ints (0-63)
# label_mapping.json: {"mmfs_labels": [0..63], "total_classes": 64}
```

**Classes (64 total):**
| Range | Category | Examples |
|-------|----------|----------|
| 0-20 | Single jumps | 1Axel, 2Flip, 3Lutz, 4Toeloop |
| 21-30 | Multi-jump combos | 1A+3T, 2A+3T, 3F+3T, 3Lz+3Lo, Comb |
| 31-58 | Spins | FCSp, CCoSp, ChCamelSp, ChSitSp, FlySitSp, LaybackSp |
| 59-63 | Steps & choreo | StepSeq1-4, ChoreSeq1 |

**Preprocessing applied:**
1. Format unification: 142-marker mocap → 17-keypoint COCO
2. Temporal uniform sampling to 150 frames
3. Coordinate normalization to [-1, 1]
4. Velocity feature computation
5. 80/20 stratified train/test split

**Usage in our project:**
- Alternative to raw MMFS for fine-tuning (pre-processed, cleaner)
- 64-class element classification
- Multi-jump combination detection
- Faster to load (normalized, fixed 150 frames)

---

## Keypoint Format Mapping

All three datasets use different conventions. Our pipeline uses H3.6M 17kp.

| Dataset | Keypoints | Format | Coordinates | Mapping needed |
|---------|-----------|--------|-------------|----------------|
| **AthletePose3D** | H3.6M 17kp (standard MMPose) | 3D (camera + image) | mm (world) | Direct use (our `H36Key` = MMPose, but different index order) |
| **MMFS** | COCO 17kp | 2D + confidence | pixels (raw) | COCO 17 → H3.6M 17 index remap |
| **Figure-Skating-Classification** | COCO 17kp | 2D + confidence | normalized [-1,1] | COCO 17 → H3.6M 17 index remap |

**COCO 17 → H3.6M 17 mapping** (already implemented in `src/pose_estimation/halpe26.py`):
```
COCO:  0=nose, 1=Leye, 2=Reye, 3=Lear, 4=Rear, 5=Lshoulder, 6=Rshoulder,
       7=Lelbow, 8=Relbow, 9=Lwrist, 10=Rwrist, 11=Lhip, 12=Rhip,
       13=Lknee, 14=Rknee, 15=Lankle, 16=Rankle

H3.6M: 0=Pelvis, 1=RHip, 2=RKnee, 3=RFoot, 4=LHip, 5=LKnee, 6=LFoot,
       7=Spine, 8=Thorax, 9=Neck, 10=Head, 11=LShoulder, 12=LElbow,
       13=LWrist, 14=RShoulder, 15=RElbow, 16=RWrist
```

---

## Not Downloaded

### YourSkatingCoach (2024)
- **Paper:** arXiv:2410.20427
- BIOES-tagged element annotations, fine-grained boundaries
- Data in supplementary materials

### FSBench (CVPR 2025)
- **Paper:** arXiv:2504.19514
- 783 videos, 76+ hours, 3D kinematics + audio + text
- Temporarily closed access

### SkatingVerse (2024)
- **Paper:** IET Computer Vision (doi:10.1049/cvi2.12287)
- **Workshop:** ECCV 2024, 1st SkatingVerse Challenge
- 1,687 videos, 28 classes, 19,993 train / 8,586 test clips
- **Status:** Access via CodaLab (competition ended) or email request
- **Contact:** SkatingVerse.163.com
- Can be replaced by MMFS + Figure-Skating-Classification for fine-tuning

### MCFS (2021)
- **Paper:** "Temporal Segmentation of Fine-grained Semantic Action: A Motion-centered Figure Skating Dataset" — AAAI 2021
- **Site:** https://shenglanliu.github.io/mcfs-dataset/
- Frame-wise annotations for action segmentation
- OpenPose BODY_25 format, 271 videos
- **Status:** Currently downloading

### FineFS (2023)
- **Paper:** "FineFS: A High-Quality Figure Skating Dataset for Fine-Grained Action Segmentation and Quality Assessment"
- 1,167 samples with scores and boundaries
- **Status:** Currently downloading

---

## Storage Summary

| Dataset | Size | Format | Status |
|---------|------|--------|--------|
| AthletePose3D | 71GB | Video + 3D/2D annotations | Active |
| MMFS | 2.1GB | Skeleton npy + pkl | Active |
| Figure-Skating-Classification | 340MB | Skeleton pkl (pre-processed) | Active |
| MCFS | downloading | Video + frame labels | In progress |
| FineFS | downloading | Annotations | In progress |
