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

### AthletePose3D (CVSports 2025)

| Field | Value |
|-------|-------|
| **Repo** | https://github.com/calvinyeungck/AthletePose3D |
| **Size** | 71GB (5154 videos, 1.3M frames, 165K postures) |
| **License** | Non-commercial research only |
| **Status** | Downloaded, coco_annotations/ deleted (bad 26kp reannotation) |
| **Local path** | `data/datasets/athletepose3d/` |

**Contents:**
- 12 sports: Axel, Flip, Lutz, Loop, Salchow, Toeloop, Spin, Running, Discus, Javelin, Shot, Glide, Comb
- Videos from multiple camera angles per subject
- 2D annotations: `pose_2d/pose_2d/annotations/` (17kp COCO format, clean)
- 3D annotations: `annotations_3d/pose_3d_v3/` (H3.6M 17kp, frame-81 windows)
- Detection results: `pose_2d/pose_2d/det_result/` (YOLOv8)
- Camera params: `cam_param.json` (24 cameras)

**Subsumes:** FS-Jump3D (Tanaka et al. 2024) — same data, extended to 12 sports.

**Caveat:** `coco_annotations/` was deleted because it contained a failed attempt to extend 17 COCO kp to 26 HALPE26 kp (foot + face keypoints). Only 18/26 were valid, rest were zeros. Use `pose_2d/pose_2d/annotations/` for clean 17kp data.

---

### MMFS (Multi-modality Multi-task Figure Skating, 2023)

| Field | Value |
|-------|-------|
| **Repo** | https://github.com/dingyn-Reno/MMFS |
| **Paper** | arXiv:2307.02730 |
| **Size** | 1.7GB (skeleton data only) |
| **License** | MIT |
| **Status** | Downloaded and extracted |
| **Local path** | `data/datasets/mmfs/extracted/MMFS/` |

**Contents:**
- 26198 skeleton sequences (.npy files) in `skeleton/`
- 256 fine-grained action categories
- Spatial + temporal labels
- RGB video available on request (copyright) — contact liusl@dlut.edu.cn

---

### Figure-Skating-Classification (HuggingFace)

| Field | Value |
|-------|-------|
| **HF Link** | https://huggingface.co/datasets/Mercity/Figure-Skating-Classification-Data |
| **Size** | 340MB (train_data.pkl + test_data.pkl) |
| **Status** | Downloaded |
| **Local path** | `data/datasets/figure-skating-classification/` |

**Contents:**
- 5405 skeleton sequences (4324 train, 1081 test)
- 64 element classes (single jumps, combinations, spins, step sequences)
- 17kp COCO format, 150 frames per sequence, normalized to [-1, 1]
- Compiled from MMFS + mocap data
- `label_mapping.json` — class name → label mapping
- `dataset_info.json` — metadata

---

## Not Downloaded

### YourSkatingCoach (2024)
- arXiv:2410.20427
- BIOES-tagged element annotations
- Data in supplementary materials

### FSBench (CVPR 2025)
- arXiv:2504.19514
- 783 videos, 76+ hours, 3D kinematics + audio + text
- Request access via GitHub

### JudgeAI-LutzEdge (2023)
- https://github.com/ryota-skating/JudgeAI-LutzEdge
- Code only — IMU data is private

---

## Storage Summary

| Dataset | Size | Status |
|---------|------|--------|
| AthletePose3D | 71GB | Active |
| Figure-Skating-Classification | 340MB | Active |
| MMFS | 1.7GB | Active |
| Available disk | 290GB | — |
