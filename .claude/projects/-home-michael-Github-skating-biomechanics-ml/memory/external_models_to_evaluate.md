# External Models to Evaluate

Saved from research on 2026-03-28

## Temporal Action Segmentation (TAS) - Jump Phases

### VIFSS (Tanaka et al. 2025)
- **Paper**: "View-Invariant and Figure Skating-Specific Pose Representation Learning"
- **Accuracy**: 92% F1@50 on element-level TAS
- **Features**: Jump type + rotation level + phase detection (entry/flight/landing)
- **Code**: Paper only (may need to contact authors)
- **Link**: arXiv:2508.10281
- **Use case**: Replace our PhaseDetector (currently 50% working)

### figure-skating-action-segmentation (mayupei/GitHub)
- **Repo**: https://github.com/mayupei/figure-skating-action-segmentation
- **Method**: LSTM-CNN two-stage framework
- **Features**: Sliding window LSTM + CNN refinement
- **Code**: ✅ Available (Jupyter + Python)
- **Data**: MCFS + MMFS datasets
- **Use case**: Action segmentation for competition videos

## Blade Edge Detection

### JudgeAI-LutzEdge (Tanaka 2023)
- **Repo**: https://github.com/ryota-skating/JudgeAI-LutzEdge
- **Method**: 3D pose estimation + IMU sensors
- **Features**: Automatic edge error judgment
- **Code**: ✅ Available
- **Paper**: "Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation from a Monocular Camera and IMUs"
- **Use case**: Improve our BladeEdgeDetector

## Datasets

### FS-Jump3D
- **Repo**: https://github.com/ryota-skating/FS-Jump3D
- **Content**: 3D pose jumps + semantic annotations (entry/flight/landing)
- **Size**: 8.84 GB videos + 505MB JSON + 302MB c3d
- **Features**: 12 viewpoints, markerless motion capture
- **Status**: Integrated into AthletePose3D (CVPR 2025)

### VIFSS Training Data
- **FS-Jump3D** for contrastive pre-training
- **SkatingVerse** for action classification fine-tuning
- **Fine-grained annotation**: entry/preparation + landing phases

## Priority for Integration

1. **HIGH**: JudgeAI-LutzEdge for blade edge detection
2. **MEDIUM**: VIFSS for phase detection (if code available)
3. **LOW**: figure-skating-action-segmentation (different task - competition programs)
