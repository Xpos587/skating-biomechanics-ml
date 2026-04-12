# ml/CLAUDE.md — ML Pipeline & arq Worker

## Architectural Constraint

**ml depends on backend, never the reverse.** The worker (`skating_ml/worker.py`) imports from `backend.app` for DB/storage/task management. The backend has **ZERO** imports from `skating_ml`.

## Project Structure

```
ml/
├── skating_ml/                       # Python package (skating_ml.*)
│   ├── __init__.py                   # Exports DeviceConfig
│   ├── types.py                      # Core types: H36Key, FrameKeypoints, PersonClick, etc.
│   ├── device.py                     # DeviceConfig — GPU/CPU auto-detection
│   ├── pipeline.py                   # AnalysisPipeline orchestrator
│   ├── cli.py                        # argparse CLI (analyze, build-ref, segment)
│   ├── worker.py                     # arq worker (process_video_task, detect_video_task)
│   ├── web_helpers.py                # Preview rendering for detect endpoint
│   ├── pose_estimation/              # 2D pose extraction
│   │   ├── rtmlib_extractor.py       # RTMPose via rtmlib (HALPE26, 26kp) — PRIMARY
│   │   ├── h36m.py                   # H3.6M 17kp format handling
│   │   ├── halpe26.py                # HALPE26 26kp format + foot angles
│   │   ├── normalizer.py             # Root-centering + scale normalization
│   │   └── person_selector.py        # Interactive person selection
│   ├── pose_3d/                      # 3D pose lifting
│   │   ├── corrective_pipeline.py    # CorrectiveLens: 3D lift → constraints → project
│   │   ├── kinematic_constraints.py  # Bone length + joint angle enforcement
│   │   ├── anchor_projection.py      # 3D→2D projection with confidence blending
│   │   ├── athletepose_extractor.py  # MotionAGFormer / TCPFormer integration
│   │   └── normalizer_3d.py          # 3D pose normalization
│   ├── analysis/                     # Biomechanics analysis
│   │   ├── metrics.py                # Airtime, height, knee angles, rotation, etc.
│   │   ├── phase_detector.py         # CoM-based takeoff/peak/landing detection
│   │   ├── physics_engine.py         # CoM, moment of inertia, angular momentum
│   │   ├── recommender.py            # Rule-based Russian text recommendations
│   │   ├── element_defs.py           # Figure skating element definitions
│   │   ├── element_segmenter.py      # Automatic element segmentation
│   │   └── rules/                    # Per-element recommendation rules
│   ├── detection/                    # Person detection & tracking
│   │   ├── person_detector.py        # YOLO-based person detection
│   │   ├── pose_tracker.py           # Kalman filter + biometric Re-ID
│   │   ├── spatial_reference.py      # Per-frame camera pose estimation
│   │   └── blade_edge_detector_3d.py # BDA algorithm (not wired into pipeline)
│   ├── tracking/                     # Multi-person tracking
│   │   ├── sports2d.py               # Sports2D centroid association
│   │   ├── tracklet_merger.py        # NaN gap filling with biometric re-association
│   │   ├── deepsort_tracker.py       # DeepSORT integration
│   │   └── skeletal_identity.py      # Anatomical ratio Re-ID
│   ├── alignment/                    # Motion comparison
│   │   ├── aligner.py                # MotionAligner class
│   │   └── motion_dtw.py             # DTW with Sakoe-Chiba window
│   ├── visualization/                # Rendering & HUD
│   │   ├── comparison.py             # Side-by-side / overlay comparison
│   │   ├── export_3d.py              # 3D skeleton export (glTF)
│   │   ├── export_3d_animated.py     # Animated 3D export
│   │   ├── hud/                      # HUD layer system (0-3)
│   │   ├── skeleton/                 # Skeleton drawer
│   │   └── layers/                   # Individual overlay layers
│   ├── utils/                        # Shared utilities
│   │   ├── video.py                  # extract_frames, get_video_meta
│   │   ├── geometry.py               # Angles, distances
│   │   ├── smoothing.py              # One-Euro filter
│   │   ├── gap_filling.py            # 3-tier gap filling
│   │   └── subtitles.py              # VTT subtitle generation
│   ├── references/                   # Reference motion database
│   │   ├── reference_builder.py      # Build .npz references from video
│   │   └── reference_store.py        # Save/load reference files
│   ├── datasets/                     # Dataset handling
│   │   ├── coco_builder.py           # COCO format builder
│   │   └── projector.py              # 3D projection utilities
│   ├── vastai/                       # Vast.ai Serverless GPU dispatch
│   │   └── client.py                 # Worker URL resolution, job submission
│   └── extras/                       # Optional ML models (not in pipeline)
│       ├── model_registry.py         # Model download/management
│       ├── depth_anything.py         # Depth estimation
│       ├── optical_flow.py           # Optical flow
│       ├── segment_anything.py       # SAM segmentation
│       ├── inpainting.py             # Video inpainting
│       └── foot_tracker.py           # Foot tracking
├── gpu_server/                       # Vast.ai GPU server
│   ├── server.py                     # FastAPI server for GPU worker
│   └── Containerfile                 # Multi-stage build (4.9GB)
├── scripts/                          # Standalone scripts
│   ├── visualize_with_skeleton.py    # Main viz script (--layer, --3d, --select-person)
│   ├── setup_cuda_compat.sh          # CUDA 12 compat libs for CUDA 13.x
│   ├── download_ml_models.py         # Download model weights
│   ├── normalize_video.py            # H.264, 1280px, 30fps
│   ├── compare_videos.py             # Side-by-side analysis comparison
│   ├── build_references.py           # Build reference database
│   └── deploy.sh                     # Deploy to Vast.ai
├── tests/                            # 93+ tests
│   ├── conftest.py                   # Shared fixtures
│   ├── analysis/                     # Metrics, phase detector, recommender
│   ├── detection/                    # Person detector, pose tracker
│   ├── tracking/                     # DeepSORT, skeletal identity
│   ├── visualization/                # HUD, skeleton, layers
│   ├── alignment/                    # DTW, aligner
│   ├── pose_3d/                      # Corrective lens, anchor projection
│   └── utils/                        # Geometry, smoothing
└── pyproject.toml                    # ML deps + skating-backend dependency
```

## Key Types (`skating_ml.types`)

| Type | Purpose |
|------|---------|
| `H36Key` (IntEnum) | H3.6M 17-keypoint indices (primary format) |
| `FrameKeypoints` | Single frame: (17, 2) or (17, 3) array |
| `PersonClick` | User's person selection (x, y, frame) |
| `TrackedExtraction` | Per-person tracked pose sequence |
| `ElementPhase` | takeoff/flight/landing phase markers |
| `BladeType` (Enum) | INSIDE/OUTSIDE/FLAT/TOE_PICK/UNKNOWN |

## Pipeline Flow

```
Video → RTMPoseExtractor (HALPE26 26kp)
     → halpe26_to_h36m() (17kp)
     → GapFiller → Smoothing (One-Euro)
     → [Optional] CorrectiveLens (3D lift → constraints → project back)
     → PhaseDetector (CoM-based)
     → BiomechanicsAnalyzer (airtime, height, angles, rotation)
     → DTW alignment vs reference
     → Recommender → Russian text report
```

## GPU Requirements

**GPU-only. CPU inference is forbidden.** Always use `device='cuda'`.

```bash
# Required after every uv sync
bash ml/scripts/setup_cuda_compat.sh
```

System has CUDA 13.2, onnxruntime-gpu needs CUDA 12 compat libs in `.venv/cuda-compat/`.

## Device Configuration

```python
from skating_ml.device import DeviceConfig

cfg = DeviceConfig.default()        # Auto-detect (CUDA preferred)
cfg = DeviceConfig(device="cpu")    # Explicit CPU
cfg.onnx_providers                  # ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Environment override: SKATING_DEVICE=cpu
```

## Worker Jobs (`skating_ml.worker`)

Two async jobs registered with arq:

| Job | Trigger | What it does |
|-----|---------|-------------|
| `process_video_task` | `POST /process` | Full ML pipeline → save results to DB |
| `detect_video_task` | `POST /detect` | RTMPose extraction → render preview → store in Valkey |

Both jobs download video from R2, process on GPU, and store results. When `VASTAI_API_KEY` is set, jobs dispatch to Vast.ai Serverless GPU.

## CorrectiveLens (Disabled by Default)

3D lifting as corrective layer for 2D skeleton. Adds ~3px max shift — not worth the compute for most use cases. Enable with `--3d` flag in visualization scripts.

## Tracking Debugging

When skeleton jumps to wrong person, follow the data-driven approach in @CLAUDE.md (Tracking Debugging Workflow section).

## Before Committing

1. **Tests**: `uv run python -m pytest ml/tests/ --no-cov`
2. **Lint**: `uv run ruff check ml/skating_ml/`
3. **Type check**: `uv run basedpyright ml/skating_ml/ --level error`
