# CLAUDE.md

> **PROJECT ROADMAP:** @ROADMAP.md — SINGLE SOURCE OF TRUTH for implementation status
> **RESEARCH:** @docs/research/RESEARCH_SUMMARY_2026-03-28.md — Exa + Gemini findings (41 papers)

---

## Project Overview

ML-based AI coach for figure skating. Analyzes video, compares attempts to professional references, provides biomechanical feedback in Russian.

**Vision:** AI-тренер по фигурному катанию — анализ видео и рекомендации на русском.

## Directory Structure

```
skating-biomechanics-ml/
├── backend/                          # FastAPI API server
│   ├── app/                          # Python package (backend.app.*)
│   │   ├── routes/                   # FastAPI routers
│   │   ├── models/                   # SQLAlchemy ORM models
│   │   ├── crud/                     # Database CRUD operations
│   │   ├── services/                 # Business logic
│   │   ├── auth/                     # JWT auth (deps.py)
│   │   ├── config.py                 # Settings (Pydantic BaseSettings)
│   │   ├── storage.py                # R2/S3 client
│   │   ├── task_manager.py           # Valkey task queue helpers
│   │   └── schemas.py                # Pydantic request/response schemas
│   ├── alembic/                      # Database migrations
│   ├── tests/                        # Backend tests
│   └── pyproject.toml                # Backend-only dependencies
├── frontend/                         # Next.js 16 app
│   ├── app/                          # App router pages
│   ├── components/                   # React components
│   ├── lib/                          # API client, hooks, utils
│   ├── i18n/                         # next-intl (ru/en)
│   └── messages/                     # Translation files
├── ml/                               # ML pipeline + arq worker
│   ├── skating_ml/                   # Python package (skating_ml.*)
│   │   ├── pose_estimation/          # RTMPose via rtmlib
│   │   ├── analysis/                 # Metrics, phase detection, recommender
│   │   ├── pose_3d/                  # 3D lifting, corrective lens
│   │   ├── detection/                # Person detection, tracking
│   │   ├── utils/                    # Smoothing, visualization, gap filling
│   │   ├── visualization/            # HUD, skeleton, comparison layers
│   │   ├── worker.py                 # arq worker (process_video_task, detect_video_task)
│   │   └── extras/                   # Optional ML models (depth, optical flow)
│   ├── gpu_server/                   # Vast.ai GPU server (Containerfile)
│   ├── tests/                        # ML tests
│   └── pyproject.toml                # ML dependencies (depends on backend)
├── docs/                             # Documentation
│   └── research/                     # Research papers and findings
├── infra/                            # Infrastructure
│   ├── Containerfile                 # Docker image for backend
│   └── Caddyfile                     # Reverse proxy config
├── data/                             # Data files (datasets, references)
├── experiments/                      # Jupyter notebooks, experiments
└── pyproject.toml                    # Root config (shared dev deps)
```

## Architecture

```
Frontend → FastAPI (backend/) → Valkey queue → arq worker (ml/skating_ml/)
  → [VASTAI_API_KEY set?]
    → YES: upload to R2 → Vast.ai route → GPU worker → download from R2
    → NO:  local GPU (process_video_pipeline)

ML Pipeline:
  Video → RTMPose (rtmlib, CUDA) → HALPE26 (26kp)
    → H3.6M (17kp) conversion → GapFiller → Smoothing
    → [Optional] CorrectiveLens (3D lift → kinematic constraints → project back to 2D)
    → Phase Detection → Biomechanics Metrics → DTW (vs reference)
    → Rule-based Recommender → Russian Text Report
```

**Key architectural constraint:** Backend (`backend/`) has **ZERO ML imports**. All ML runs in the arq worker (`ml/skating_ml/worker.py`). The worker depends on `backend` for DB/storage access, but never the reverse.

**Key decisions:**
- **rtmlib**: sole pose estimation backend — HALPE26 (26kp), ONNX (CPU+GPU), foot keypoints
- **HALPE26 (26kp)** as intermediate format, converted to H3.6M (17kp) for downstream
- **CorrectiveLens**: 3D lifting as corrective layer for 2D skeleton (Kinovea-style angles)
- **PoseTracker**: anatomical biometric Re-ID instead of color (solves black clothing on ice)
- **CoM trajectory** instead of flight time (eliminates 60% error for low jumps)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Pipeline** | Python, rtmlib, onnxruntime-gpu, scipy |
| **Backend API** | FastAPI, SQLAlchemy, Alembic, arq + Valkey |
| **Frontend** | Next.js 16, React, Tailwind CSS, shadcn/ui, Recharts |
| **Storage** | Cloudflare R2 (S3-compatible) |
| **Remote GPU** | Vast.ai Serverless |
| **Testing** | pytest (backend), tsc + next lint (frontend) |

## Git & GitHub Workflow

### Branches

- **Format**: `feature/<short-name>` (e.g., `feature/onnx-export`)
- **Main branch**: `master`
- **Before push**: `git fetch origin && git merge origin/master`

### Commits

- **Format**: `<type>(<scope>): <description>`
- **Types**: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `ci`
- **Scopes**: `pose`, `viz`, `tracking`, `analysis`, `pipeline`, `cli`, `models`, `repo`, `frontend`, `backend`, `dev`, `ci`, `vastai`, `infra`

### Pull Requests

| Field | Value |
|-------|-------|
| Base branch | `master` |
| Title | Same format as commit |
| Description | Must include "Что сделано" and "Как проверить" sections |

## GPU Requirements

**GPU-only. CPU inference is forbidden.** Always use `device='cuda'`.
Before running: `bash ml/scripts/setup_cuda_compat.sh` (required after `uv sync`).
System has CUDA 13.2, onnxruntime-gpu needs CUDA 12 compat libs in `.venv/cuda-compat/`.

## Key Concepts

- `poses_norm` — Normalized [0,1], `poses_px` — Pixel coordinates. Validate with `assert_pose_format()`.
- `halpe26_to_h36m()`: 26kp (COCO 17 + 6 foot + 3 face) → 17kp H3.6M. Foot keypoints preserved separately.
- **CorrectiveLens**: 2D → MotionAGFormer 3D lift → kinematic constraints → anchor projection → blend.
- **CUDA compat**: standalone CUDA 12 libs in `.venv/cuda-compat/` with patched RUNPATH.

## Remote GPU Processing (Vast.ai Serverless)

Worker dispatches to Vast.ai Serverless GPU when `VASTAI_API_KEY` is set, falls back to local GPU. Worker code lives in `ml/skating_ml/worker.py`, Vast.ai server in `ml/gpu_server/`.

**Image**: `ghcr.io/xpos587/skating-ml-gpu:latest` — multi-stage, 4.9GB, no torch/timm/triton.

## Tracking Debugging Workflow

When tracking quality degrades (skeleton jumps to wrong person), follow this data-driven analysis approach. **Do NOT guess — extract data and find the exact divergence frame.**

### Step 1: Isolate the layer

The tracking pipeline has 3 layers that can independently cause track switches:
1. **Sports2DTracker** — per-frame centroid association (Kalman-predicted distance matrix)
2. **Anti-steal logic** — in `ml/skating_ml/pose_estimation/rtmlib_extractor.py`, guards against centroid jumps
3. **Tracklet merger** — post-hoc NaN gap filling with biometric re-association

### Step 2: Analyze centroid trajectories

In the CSV, look for:
- **Sports2D misassignment**: track ID on wrong detection index without anti-steal trigger
- **Anti-steal false positive**: `target_track_id` switched even though Sports2D assigned correctly
- **Tracklet merger error**: wrong track merged into the gap

### Step 3: Fix the root cause

Key lessons learned:
- **Anti-steal must use AND (not OR)** for combining position + biometric signals.
- **Kalman dt=1.0 (frame-based), not dt=1/fps**.
- **Figure skating movements are NOT anomalies** — leg swings, rotations are normal.

### Anti-steal thresholds

- Centroid jump: `> 0.15` (normalized coordinates)
- Skeletal anomaly: `> 0.25` (bone ratio change)
- **Logic: AND** — both must exceed threshold simultaneously

## References

- @ROADMAP.md — project status (SINGLE SOURCE OF TRUTH)
- @docs/research/RESEARCH_SUMMARY_2026-03-28.md — research findings (41 papers)
- @docs/research/RESEARCH.md — research memory bank (index)
