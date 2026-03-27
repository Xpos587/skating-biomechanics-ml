# CLAUDE.md

## Project Overview

ML-based system for figure skating biomechanics analysis using computer vision.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11+ |
| **Package Manager** | `uv` |
| **Detection** | YOLOv11m (Ultralytics) |
| **2D Pose** | MediaPipe BlazePose (33 keypoints) |
| **3D Lifting** | Pose3DM-L (State-Space Model) |
| **Alignment** | DTW (dtw-python), MotionDTW |
| **Analysis** | KISMAM (custom biomechanics metrics) |
| **RAG** | Qwen3 / GPT-4o + Pinecone/pgvector |
| **Testing** | Pytest + pytest-cov |

## ML Pipeline Architecture

```
Video Input → YOLOv11 (detect) → BlazePose (2D keypoints) → Pose3DM-L (3D lift)
    ↓
MotionDTW (temporal alignment) → KISMAM (biomechanics analysis)
    ↓
RAG (Qwen3/GPT-4o) → Coach recommendations (natural language)
```

## Project Structure

```
src/skating_biomechanics_ml/
├── detection/      # YOLOv11 object detection (athlete localization)
├── pose_2d/        # BlazePose 2D keypoint extraction (33 points)
├── pose_3d/        # Pose3DM-L 2D→3D lifting with FTV regularization
├── alignment/      # MotionDTW for temporal synchronization
├── analysis/       # KISMAM biomechanics metrics & semantic diagnosis
└── rag/            # Multimodal RAG for coach recommendations

research/           # Research docs (see RESEARCH.md for architecture details)
data/
├── raw/            # AthletePose3D, FS-Jump3D datasets
└── processed/      # Normalized 3D sequences
```

## Development Workflow

### Quality Checks

```bash
# Run all checks
uv run python scripts/check_all.py

# Individual checks
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run mypy src/             # Type check
uv run vulture src/ tests/   # Dead code
uv run pytest tests/ -v      # Tests
```

### Adding Dependencies

```bash
uv add <package>              # Runtime dependency
uv add --dev <package>        # Dev dependency
uv sync                       # Install all
```

### Commit Convention

```
type(scope): description

Types: feat, fix, docs, refactor, test, chore
Scopes: detection, pose_2d, pose_3d, alignment, analysis, rag, tests, config
```

## Key Concepts

### Biomechanics Metrics
- **MPJPE** (Mean Per Joint Position Error): 3D accuracy target <40mm
- **Airtime**: $H = 4.905 \times (t/2)^2$ for jump height
- **Angular velocity**: Peak >1500°/s for elite athletes
- **Edge detection**: Toe pick (tooth) vs rocker (edge) takeoff

### Data Normalization
1. Camera motion compensation (homography matrix)
2. Root centering (pelvis → origin)
3. Scale normalization (spine length = 0.4)

### Datasets
- **AthletePose3D**: 1.3M frames, 12 sports, fine-tuning reduces MPJPE by 69%
- **FS-Jump3D**: Figure skating specific with phase annotations (entry/flight/landing)

## Environment

- **OS**: Artix Linux (Ryzen 7 5800H / RTX 3050 Ti)
- **Container Runtime**: Podman (not Docker)
- **Python**: 3.11+ managed via `uv`

## References

- Full architecture research: [`research/RESEARCH.md`](research/RESEARCH.md)
- ISU Scale of Values: Official scoring rules
- TTGNN: Table-Text Graph Neural Network for rule enforcement
