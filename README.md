# Skating Biomechanics ML

[![CI](https://img.shields.io/github/actions/workflow/status/Xpos587/skating-biomechanics-ml/ci.yml?branch=master&label=CI)](https://github.com/Xpos587/skating-biomechanics-ml/actions)
[![Tests](https://img.shields.io/github/actions/workflow/status/Xpos587/skating-biomechanics-ml/ci.yml?branch=master&label=tests)](https://github.com/Xpos587/skating-biomechanics-ml/actions)
[![codecov](https://codecov.io/github/Xpos587/skating-biomechanics-ml/graph/badge.svg?token=0QK5TTR8QZ)](https://codecov.io/github/Xpos587/skating-biomechanics-ml)
[![License: MIT](https://img.shields.io/github/license/Xpos587/skating-biomechanics-ml)](https://github.com/Xpos587/skating-biomechanics-ml/blob/master/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Xpos587/skating-biomechanics-ml)](https://github.com/Xpos587/skating-biomechanics-ml/commits/master)
[![Issues](https://img.shields.io/github/issues/Xpos587/skating-biomechanics-ml)](https://github.com/Xpos587/skating-biomechanics-ml/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/Xpos587/skating-biomechanics-ml/pulls)

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/mypy-checked-blue?logo=mypy&logoColor=white)](https://mypy-lang.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)

AI-тренер по фигурному катанию — анализ видео, сравнение с эталонами, биомеханическая обратная связь на русском.

## Quick Start

```bash
uv sync
bash scripts/setup_cuda_compat.sh   # CUDA GPU setup (RTX 3050 Ti)

# Анализ видео
uv run python -m src.cli analyze video.mp4 --element waltz_jump --pose-backend rtmlib

# Сравнение двух видео (тренировочный режим)
uv run python -m src.cli compare attempt.mp4 reference.mp4 --overlays skeleton,angles,timer

# Визуализация с 3D-коррекцией скелета
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 2 --3d --output out.mp4
```

## Architecture

```
Video → RTMPose (rtmlib, CUDA) → HALPE26 (26kp) → H3.6M (17kp)
  → GapFiller → Smoothing → [Optional] CorrectiveLens (3D→2D correction)
  → Phase Detection → Biomechanics Metrics → DTW → Recommender → Russian Report
```

| Component | Technology |
|-----------|-----------|
| **2D Pose** | RTMPose via rtmlib (HALPE26, 26kp, CUDA) |
| **3D Lifting** | MotionAGFormer-S / Biomechanics3DEstimator |
| **3D Correction** | CorrectiveLens (kinematic constraints + anchor projection) |
| **Tracking** | OC-SORT + anatomical biometric Re-ID |
| **Physics** | CoM trajectory, Dempster anthropometric tables |
| **GPU** | CUDA via onnxruntime-gpu (7.1x speedup) |

## Project Structure

```
src/
├── pose_estimation/     # RTMPose (rtmlib), YOLO26-Pose
├── pose_3d/             # CorrectiveLens, MotionAGFormer, TCPFormer
├── detection/           # PoseTracker, spatial reference, blade detection
├── analysis/            # Physics engine, metrics, recommender
├── visualization/       # Layered HUD, comparison, skeleton
├── alignment/           # DTW motion alignment
└── utils/               # GapFiller, geometry, smoothing
```

## Research

See [`research/RESEARCH.md`](research/RESEARCH.md) — index of all research materials, memory bank.

## Quality

```bash
uv run pytest tests/ -v -m "not slow"   # 272+ tests
uv run ruff check .                      # Lint
uv run ruff format .                     # Format
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
