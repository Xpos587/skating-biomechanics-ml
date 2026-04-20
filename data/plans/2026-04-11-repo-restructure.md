# Repo Restructure: frontend/, backend/, ml/ at Root

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split monolithic `src/` into three root-level packages — `frontend/` (Next.js), `backend/` (FastAPI, zero ML imports), `ml/` (ML pipeline + arq worker) — with separate deployable `pyproject.toml` for each.

**Architecture:** Backend communicates with ML worker exclusively via arq/Valkey queue. No direct ML imports in backend. Worker depends on backend package for DB access and config (unidirectional: ml → backend). `/detect` route becomes async (same pattern as existing `/process/queue`).

**Tech Stack:** Python (uv, FastAPI, arq, SQLAlchemy), Next.js, bun, Valkey, Cloudflare R2

---

## Target Structure

```
./
├── frontend/                     # Next.js 16 (moved from src/frontend/)
│   ├── src/                      # React app source
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   └── next.config.ts
├── backend/                      # FastAPI (moved from src/backend/ + shared infra)
│   ├── app/                      # Python package "backend"
│   │   ├── __init__.py
│   │   ├── main.py               # uvicorn entry: backend.app.main:app
│   │   ├── config.py             # Settings (was src/config.py)
│   │   ├── database.py           # SQLAlchemy engine
│   │   ├── logging_config.py
│   │   ├── metrics_registry.py
│   │   ├── schemas.py            # Pydantic schemas + MLModelFlags
│   │   ├── storage.py            # R2 operations (was src/storage.py)
│   │   ├── task_manager.py       # Valkey task state (was src/task_manager.py)
│   │   ├── auth/                 # JWT, deps
│   │   ├── crud/                 # DB operations
│   │   ├── models/               # SQLAlchemy ORM
│   │   ├── routes/               # API endpoints (detect.py = async, no ML)
│   │   └── services/             # Business logic
│   ├── alembic/                  # DB migrations (moved from ./alembic/)
│   ├── alembic.ini
│   ├── tests/                    # Backend tests (moved from tests/backend/)
│   └── pyproject.toml            # Backend-only deps (FastAPI, SQLAlchemy, etc.)
├── ml/                           # ML pipeline + worker
│   ├── src/               # Python package "src"
│   │   ├── __init__.py
│   │   ├── cli.py                # CLI entry point (was src/cli.py)
│   │   ├── pipeline.py           # Analysis pipeline (was src/pipeline.py)
│   │   ├── device.py             # GPU/CPU (was src/device.py)
│   │   ├── types.py              # H36Key, BladeType, etc. (was src/types.py)
│   │   ├── worker.py             # arq worker (was src/worker.py)
│   │   ├── web_helpers.py        # Viz helpers (was src/web_helpers.py)
│   │   ├── alignment/
│   │   ├── analysis/
│   │   ├── detection/
│   │   ├── extras/               # was src/ml/ (model_registry, depth, flow, etc.)
│   │   ├── models/               # athletepose3d model defs
│   │   ├── pose_estimation/
│   │   ├── pose_3d/
│   │   ├── references/
│   │   ├── tracking/
│   │   ├── utils/
│   │   ├── visualization/
│   │   └── vastai/               # was src/vastai/ (client.py only)
│   ├── gpu_server/               # was ./vastai/ (server.py + Containerfile)
│   │   ├── server.py
│   │   └── Containerfile
│   ├── scripts/                  # ML scripts (moved from ./scripts/)
│   ├── tests/                    # ML tests (moved from ./tests/)
│   └── pyproject.toml            # ML deps + depends on backend package
├── docs/
│   ├── research/                 # (moved from ./research/)
│   ├── plans/                    # (moved from data/plans/)
│   └── specs/                    # (moved from data/specs/)
├── experiments/                  # (moved from data/experiments/)
├── data/
│   ├── models/
│   ├── references/
│   ├── raw/
│   ├── processed/
│   ├── uploads/
│   └── datasets/
├── infra/
│   ├── compose.yaml              # (moved from ./)
│   ├── Caddyfile                 # (moved from ./)
│   └── Containerfile             # Backend container (moved from ./)
├── CLAUDE.md
├── ROADMAP.md
├── pyproject.toml                # Root: dev tooling (pytest, ruff, basedpyright)
├── uv.lock
├── Taskfile.yml
├── lefthook.yml
└── .gitignore
```

## Import Mapping

| Old Import | New Import | Context |
|---|---|---|
| `from src.backend.xxx` | `from backend.app.xxx` | Backend internal |
| `from src.config` | `from backend.app.config` | Backend + worker |
| `from src.storage` | `from backend.app.storage` | Backend routes |
| `from src.task_manager` | `from backend.app.task_manager` | Backend + worker |
| `from src.worker import MLModelFlags` | `from backend.app.schemas import MLModelFlags` | Process route |
| `from src.types` | `from src.types` | ML code |
| `from src.device` | `from src.device` | ML code |
| `from src.utils.xxx` | `from src.utils.xxx` | ML code |
| `from src.analysis.xxx` | `from src.analysis.xxx` | ML code |
| `from src.visualization.xxx` | `from src.visualization.xxx` | ML code |
| `from src.pose_estimation.xxx` | `from src.pose_estimation.xxx` | ML code |
| `from src.ml.xxx` | `from src.extras.xxx` | ML code |
| `from src.alignment.xxx` | `from src.alignment.xxx` | ML code |
| `from src.tracking.xxx` | `from src.tracking.xxx` | ML code |
| `from src.references.xxx` | `from src.references.xxx` | ML code |
| `from src.datasets.xxx` | `from src.datasets.xxx` | ML code |
| `from src.detection.xxx` | `from src.detection.xxx` | ML code |
| `from src.pose_3d.xxx` | `from src.pose_3d.xxx` | ML code |
| `from src.vastai.xxx` | `from src.vastai.xxx` | ML code |
| `from src.pipeline` | `from src.pipeline` | ML code |
| `from src.cli` | `from src.cli` | ML code |
| `from skating_biomechanics_ml.xxx` | `from src.xxx` | Scripts |

---

## Phase 1: Make /detect Async (Prerequisite)

Backend must have zero ML imports before restructuring. Currently `/detect` synchronously calls `RTMPoseExtractor`. Refactor to async via arq worker (same pattern as `/process/queue`).

### Task 1: Add detection schemas to backend

**Files:**
- Modify: `src/backend/schemas.py:79-97`

- [ ] **Step 1: Add MLModelFlags and async detect schemas**

Add to `src/backend/schemas.py` after the existing `PersonClick` class:

```python
@dataclass
class MLModelFlags:
    """ML model feature flags for video processing."""
    depth: bool = False
    optical_flow: bool = False
    segment: bool = False
    foot_track: bool = False
    matting: bool = False
    inpainting: bool = False


class DetectQueueResponse(BaseModel):
    task_id: str
    video_key: str
    status: str = "pending"


class DetectResultResponse(BaseModel):
    persons: list[PersonInfo]
    preview_image: str
    video_key: str
    auto_click: PersonClick | None = None
    status: str
```

Add `from dataclasses import dataclass` to the file imports.

- [ ] **Step 2: Verify no import errors**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run python -c "from src.backend.schemas import MLModelFlags, DetectQueueResponse, DetectResultResponse"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/backend/schemas.py
git commit -m "feat(backend): add MLModelFlags and async detect schemas"
```

### Task 2: Create detection worker job

**Files:**
- Modify: `src/worker.py`

- [ ] **Step 1: Add detect_video_task function**

Add `detect_video_task` to `src/worker.py` after the existing `process_video_task`. This function:
1. Downloads video from R2
2. Runs RTMPoseExtractor to detect persons
3. Renders annotated preview
4. Stores result JSON in Valkey via task_manager

```python
async def detect_video_task(
    ctx: dict[str, Any],
    *,
    task_id: str,
    video_key: str,
    tracking: str = "auto",
) -> dict[str, Any]:
    """arq task: detect persons in uploaded video."""
    settings = get_settings()
    valkey = await get_valkey_client()

    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"task:{task_id}",
            mapping={"status": TaskStatus.RUNNING, "started_at": now},
        )

        import tempfile
        from pathlib import Path

        import cv2

        from src.device import DeviceConfig
        from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
        from src.storage import download_file
        from src.utils.video import get_video_meta
        from src.web_helpers import render_person_preview

        # Download video from R2
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "input.mp4"
            download_file(video_key, str(video_path))

            # Run pose detection
            cfg = DeviceConfig.default()
            extractor = RTMPoseExtractor(
                mode="balanced",
                tracking_backend="rtmlib",
                tracking_mode=tracking,
                conf_threshold=0.3,
                output_format="normalized",
                device=cfg.device,
            )
            persons, _ = extractor.preview_persons(video_path, num_frames=30)

            if not persons:
                result_data = {
                    "persons": [],
                    "preview_image": "",
                    "video_key": video_key,
                    "auto_click": None,
                    "status": "Люди не найдены. Попробуйте другое видео.",
                }
                await store_result(task_id, result_data, valkey=valkey)
                return result_data

            # Read first frame for preview
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError("Failed to read video frame")

            meta = get_video_meta(video_path)
            w, h = meta.width, meta.height

            annotated = render_person_preview(frame, persons, selected_idx=None)
            success, buf = cv2.imencode(".png", annotated)
            if not success:
                raise RuntimeError("Failed to encode preview image")
            import base64
            preview_b64 = base64.b64encode(buf).decode("ascii")

            # Auto-select if only one person
            auto_click = None
            status_msg: str
            if len(persons) == 1:
                mid_hip = persons[0]["mid_hip"]
                auto_click = {"x": int(mid_hip[0] * w), "y": int(mid_hip[1] * h)}
                status_msg = "Обнаружен 1 человек — выбран автоматически"
            else:
                status_msg = f"Обнаружено {len(persons)} человек. Выберите на превью или из списка."

            persons_out = [
                {
                    "track_id": p["track_id"],
                    "hits": p["hits"],
                    "bbox": p["bbox"],
                    "mid_hip": p["mid_hip"],
                }
                for p in persons
            ]

            result_data = {
                "persons": persons_out,
                "preview_image": preview_b64,
                "video_key": video_key,
                "auto_click": auto_click,
                "status": status_msg,
            }
            await store_result(task_id, result_data, valkey=valkey)
            return result_data

    except Exception as e:
        logger.exception("Detection task %s failed", task_id)
        await store_error(task_id, str(e), valkey=valkey)
        raise
    finally:
        await valkey.close()
```

Also register the function in `WorkerSettings.functions`:

```python
functions: ClassVar[list] = [process_video_task, detect_video_task]
```

Remove the `MLModelFlags` dataclass from `src/worker.py` (it's now in schemas.py).

- [ ] **Step 2: Commit**

```bash
git add src/worker.py
git commit -m "feat(worker): add detect_video_task for async person detection"
```

### Task 3: Refactor /detect route to async

**Files:**
- Modify: `src/backend/routes/detect.py`

- [ ] **Step 1: Replace detect.py with async version**

Replace entire `src/backend/routes/detect.py` with:

```python
"""POST /api/detect — enqueue person detection job."""

from __future__ import annotations

import uuid
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, HTTPException, UploadFile

from src.backend.schemas import (
    DetectQueueResponse,
    DetectResultResponse,
    PersonClick,
    PersonInfo,
    TaskStatusResponse,
)
from src.config import get_settings
from src.storage import upload_bytes
from src.task_manager import get_task_state, get_valkey_client, create_task_state

router = APIRouter()


@router.post("/detect", response_model=DetectQueueResponse)
async def enqueue_detect(
    video: UploadFile,
    tracking: str = "auto",
) -> DetectQueueResponse:
    """Upload video, enqueue detection job, return task_id immediately."""
    suffix = Path(video.filename or "video.mp4").suffix
    video_key = f"input/{uuid.uuid4().hex}{suffix}"

    content = await video.read()
    upload_bytes(content, video_key)

    settings = get_settings()
    task_id = f"det_{uuid.uuid4().hex[:12]}"

    valkey = await get_valkey_client()
    try:
        await create_task_state(task_id, video_key=video_key, valkey=valkey)
    finally:
        await valkey.close()

    arq_pool = await create_pool(
        RedisSettings(
            host=settings.valkey.host,
            port=settings.valkey.port,
            database=settings.valkey.db,
            password=settings.valkey.password.get_secret_value(),
        )
    )
    try:
        await arq_pool.enqueue_job(
            "detect_video_task",
            task_id=task_id,
            video_key=video_key,
            tracking=tracking,
        )
    finally:
        await arq_pool.close()

    return DetectQueueResponse(task_id=task_id, video_key=video_key)


@router.get("/detect/{task_id}/status", response_model=TaskStatusResponse)
async def get_detect_status(task_id: str):
    """Poll detection task status."""
    valkey = await get_valkey_client()
    try:
        state = await get_task_state(task_id, valkey=valkey)
    finally:
        await valkey.close()

    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")

    result = None
    if state.get("result"):
        result = DetectResultResponse(**state["result"])

    return TaskStatusResponse(
        task_id=task_id,
        status=state["status"],
        progress=state["progress"],
        message=state.get("message", ""),
        result=result,
        error=state.get("error"),
    )


@router.get("/detect/{task_id}/result", response_model=DetectResultResponse)
async def get_detect_result(task_id: str):
    """Get detection result (persons, preview)."""
    valkey = await get_valkey_client()
    try:
        state = await get_task_state(task_id, valkey=valkey)
    finally:
        await valkey.close()

    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if state["status"] != TaskStatus.COMPLETED if "status" in state else True:
        raise HTTPException(status_code=400, detail="Task not completed yet")

    if not state.get("result"):
        raise HTTPException(status_code=500, detail="No result stored")

    return DetectResultResponse(**state["result"])
```

Note: `TaskStatus` import is needed — add `from src.task_manager import TaskStatus` (it's already imported via `get_task_state`).

- [ ] **Step 2: Verify backend has zero ML imports**

Run: `grep -rn "from src\." src/backend/routes/detect.py`
Expected: Only `from src.backend.schemas`, `from src.config`, `from src.storage`, `from src.task_manager`

- [ ] **Step 3: Update process.py to use MLModelFlags from schemas**

In `src/backend/routes/process.py`, change:
```python
from src.worker import MLModelFlags
```
to:
```python
from src.backend.schemas import MLModelFlags
```

- [ ] **Step 4: Commit**

```bash
git add src/backend/routes/detect.py src/backend/routes/process.py
git commit -m "feat(backend): make /detect async via arq worker (zero ML imports)"
```

### Task 4: Update frontend for async detection

**Files:**
- Modify: `src/frontend/src/lib/api-client.ts` (or wherever detect API call lives)
- Modify: `src/frontend/src/app/(app)/upload/page.tsx`

- [ ] **Step 1: Add async detect API functions**

Add to the frontend API layer (wherever `apiPost` etc. are defined):

```typescript
export async function apiDetectEnqueue(
  file: File,
  tracking = "auto",
): Promise<{ task_id: string; video_key: string }> {
  const form = new FormData()
  form.append("video", file)
  form.append("tracking", tracking)
  return apiPost("/detect", form)
}

export async function apiDetectStatus(
  taskId: string,
): Promise<TaskStatusResponse> {
  return apiFetch(`/detect/${taskId}/status`, TaskStatusSchema)
}

export async function apiDetectResult(
  taskId: string,
): Promise<DetectResultResponse> {
  return apiFetch(`/detect/${taskId}/result`, DetectResultSchema)
}
```

- [ ] **Step 2: Update upload page to poll for detection results**

In the upload page, replace the synchronous detect call with:
1. Call `apiDetectEnqueue(file)` → get `task_id`
2. Poll `apiDetectStatus(task_id)` every 2s until completed
3. Call `apiDetectResult(task_id)` → show persons + preview

Use React Query's polling: `refetchInterval: (data) => data?.status === "completed" ? false : 2000`

- [ ] **Step 3: Verify frontend builds**

Run: `cd src/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/frontend/
git commit -m "feat(frontend): update upload page for async detection"
```

---

## Phase 2: Directory Restructure

### Task 5: Create target directories and move frontend

- [ ] **Step 1: Create all new directories**

```bash
mkdir -p backend/app backend/tests backend/alembic
mkdir -p ml/src ml/gpu_server ml/scripts ml/tests
mkdir -p docs/research docs/plans docs/specs
mkdir -p infra experiments
```

- [ ] **Step 2: Move frontend to root**

```bash
git mv src/frontend frontend
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(repo): move frontend to root directory"
```

### Task 6: Move backend to root

- [ ] **Step 1: Move backend app**

```bash
# Move backend app
git mv src/backend/auth backend/app/auth
git mv src/backend/crud backend/app/crud
git mv src/backend/models backend/app/models
git mv src/backend/routes backend/app/routes
git mv src/backend/services backend/app/services
git mv src/backend/__init__.py backend/app/__init__.py
git mv src/backend/main.py backend/app/main.py
git mv src/backend/schemas.py backend/app/schemas.py
git mv src/backend/database.py backend/app/database.py
git mv src/backend/logging_config.py backend/app/logging_config.py
git mv src/backend/metrics_registry.py backend/app/metrics_registry.py

# Move shared infra into backend
git mv src/config.py backend/app/config.py
git mv src/storage.py backend/app/storage.py
git mv src/task_manager.py backend/app/task_manager.py

# Move alembic
git mv alembic/versions backend/alembic/versions
git mv alembic/env.py backend/alembic/env.py
git mv alembic/script.py.mako backend/alembic/script.py.mako
git mv alembic/README backend/alembic/README
git mv alembic.ini backend/alembic.ini
```

- [ ] **Step 2: Update backend internal imports**

All files in `backend/app/` that import `from src.backend.xxx` must change to `from backend.app.xxx`. Run:

```bash
# In all backend/app/**/*.py files:
# from src.backend.auth → from backend.app.auth
# from src.backend.crud → from backend.app.crud
# from src.backend.models → from backend.app.models
# from src.backend.routes → from backend.app.routes
# from src.backend.services → from backend.app.services
# from src.backend.schemas → from backend.app.schemas
# from src.backend.database → from backend.app.database
# from src.backend.logging_config → from backend.app.logging_config
# from src.backend.metrics_registry → from backend.app.metrics_registry

find backend/app -name '*.py' -exec sed -i 's/from src\.backend\./from backend.app./g' {} +
```

- [ ] **Step 3: Update shared module imports in backend**

Files in `backend/app/` that import `from src.config`, `from src.storage`, `from src.task_manager`:

```bash
find backend/app -name '*.py' -exec sed -i 's/from src\.config import/from backend.app.config import/g' {} +
find backend/app -name '*.py' -exec sed -i 's/from src\.storage import/from backend.app.storage import/g' {} +
find backend/app -name '*.py' -exec sed -i 's/from src\.task_manager import/from backend.app.task_manager import/g' {} +
```

- [ ] **Step 4: Update alembic/env.py imports**

```bash
sed -i 's/from src\.backend\.models import Base/from backend.app.models import Base/g' backend/alembic/env.py
sed -i 's/from src\.config import get_settings/from backend.app.config import get_settings/g' backend/alembic/env.py
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(repo): move backend to root directory"
```

### Task 7: Move ML pipeline to ml/

- [ ] **Step 1: Move all remaining src/ modules to ml/src/**

```bash
# Core ML modules
git mv src/alignment ml/src/alignment
git mv src/analysis ml/src/analysis
git mv src/detection ml/src/detection
git mv src/datasets ml/src/datasets
git mv src/models ml/src/models
git mv src/pose_estimation ml/src/pose_estimation
git mv src/pose_3d ml/src/pose_3d
git mv src/references ml/src/references
git mv src/tracking ml/src/tracking
git mv src/utils ml/src/utils
git mv src/visualization ml/src/visualization

# Root-level ML files
git mv src/cli.py ml/src/cli.py
git mv src/pipeline.py ml/src/pipeline.py
git mv src/device.py ml/src/device.py
git mv src/types.py ml/src/types.py
git mv src/worker.py ml/src/worker.py
git mv src/web_helpers.py ml/src/web_helpers.py
git mv src/__init__.py ml/src/__init__.py

# Rename src/ml/ (model_registry etc) to extras/
git mv src/ml ml/src/extras

# Vast.ai client
git mv src/vastai ml/src/vastai

# GPU server
git mv vastai/server.py ml/gpu_server/server.py
git mv vastai/Containerfile ml/gpu_server/Containerfile
```

- [ ] **Step 2: Update ML internal imports**

All `from src.xxx` imports in ML code become `from src.xxx`:

```bash
# src.xxx → src.xxx (but NOT src.backend — those stay)
find ml/src -name '*.py' -exec sed -i 's/from src\.\([a-z_]\)/from src.\1/g' {} +

# Handle src.ml. → src.extras. (the model_registry etc)
find ml/src -name '*.py' -exec sed -i 's/from src\.ml\./from src.extras./g' {} +
find ml/src -name '*.py' -exec sed -i 's/from src\.ml import/from src.extras import/g' {} +

# Fix inline imports in web_helpers.py and worker.py that reference backend
# These need to point to backend.app instead
find ml/src -name '*.py' -exec sed -i 's/from src\.backend\./from backend.app./g' {} +

# Fix config import in worker (worker imports backend config)
find ml/src -name '*.py' -exec sed -i 's/from src\.config import/from backend.app.config import/g' {} +

# Fix task_manager import in worker
find ml/src -name '*.py' -exec sed -i 's/from src\.task_manager import/from backend.app.task_manager import/g' {} +

# Fix skating_biomechanics_ml package import in scripts
find ml/scripts -name '*.py' -exec sed -i 's/from skating_biomechanics_ml\./from src./g' {} +
```

- [ ] **Step 3: Update gpu_server imports**

```bash
sed -i 's/from src\.types import/from src.types import/g' ml/gpu_server/server.py
sed -i 's/from src\.web_helpers import/from src.web_helpers import/g' ml/gpu_server/server.py
```

- [ ] **Step 4: Move ML scripts**

```bash
git mv scripts/build_references.py ml/scripts/
git mv scripts/compare_models.py ml/scripts/
git mv scripts/compare_videos.py ml/scripts/
git mv scripts/download_ml_models.py ml/scripts/
git mv scripts/normalize_video.py ml/scripts/
git mv scripts/organize_dataset.py ml/scripts/
git mv scripts/prepare_athletepose3d.py ml/scripts/
git mv scripts/visualize_segmentation.py ml/scripts/
git mv scripts/visualize_with_skeleton.py ml/scripts/
git mv scripts/batch_validate_labels.py ml/scripts/
git mv scripts/setup_cuda_compat.sh ml/scripts/
```

Update script imports:

```bash
find ml/scripts -name '*.py' -exec sed -i 's/from src\./from src./g' {} +
```

- [ ] **Step 5: Move ML tests**

```bash
# Move ML test directories
git mv tests/alignment ml/tests/alignment
git mv tests/analysis ml/tests/analysis
git mv tests/detection ml/tests/detection
git mv tests/pose_2d ml/tests/pose_2d
git mv tests/pose_3d ml/tests/pose_3d
git mv tests/pose_estimation ml/tests/pose_estimation
git mv tests/segmentation ml/tests/segmentation
git mv tests/tracking ml/tests/tracking
git mv tests/utils ml/tests/utils
git mv tests/visualization ml/tests/visualization
git mv tests/test_*.py ml/tests/

# Move conftest.py
git mv tests/conftest.py ml/tests/conftest.py
```

Update test imports:

```bash
find ml/tests -name '*.py' -exec sed -i 's/from src\./from src./g' {} +
```

- [ ] **Step 6: Clean up empty src/ and old directories**

```bash
# Remove empty src/ (git mv should have emptied it)
ls src/
# If anything remains, check before removing
git rm -r src/ 2>/dev/null || true
git rm -r scripts/ 2>/dev/null || true
git rm -r tests/ 2>/dev/null || true
git rm -r vastai/ 2>/dev/null || true
git rm -r alembic/ 2>/dev/null || true
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(repo): move ML pipeline to ml/ directory"
```

### Task 8: Reorganize docs, experiments, infra

- [ ] **Step 1: Move documentation**

```bash
git mv research/ docs/research
git mv data/plans/ docs/plans
git mv data/specs/ docs/specs
```

- [ ] **Step 2: Move experiments**

```bash
git mv data/experiments/ experiments/
```

- [ ] **Step 3: Move infrastructure files**

```bash
git mv compose.yaml infra/compose.yaml
git mv Caddyfile infra/Caddyfile
git mv Containerfile infra/Containerfile
```

- [ ] **Step 4: Clean up empty data/ subdirectories if any**

```bash
ls data/
# Keep data/ with remaining contents (models, references, raw, processed, uploads, datasets)
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(repo): reorganize docs, experiments, and infra"
```

---

## Phase 3: Update Configuration Files

### Task 9: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update package discovery**

Change `[tool.hatch.build.targets.wheel]`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["backend/app", "ml/src"]
```

- [ ] **Step 2: Update pytest pythonpath**

Change `[tool.pytest.ini_options]`:
```toml
[tool.pytest.ini_options]
pythonpath = ["backend", "ml"]
```

- [ ] **Step 3: Update coverage source**

Change `[tool.coverage.run]`:
```toml
[tool.coverage.run]
source = ["backend/app", "ml/src"]
```

- [ ] **Step 4: Update ruff known-first-party**

Change `[tool.ruff.lint.isort]`:
```toml
[tool.ruff.lint.isort]
known-first-party = ["backend", "src"]
```

- [ ] **Step 5: Update ruff per-file-ignores paths**

Change any `src/**` patterns to `backend/**` and `ml/**`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "chore: update pyproject.toml for new package structure"
```

### Task 10: Update Taskfile.yml

**Files:**
- Modify: `Taskfile.yml`

- [ ] **Step 1: Update all path references**

```yaml
tasks:
  py-lint:
    cmd: "{{.UV_RUN}} ruff check backend/ ml/ ml/tests/ backend/tests/ --fix && {{.UV_RUN}} ruff format backend/ ml/ ml/tests/ backend/tests/"

  py-test:
    cmd: "{{.UV_RUN}} pytest ml/tests/ backend/tests/ -v -m \"not slow\" --tb=short"

  py-typecheck:
    cmd: "{{.UV_RUN}} basedpyright --level error backend/app ml/src"

  dev-backend:
    cmd: "{{.UV_RUN}} uvicorn backend.app.main:app --reload --port 8000"

  dev-frontend:
    dir: frontend
    cmd: "bun run dev"

  fe-lint:
    dir: frontend
    cmd: "{{.BUNX}} biome check --write ."

  fe-test:
    dir: frontend
    cmd: "bun test --run"

  fe-typecheck:
    dir: frontend
    cmd: "{{.BUNX}} tsc -p tsconfig.app.json"

  fe-build:
    dir: frontend
    cmd: "bun run build"

  vastai-build:
    cmd: podman build -f ml/gpu_server/Containerfile -t ghcr.io/xpos587/skating-ml-gpu:latest .
```

- [ ] **Step 2: Commit**

```bash
git add Taskfile.yml
git commit -m "chore: update Taskfile.yml for new directory structure"
```

### Task 11: Update remaining config files

- [ ] **Step 1: Update lefthook.yml**

Change all `src/` references. The file has these lines to update:
- Line 39: `glob: "src/frontend/**/*.{ts,tsx,js,jsx,json,css}"` → `glob: "frontend/**/*.{ts,tsx,js,jsx,json,css}"`
- Line 40: `run: cd src/frontend && bunx biome check --write --staged` → `run: cd frontend && bunx biome check --write --staged`
- Line 122: `run: uv run ruff check src/ tests/ scripts/ --fix` → `run: uv run ruff check backend/ ml/ ml/tests/ backend/tests/ --fix`
- Line 124: `run: cd src/frontend && bunx biome check --write .` → `run: cd frontend && bunx biome check --write .`
- Line 131: `run: cd src/frontend && bun test --run` → `run: cd frontend && bun test --run`
- Line 136: `run: uv run basedpyright --level error src/` → `run: uv run basedpyright --level error backend/app ml/src`
- Line 138: `run: cd src/frontend && bunx tsc -p tsconfig.app.json` → `run: cd frontend && bunx tsc -p tsconfig.app.json`

```bash
sed -i 's|src/frontend/|frontend/|g' lefthook.yml
sed -i 's|uv run ruff check src/ tests/ scripts/|uv run ruff check backend/ ml/ ml/tests/ backend/tests/|g' lefthook.yml
sed -i 's|uv run ruff format src/ tests/ scripts/|uv run ruff format backend/ ml/ ml/tests/ backend/tests/|g' lefthook.yml
sed -i 's|uv run basedpyright --level error src/|uv run basedpyright --level error backend/app ml/src|g' lefthook.yml
```

- [ ] **Step 2: Update infra/Containerfile**

Current `Containerfile` references:
- `COPY src/frontend/package.json src/frontend/bun.lock* src/frontend/` → `COPY frontend/package.json frontend/bun.lock* frontend/`
- `RUN cd src/frontend && bun install ...` → `RUN cd frontend && bun install ...`
- `COPY src/ src/` → `COPY backend/ backend/`
- `COPY scripts/ scripts/` → `COPY ml/scripts/ ml/scripts/` (only ML scripts remain)
- `RUN cd src/frontend && bun run build` → `RUN cd frontend && bun run build`
- `CMD ["uv", "run", "uvicorn", "src.backend.main:app", ...]` → `CMD ["uv", "run", "uvicorn", "backend.app.main:app", ...]`

- [ ] **Step 3: Update ml/gpu_server/Containerfile**

Current `vastai/Containerfile` lines 91-92:
```dockerfile
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser vastai/server.py vastai/server.py
```

Change to:
```dockerfile
COPY --chown=appuser:appuser ml/src/ src/
COPY --chown=appuser:appuser ml/gpu_server/ gpu_server/
```

Also update line 98 CMD:
```dockerfile
CMD ["python", "-m", "uvicorn", "gpu_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Update infra/Caddyfile**

Change `root * src/frontend/dist` to `root * frontend/dist`.

- [ ] **Step 5: Remove root package.json and bun.lock (frontend-only)**

```bash
git rm package.json bun.lock package-lock.json 2>/dev/null || true
```

- [ ] **Step 6: Update .gitignore if needed**

Add `ml/src/__pycache__/` patterns if not already covered by generic `__pycache__/`.

- [ ] **Step 7: Update ast-grep config**

Check `ast-grep/` directory for any `src/` path references and update them:
```bash
grep -rn "src/" ast-grep/
# Update any found references from src/frontend/ → frontend/, src/backend/ → backend/app/
```

- [ ] **Step 8: Handle scripts/check_all.py**

The `check_all.py` script is a dev utility. Either move to infra/ or delete if obsolete:
```bash
# Check what it does first
cat scripts/check_all.py
# Move if useful
git mv scripts/check_all.py infra/ 2>/dev/null || git rm scripts/check_all.py
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: update all config files for new structure"
```

---

## Phase 4: Separate pyproject.toml for Deployment

### Task 12: Create backend/pyproject.toml

- [ ] **Step 1: Create backend package config**

```toml
[project]
name = "skating-backend"
version = "0.1.0"
description = "FastAPI backend for figure skating biomechanics"
requires-python = ">=3.11"
dependencies = [
    "asyncpg>=0.30",
    "bcrypt>=4.0,<4.1",
    "email-validator>=2.3.0",
    "fastapi>=0.135.2",
    "passlib[bcrypt]>=1.7",
    "PyJWT>=2.9",
    "pydantic-settings>=2.0.0",
    "python-multipart>=0.0.22",
    "redis>=5.0.0",
    "sse-starlette>=3.3.4",
    "sqlalchemy[asyncio]>=2.0",
    "structlog>=25.5.0",
    "uvicorn[standard]>=0.42.0",
    "arq>=0.27.0",
    "boto3>=1.42.83",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app"]
```

- [ ] **Step 2: Create ml/pyproject.toml**

```toml
[project]
name = "skating-ml"
version = "0.1.0"
description = "ML pipeline for figure skating biomechanics analysis"
requires-python = ">=3.11"
dependencies = [
    "rtmlib>=0.0.7",
    "onnxruntime-gpu>=1.24.4",
    "opencv-python>=4.10.0",
    "numpy>=2.0.0",
    "dtw-python>=1.5.0",
    "scipy>=1.14.0",
    "pillow>=12.1.1",
    "deep-sort-realtime>=1.3.2",
    "av>=17.0.0",
    "trimesh>=4.11.5",
    "pygltflib>=1.16.5",
    "arq>=0.27.0",
    "boto3>=1.42.83",
    "redis>=5.0.0",
    "structlog>=25.5.0",
    "ultralytics>=8.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

Note: For deployment, ml/pyproject.toml should add `skating-backend` as a dependency:
```toml
dependencies = [
    "...",
    "skating-backend @ file:///../backend",
]
```

- [ ] **Step 3: Commit**

```bash
git add backend/pyproject.toml ml/pyproject.toml
git commit -m "feat(repo): add separate pyproject.toml for backend and ml packages"
```

---

## Phase 5: Update Documentation

### Task 13: Update CLAUDE.md files

- [ ] **Step 1: Update root CLAUDE.md**

Update the Architecture section and all `src/` path references to new paths. Update the project structure tree. Remove references to `src/backend/` and `src/frontend/`.

- [ ] **Step 2: Update backend CLAUDE.md**

Update all paths from `src/backend/` to `backend/app/`. Update the project structure tree. Update the "Before Committing" commands.

- [ ] **Step 3: Update frontend CLAUDE.md**

Update all paths from `src/frontend/` to `frontend/`. Update the project structure tree.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md backend/CLAUDE.md frontend/CLAUDE.md
git commit -m "docs: update CLAUDE.md files for new directory structure"
```

---

## Verification

### Task 14: Full verification

- [ ] **Step 1: Python imports resolve**

```bash
cd /home/michael/Github/skating-biomechanics-ml
uv run python -c "from backend.app.main import app; print('backend OK')"
uv run python -c "from src.types import H36Key; print('ml OK')"
uv run python -c "from src.worker import WorkerSettings; print('worker OK')"
```

- [ ] **Step 2: Backend typecheck**

```bash
uv run basedpyright --level error backend/app
```

- [ ] **Step 3: ML typecheck**

```bash
uv run basedpyright --level error ml/src
```

- [ ] **Step 4: Run ML tests**

```bash
uv run pytest ml/tests/ -v -m "not slow" --tb=short
```

- [ ] **Step 5: Run backend lint**

```bash
uv run ruff check backend/app --fix
```

- [ ] **Step 6: Run ML lint**

```bash
uv run ruff check ml/src --fix
```

- [ ] **Step 7: Frontend builds**

```bash
cd frontend && bunx tsc --noEmit
```

- [ ] **Step 8: Verify no stale src/ imports remain**

```bash
grep -rn "from src\." backend/ ml/ frontend/ --include='*.py' --include='*.ts' --include='*.tsx'
```
Expected: No output (zero matches)

- [ ] **Step 9: Verify backend has zero ML imports**

```bash
grep -rn "from src\." backend/ --include='*.py'
```
Expected: No output (zero matches)

- [ ] **Step 10: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix(repo): address remaining issues from restructure"
```
