# arq + Valkey Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ThreadPoolExecutor with arq+valkey task queue for video processing, enabling future serverless GPU scaling.

**Architecture:** Single arq worker wraps the existing sync `process_video_pipeline()` via `asyncio.to_thread()`. Task state (status/progress/result/cancel) stored in Valkey. FastAPI endpoints enqueue jobs and expose polling API. Frontend switches from SSE to polling.

**Tech Stack:** arq, redis (valkey-compatible), pydantic-settings, podman-compose

---

## Context

The current video processing pipeline runs in a `ThreadPoolExecutor(max_workers=1)` with direct SSE streaming. This blocks the API process, prevents scaling, and doesn't survive restarts. The sibling project `Control_Plane` uses arq+valkey successfully — we replicate those patterns here.

Key constraint: RTX 3050 Ti (4GB VRAM), single GPU — `worker_max_jobs=1`.

---

## Task 1: Infrastructure — Config, Dependencies, Valkey Service

**Files:**
- Create: `src/config.py`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

### Step 1: Add dependencies to `pyproject.toml`

Add to `dependencies`:
```
"arq>=0.26.0",
"redis>=5.0.0",
"pydantic-settings>=2.0.0",
```

### Step 2: Create `src/config.py`

```python
"""Application settings for the skating biomechanics web API and worker."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central settings, loaded from environment variables or .env file."""

    valkey_host: str = "localhost"
    valkey_port: int = 6379
    valkey_db: int = 0
    valkey_password: str | None = None

    outputs_dir: str = "data/uploads"
    worker_max_jobs: int = 1
    worker_retry_delays: list[int] = [30, 120]
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    task_ttl_seconds: int = 24 * 60 * 60

    def build_valkey_url(self) -> str:
        auth = f":{self.valkey_password}@" if self.valkey_password else ""
        return f"redis://{auth}{self.valkey_host}:{self.valkey_port}/{self.valkey_db}"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


def get_settings() -> Settings:
    if get_settings._instance is None:  # type: ignore[attr-defined]
        get_settings._instance = Settings()  # type: ignore[attr-defined]
    return get_settings._instance  # type: ignore[attr-defined]


get_settings._instance = None  # type: ignore[attr-defined]
```

### Step 3: Create `docker-compose.yml`

```yaml
services:
  valkey:
    image: docker.io/valkey/valkey:alpine
    restart: unless-stopped
    ports:
      - "127.0.0.1:${VALKEY_HOST_PORT:-6379}:6379"
    healthcheck:
      test: ["CMD", "valkey-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 5s
    volumes:
      - valkey-data:/data

volumes:
  valkey-data:
```

### Step 4: Create `.env.example`

```
VALKEY_HOST=localhost
VALKEY_PORT=6379
VALKEY_DB=0
# VALKEY_PASSWORD=
OUTPUTS_DIR=data/uploads
WORKER_MAX_JOBS=1
LOG_LEVEL=INFO
```

### Step 5: Add `.env` to `.gitignore`

Append `.env` if not already present.

### Step 6: Install dependencies

```bash
uv sync
```

### Step 7: Start Valkey

```bash
podman compose up -d valkey
```

### Step 8: Verify config loads

```bash
uv run python -c "from src.config import get_settings; s = get_settings(); print(s.build_valkey_url())"
```

Expected: `redis://localhost:6379/0`

### Step 9: Verify Valkey connectivity

```bash
uv run python -c "
import redis.asyncio as aioredis
import asyncio
async def main():
    r = aioredis.Redis(host='localhost', port=6379, decode_responses=True)
    await r.ping()
    print('Valkey OK')
    await r.close()
asyncio.run(main())
"
```

### Step 10: Commit

```bash
git add src/config.py docker-compose.yml .env.example pyproject.toml .gitignore uv.lock
git commit -m "feat(infra): add pydantic-settings, valkey service, and docker-compose"
```

---

## Task 2: Task Manager + Worker

**Files:**
- Create: `src/task_manager.py`
- Create: `src/worker.py`
- Modify: `src/web_helpers.py` (import `PipelineCancelled`)

### Step 1: Create `src/task_manager.py`

```python
"""Valkey-backed task state management for video processing jobs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import redis.asyncio as aioredis

from src.config import get_settings

logger = logging.getLogger(__name__)

TASK_KEY_PREFIX = "task:"
TASK_CANCEL_PREFIX = "task_cancel:"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


async def get_valkey_client() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.Redis(
        host=settings.valkey_host,
        port=settings.valkey_port,
        db=settings.valkey_db,
        password=settings.valkey_password,
        decode_responses=True,
    )


async def create_task_state(
    task_id: str,
    video_path: str,
    valkey: aioredis.Redis | None = None,
) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        ttl = get_settings().task_ttl_seconds
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={
                "task_id": task_id,
                "status": TaskStatus.PENDING,
                "video_path": video_path,
                "progress": "0.0",
                "message": "Queued",
                "created_at": now,
                "started_at": "",
                "completed_at": "",
                "error": "",
            },
        )
        await valkey.expire(f"{TASK_KEY_PREFIX}{task_id}", ttl)
    finally:
        if close:
            await valkey.close()


async def update_progress(
    task_id: str,
    fraction: float,
    message: str,
    valkey: aioredis.Redis | None = None,
) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={"progress": str(round(fraction, 3)), "message": message},
        )
    finally:
        if close:
            await valkey.close()


async def store_result(
    task_id: str,
    result: dict[str, Any],
    valkey: aioredis.Redis | None = None,
) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={
                "status": TaskStatus.COMPLETED,
                "progress": "1.0",
                "message": "Done",
                "completed_at": now,
                "result": json.dumps(result),
            },
        )
    finally:
        if close:
            await valkey.close()


async def store_error(
    task_id: str,
    error_message: str,
    valkey: aioredis.Redis | None = None,
) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={
                "status": TaskStatus.FAILED,
                "completed_at": now,
                "error": error_message,
            },
        )
    finally:
        if close:
            await valkey.close()


async def mark_cancelled(task_id: str, valkey: aioredis.Redis | None = None) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={
                "status": TaskStatus.CANCELLED,
                "completed_at": now,
                "message": "Cancelled",
            },
        )
    finally:
        if close:
            await valkey.close()


async def get_task_state(task_id: str, valkey: aioredis.Redis | None = None) -> dict[str, Any] | None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        data = await valkey.hgetall(f"{TASK_KEY_PREFIX}{task_id}")
        if not data:
            return None
        result = data.get("result")
        data["result"] = json.loads(result) if result else None
        data["progress"] = float(data.get("progress", "0"))
        return data
    finally:
        if close:
            await valkey.close()


async def is_cancelled(task_id: str, valkey: aioredis.Redis | None = None) -> bool:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        return await valkey.get(f"{TASK_CANCEL_PREFIX}{task_id}") == "1"
    finally:
        if close:
            await valkey.close()


async def set_cancel_signal(task_id: str, valkey: aioredis.Redis | None = None) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        ttl = get_settings().task_ttl_seconds
        await valkey.setex(f"{TASK_CANCEL_PREFIX}{task_id}", ttl, "1")
    finally:
        if close:
            await valkey.close()
```

### Step 2: Write failing test for task_manager

Create `tests/test_task_manager.py`:

```python
"""Tests for Valkey-backed task state management."""

import pytest

from src.task_manager import (
    TaskStatus,
    create_task_state,
    get_task_state,
    is_cancelled,
    mark_cancelled,
    set_cancel_signal,
    store_error,
    store_result,
    update_progress,
)


@pytest.fixture
async def valkey():
    import redis.asyncio as aioredis

    client = aioredis.Redis(host="localhost", port=6379, db=1, decode_responses=True)
    yield client
    keys = await client.keys("test_task:*")
    if keys:
        await client.delete(*keys)
    keys = await client.keys("test_task_cancel:*")
    if keys:
        await client.delete(*keys)
    await client.close()


@pytest.mark.asyncio
async def test_create_and_get_state(valkey):
    await create_task_state("test_task_1", "/tmp/video.mp4", valkey=valkey)
    state = await get_task_state("test_task_1", valkey=valkey)

    assert state is not None
    assert state["task_id"] == "test_task_1"
    assert state["status"] == TaskStatus.PENDING
    assert state["progress"] == 0.0
    assert state["video_path"] == "/tmp/video.mp4"


@pytest.mark.asyncio
async def test_update_progress(valkey):
    await create_task_state("test_task_2", "/tmp/video.mp4", valkey=valkey)
    await update_progress("test_task_2", 0.5, "Rendering...", valkey=valkey)
    state = await get_task_state("test_task_2", valkey=valkey)

    assert state["progress"] == 0.5
    assert state["message"] == "Rendering..."


@pytest.mark.asyncio
async def test_store_result(valkey):
    await create_task_state("test_task_3", "/tmp/video.mp4", valkey=valkey)
    await store_result("test_task_3", {"video_path": "out.mp4", "stats": {}}, valkey=valkey)
    state = await get_task_state("test_task_3", valkey=valkey)

    assert state["status"] == TaskStatus.COMPLETED
    assert state["result"] == {"video_path": "out.mp4", "stats": {}}


@pytest.mark.asyncio
async def test_store_error(valkey):
    await create_task_state("test_task_4", "/tmp/video.mp4", valkey=valkey)
    await store_error("test_task_4", "OOM", valkey=valkey)
    state = await get_task_state("test_task_4", valkey=valkey)

    assert state["status"] == TaskStatus.FAILED
    assert state["error"] == "OOM"


@pytest.mark.asyncio
async def test_cancel_flow(valkey):
    await create_task_state("test_task_5", "/tmp/video.mp4", valkey=valkey)
    assert not await is_cancelled("test_task_5", valkey=valkey)

    await set_cancel_signal("test_task_5", valkey=valkey)
    assert await is_cancelled("test_task_5", valkey=valkey)

    await mark_cancelled("test_task_5", valkey=valkey)
    state = await get_task_state("test_task_5", valkey=valkey)
    assert state["status"] == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(valkey):
    state = await get_task_state("nonexistent", valkey=valkey)
    assert state is None
```

### Step 3: Run task_manager tests

```bash
uv run pytest tests/test_task_manager.py -v
```

Expected: all 6 pass (requires running Valkey)

### Step 4: Create `src/worker.py`

```python
"""arq worker for video processing pipeline.

Run with: uv run python -m src.worker
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from arq import Retry
from arq.connections import RedisSettings

from src.config import get_settings
from src.task_manager import (
    TaskStatus,
    create_task_state,
    get_valkey_client,
    is_cancelled,
    mark_cancelled,
    store_error,
    store_result,
    update_progress,
)

logger = logging.getLogger(__name__)

_cancel_events: dict[str, threading.Event] = {}


async def startup(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker starting up")


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker shutting down")
    _cancel_events.clear()


async def _poll_cancel(task_id: str, cancel_event: threading.Event) -> None:
    """Poll Valkey cancel signal and set threading event when detected."""
    valkey = await get_valkey_client()
    try:
        while not cancel_event.is_set():
            if await is_cancelled(task_id, valkey=valkey):
                cancel_event.set()
                break
            await asyncio.sleep(0.5)
    finally:
        await valkey.close()


async def _async_update_progress(task_id: str, fraction: float, message: str) -> None:
    valkey = await get_valkey_client()
    try:
        await update_progress(task_id, fraction, message, valkey=valkey)
    finally:
        await valkey.close()


async def process_video_task(
    ctx: dict[str, Any],
    *,
    task_id: str,
    video_path: str,
    person_click: dict[str, int],
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    depth: bool = False,
    optical_flow: bool = False,
    segment: bool = False,
    foot_track: bool = False,
    matting: bool = False,
    inpainting: bool = False,
) -> dict[str, Any]:
    """arq task: run process_video_pipeline() with Valkey state tracking."""
    settings = get_settings()
    valkey = await get_valkey_client()

    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"task:{task_id}",
            mapping={"status": TaskStatus.RUNNING, "started_at": now},
        )

        cancel_event = threading.Event()
        _cancel_events[task_id] = cancel_event

        outputs_dir = Path(settings.outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(outputs_dir / f"{Path(video_path).stem}_analyzed.mp4")

        def progress_cb(fraction: float, message: str) -> None:
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    _async_update_progress, task_id, fraction, message
                )
            except RuntimeError:
                pass

        poll_task = asyncio.create_task(_poll_cancel(task_id, cancel_event))

        try:
            from src.types import PersonClick
            from src.web_helpers import PipelineCancelled, process_video_pipeline

            click = PersonClick(x=person_click["x"], y=person_click["y"])
            result = await asyncio.to_thread(
                process_video_pipeline,
                video_path=video_path,
                person_click=click,
                frame_skip=frame_skip,
                layer=layer,
                tracking=tracking,
                blade_3d=False,
                export=export,
                output_path=output_path,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
                depth=depth,
                optical_flow=optical_flow,
                segment=segment,
                foot_track=foot_track,
                matting=matting,
                inpainting=inpainting,
            )
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass

        stats = result["stats"]
        out_path = Path(output_path)
        video_rel = (
            str(out_path.relative_to(outputs_dir))
            if out_path.is_relative_to(outputs_dir)
            else out_path.name
        )
        poses_rel = None
        csv_rel = None
        if result.get("poses_path"):
            pp = Path(result["poses_path"])
            poses_rel = (
                str(pp.relative_to(outputs_dir))
                if pp.is_relative_to(outputs_dir)
                else pp.name
            )
        if result.get("csv_path"):
            cp = Path(result["csv_path"])
            csv_rel = (
                str(cp.relative_to(outputs_dir))
                if cp.is_relative_to(outputs_dir)
                else cp.name
            )

        response_data = {
            "video_path": video_rel,
            "poses_path": poses_rel,
            "csv_path": csv_rel,
            "stats": stats,
            "status": "Analysis complete!",
        }
        await store_result(task_id, response_data, valkey=valkey)
        return response_data

    except PipelineCancelled:
        await mark_cancelled(task_id, valkey=valkey)
        return {"status": "cancelled", "task_id": task_id}

    except Exception as e:
        logger.exception("Pipeline task %s failed", task_id)
        await store_error(task_id, str(e), valkey=valkey)
        error_msg = str(e).lower()
        if any(term in error_msg for term in ["timeout", "connection", "network"]):
            raise Retry(defer=ctx.get("job_try", 1) * 10) from e
        raise

    finally:
        _cancel_events.pop(task_id, None)
        await valkey.close()


_settings = get_settings()


class WorkerSettings:
    """arq worker configuration."""

    queue_name: str = "skating:queue"
    max_jobs: int = _settings.worker_max_jobs
    retry_jobs: bool = True
    retry_delays: ClassVar[list[int]] = _settings.worker_retry_delays

    on_startup = startup
    on_shutdown = shutdown

    functions: ClassVar[list] = [process_video_task]
    cron_jobs: ClassVar[list] = []

    redis_settings = RedisSettings(
        host=_settings.valkey_host,
        port=_settings.valkey_port,
        database=_settings.valkey_db,
        password=_settings.valkey_password,
    )
```

### Step 5: Verify worker starts (no crash)

```bash
timeout 5 uv run python -m src.worker 2>&1 || true
```

Expected output contains: `Video processing worker starting up` (no import errors)

### Step 6: Commit

```bash
git add src/task_manager.py src/worker.py tests/test_task_manager.py
git commit -m "feat(worker): add arq worker with Valkey task state management"
```

---

## Task 3: API Endpoints + Frontend Polling

**Files:**
- Modify: `src/backend/schemas.py`
- Modify: `src/backend/routes/process.py`
- Modify: `src/frontend/src/lib/api.ts`
- Modify: `src/frontend/src/lib/schemas.ts`
- Modify: `src/frontend/src/pages/AnalyzePage.tsx`

### Step 1: Add schemas to `src/backend/schemas.py`

Add after existing `ProcessResponse` class:

```python
class QueueProcessResponse(BaseModel):
    """Response for POST /api/process/queue."""

    task_id: str
    status: str = "pending"


class TaskStatusResponse(BaseModel):
    """Response for GET /api/process/{task_id}/status."""

    task_id: str
    status: str
    progress: float
    message: str
    result: ProcessResponse | None = None
    error: str | None = None
```

### Step 2: Add queue endpoints to `src/backend/routes/process.py`

Add these imports at the top of the file (alongside existing ones):

```python
import uuid
from arq import create_pool
from arq.connections import RedisSettings

from src.config import get_settings
from src.task_manager import (
    create_task_state,
    get_task_state,
    set_cancel_signal,
)
```

Add these endpoints **after** the existing `cancel_processing` endpoint:

```python
@router.post("/api/process/queue", response_model=QueueProcessResponse)
async def enqueue_process(req: ProcessRequest):
    """Enqueue video processing job and return task_id immediately."""
    settings = get_settings()
    task_id = f"proc_{uuid.uuid4().hex[:12]}"

    valkey = await get_valkey_client()
    try:
        await create_task_state(task_id, video_path=req.video_path, valkey=valkey)
    finally:
        await valkey.close()

    arq_pool = await create_pool(
        RedisSettings(
            host=settings.valkey_host,
            port=settings.valkey_port,
            database=settings.valkey_db,
            password=settings.valkey_password,
        )
    )
    try:
        await arq_pool.enqueue_job(
            "process_video_task",
            task_id=task_id,
            video_path=req.video_path,
            person_click={"x": req.person_click.x, "y": req.person_click.y},
            frame_skip=req.frame_skip,
            layer=req.layer,
            tracking=req.tracking,
            export=req.export,
            depth=req.depth,
            optical_flow=req.optical_flow,
            segment=req.segment,
            foot_track=req.foot_track,
            matting=req.matting,
            inpainting=req.inpainting,
        )
    finally:
        await arq_pool.close()

    return QueueProcessResponse(task_id=task_id)


@router.get("/api/process/{task_id}/status", response_model=TaskStatusResponse)
async def get_process_status(task_id: str):
    """Poll task status."""
    valkey = await get_valkey_client()
    try:
        state = await get_task_state(task_id, valkey=valkey)
    finally:
        await valkey.close()

    if state is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")

    result = None
    if state.get("result"):
        result = ProcessResponse(**state["result"])

    return TaskStatusResponse(
        task_id=task_id,
        status=state["status"],
        progress=state["progress"],
        message=state.get("message", ""),
        result=result,
        error=state.get("error"),
    )


@router.post("/api/process/{task_id}/cancel")
async def cancel_queued_process(task_id: str):
    """Cancel a queued or running task via Valkey signal."""
    await set_cancel_signal(task_id)
    return {"status": "cancel_requested", "task_id": task_id}
```

### Step 3: Add frontend API functions to `src/frontend/src/lib/api.ts`

Add after `cancelProcessing`:

```typescript
export async function enqueueProcess(
  request: ProcessRequest,
): Promise<{ task_id: string }> {
  const validated = ProcessRequestSchema.parse(request)
  const res = await fetch(`${API_BASE}/process/queue`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(validated),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text)
  }
  return res.json()
}

export interface TaskStatusResponse {
  task_id: string
  status: "pending" | "running" | "completed" | "failed" | "cancelled"
  progress: number
  message: string
  result: ProcessResponse | null
  error: string | null
}

export async function pollTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  const res = await fetch(`${API_BASE}/process/${taskId}/status`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text)
  }
  return res.json()
}

export async function cancelQueuedProcess(taskId: string): Promise<void> {
  await fetch(`${API_BASE}/process/${taskId}/cancel`, { method: "POST" })
}
```

### Step 4: Add Zod schema to `src/frontend/src/lib/schemas.ts`

Add after `ProcessResponseSchema`:

```typescript
export const TaskStatusResponseSchema = z.object({
  task_id: z.string().min(1),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]),
  progress: z.number().min(0).max(1),
  message: z.string(),
  result: ProcessResponseSchema.nullable(),
  error: z.string().nullable(),
})
```

### Step 5: Update `AnalyzePage.tsx` to use polling

Replace the `processVideo` import and SSE-based `startProcessing` with polling:

```typescript
import { cancelQueuedProcess, enqueueProcess, pollTaskStatus } from "@/lib/api"
```

Replace the `startProcessing` callback with:

```typescript
  const taskIdRef = useRef<string | null>(null)

  const startProcessing = useCallback(() => {
    if (startedRef.current) return
    startedRef.current = true

    setPhase("processing")
    setProgress(0)
    setMessage("Queuing analysis...")

    let cancelled = false

    enqueueProcess(processRequest)
      .then((res) => {
        taskIdRef.current = res.task_id
        setMessage("Waiting for worker...")
        const poll = setInterval(async () => {
          if (cancelled) {
            clearInterval(poll)
            return
          }
          try {
            const status = await pollTaskStatus(res.task_id)
            setProgress(Math.round(status.progress * 100))
            setMessage(status.message)

            if (status.status === "completed" && status.result) {
              clearInterval(poll)
              setResult(status.result)
              setPhase("done")
            } else if (status.status === "failed") {
              clearInterval(poll)
              setError(status.error || "Unknown error")
              setPhase("error")
              startedRef.current = false
            } else if (status.status === "cancelled") {
              clearInterval(poll)
              setPhase("cancelled")
              startedRef.current = false
            }
          } catch {
            // Network error — keep polling
          }
        }, 1000)
      })
      .catch((err) => {
        setError(err.message)
        setPhase("error")
        startedRef.current = false
      })
  }, [processRequest])
```

Update `handleCancel`:

```typescript
  const handleCancel = useCallback(async () => {
    try {
      if (taskIdRef.current) {
        await cancelQueuedProcess(taskIdRef.current)
      }
    } catch {
      // ignore
    }
    setPhase("cancelled")
    startedRef.current = false
  }, [])
```

Remove unused imports: `processVideo`, `cancelProcessing`, `AbortController` (from api.ts). Remove `abortRef`.

### Step 6: TypeScript check

```bash
cd src/frontend && npx tsc --noEmit
```

### Step 7: Commit

```bash
git add src/backend/schemas.py src/backend/routes/process.py \
  src/frontend/src/lib/api.ts src/frontend/src/lib/schemas.ts \
  src/frontend/src/pages/AnalyzePage.tsx
git commit -m "feat(api): add queue-based processing with polling endpoints"
```

---

## Verification

### End-to-end test (manual)

1. Start Valkey:
```bash
podman compose up -d valkey
```

2. Start worker (terminal 1):
```bash
uv run python -m src.worker
```

3. Start API (terminal 2):
```bash
uv run uvicorn src.backend.main:app --reload
```

4. Start frontend (terminal 3):
```bash
cd src/frontend && npm run dev
```

5. Test flow:
   - Upload video → select person → click "Analyze"
   - Observe progress polling on AnalyzePage
   - Cancel → verify "Cancelled" state
   - Re-analyze → verify result appears

### Check list
- [ ] `uv sync` installs arq, redis, pydantic-settings
- [ ] `podman compose up -d valkey` starts successfully
- [ ] `uv run python -m src.worker` starts without errors
- [ ] `POST /api/process/queue` returns task_id
- [ ] `GET /api/process/{id}/status` returns progress updates
- [ ] `POST /api/process/{id}/cancel` sets cancelled status
- [ ] Frontend polls every 1s and shows progress
- [ ] Cancel button works from frontend
- [ ] Old SSE endpoint still works (backward compat)
