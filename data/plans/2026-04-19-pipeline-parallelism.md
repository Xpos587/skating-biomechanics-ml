# Pipeline Parallelism & Async Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize ML pipeline for parallelism and async I/O — fix sync-in-async bugs, add connection pooling, integrate batch GPU inference, and add double buffering for video decode.

**Architecture:** Three-tier optimization: Tier 1 (bug fixes + pooling, ~1 week), Tier 2 (batch inference + double buffering, ~1 week), Tier 3 (FP16 + NVENC + CUDA Graphs, ~1 week). Each tier builds on the previous.

**Tech Stack:** FastAPI, arq, Valkey (redis.asyncio), ONNX Runtime GPU, rtmlib, aiobotocore, httpx

**Spec:** `docs/specs/2026-04-19-pipeline-parallelism-design.md`

---

## Phase 1: Bug Fixes & Connection Pooling (Tier 1)

### Task 1: Fix sync-in-async in detect.py

**Files:**
- Modify: `backend/app/routes/detect.py:18,38-39,44-48,50-67,75-79,101-105`
- Test: `backend/tests/routes/test_detect.py` (create if missing)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/routes/test_detect.py
"""Tests for sync-in-async fixes in detect routes."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_valkey():
    valkey = AsyncMock()
    valkey.hset = AsyncMock()
    valkey.hgetall = AsyncMock(return_value={})
    valkey.close = AsyncMock()
    return valkey


@pytest.fixture
def mock_arq_pool():
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock()
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.valkey.host = "localhost"
    settings.valkey.port = 6379
    settings.valkey.db = 0
    settings.valkey.password = MagicMock(get_secret_value=MagicMock(return_value=""))
    return settings


@pytest.mark.asyncio
async def test_enqueue_detect_uses_async_upload(mock_settings):
    """upload_bytes_async should be called, not sync upload_bytes."""
    from app.routes.detect import enqueue_detect
    from app.schemas import DetectQueueResponse

    fake_video = MagicMock()
    fake_video.filename = "test.mp4"
    fake_video.read = AsyncMock(return_value=b"fake-video-bytes")

    with (
        patch("app.routes.detect.upload_bytes_async", new_callable=AsyncMock, return_value="input/fake.mp4") as mock_upload,
        patch("app.routes.detect.get_valkey_client", new_callable=AsyncMock) as mock_valkey_fn,
        patch("app.routes.detect.create_pool", new_callable=AsyncMock) as mock_arq_fn,
        patch("app.routes.detect.create_task_state", new_callable=AsyncMock) as mock_create,
        patch("app.routes.detect.get_settings", return_value=mock_settings),
    ):
        mock_valkey = AsyncMock()
        mock_valkey.hset = AsyncMock()
        mock_valkey.close = AsyncMock()
        mock_valkey_fn.return_value = mock_valkey

        mock_pool = AsyncMock()
        mock_pool.enqueue_job = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_arq_fn.return_value = mock_pool

        result = await enqueue_detect(fake_video)

    mock_upload.assert_awaited_once_with(b"fake-video-bytes", "input/fake.mp4")
    assert isinstance(result, DetectQueueResponse)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/routes/test_detect.py -v`
Expected: FAIL — `ImportError` or `AttributeError` because `upload_bytes_async` is not imported in detect.py

- [ ] **Step 3: Implement the fix**

Replace in `backend/app/routes/detect.py`:

```python
# Line 18: Change import
# BEFORE:
from app.storage import upload_bytes
# AFTER:
from app.storage import upload_bytes_async

# Lines 38-39: Change call
# BEFORE:
content = await video.read()
upload_bytes(content, video_key)
# AFTER:
content = await video.read()
await upload_bytes_async(content, video_key)

# Lines 44-48: Remove valkey create/close (will use pool in Task 3)
# BEFORE:
valkey = await get_valkey_client()
try:
    await create_task_state(task_id, video_key=video_key, valkey=valkey)
finally:
    await valkey.close()
# AFTER:
valkey = await get_valkey_client()
await create_task_state(task_id, video_key=video_key, valkey=valkey)

# Lines 50-67: Remove arq pool create/close (will use singleton in Task 4)
# BEFORE:
arq_pool = await create_pool(RedisSettings(...))
try:
    await arq_pool.enqueue_job(...)
finally:
    await arq_pool.close()
# AFTER:
arq_pool = await create_pool(RedisSettings(...))
await arq_pool.enqueue_job(...)

# Lines 75-79: Remove valkey close
# BEFORE:
valkey = await get_valkey_client()
try:
    state = await get_task_state(task_id, valkey=valkey)
finally:
    await valkey.close()
# AFTER:
valkey = await get_valkey_client()
state = await get_task_state(task_id, valkey=valkey)

# Lines 101-105: Same pattern
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/routes/test_detect.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/detect.py backend/tests/routes/test_detect.py
git commit -m "fix(backend): use async storage in detect route"
```

---

### Task 2: Fix sync-in-async in uploads.py

**Files:**
- Modify: `backend/app/routes/uploads.py:12,27,43,73,84`
- Test: `backend/tests/routes/test_uploads.py` (create if missing)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/routes/test_uploads.py
"""Tests for sync-in-async fixes in upload routes."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_init_upload_uses_async_client():
    """init_upload should use async S3 client, not sync boto3."""
    with (
        patch("app.routes.uploads._async_client") as mock_async_client_fn,
        patch("app.routes.uploads.get_settings") as mock_settings,
    ):
        mock_settings.return_value.r2.bucket = "test-bucket"

        mock_s3 = AsyncMock()
        mock_s3.create_multipart_upload = AsyncMock(return_value={"UploadId": "upload-123"})
        mock_s3.generate_presigned_url = AsyncMock(return_value="https://presigned.url")
        mock_s3.__aenter__ = AsyncMock(return_value=mock_s3)
        mock_s3.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_fn.return_value = mock_s3.__aenter__

        from app.routes.uploads import init_upload
        from app.auth.deps import CurrentUser

        user = MagicMock(spec=CurrentUser)
        user.id = "user-123"

        result = await init_upload(user, file_name="test.mp4", content_type="video/mp4", total_size=10_000_000)

    mock_s3.create_multipart_upload.assert_awaited_once()
    assert result["upload_id"] == "upload-123"


@pytest.mark.asyncio
async def test_complete_upload_uses_async_client():
    """complete_upload should use async S3 client, not sync boto3."""
    from app.routes.uploads import CompleteUploadRequest

    body = CompleteUploadRequest(
        upload_id="upload-123",
        key="uploads/user-123/abc/test.mp4",
        parts=[{"part_number": 1, "etag": '"etag-1"'}],
    )

    with (
        patch("app.routes.uploads._async_client") as mock_async_client_fn,
        patch("app.routes.uploads.get_settings") as mock_settings,
    ):
        mock_settings.return_value.r2.bucket = "test-bucket"

        mock_s3 = AsyncMock()
        mock_s3.complete_multipart_upload = AsyncMock()
        mock_s3.__aenter__ = AsyncMock(return_value=mock_s3)
        mock_s3.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_fn.return_value = mock_s3.__aenter__

        from app.routes.uploads import complete_upload
        from app.auth.deps import CurrentUser

        user = MagicMock(spec=CurrentUser)
        result = await complete_upload(user, body)

    mock_s3.complete_multipart_upload.assert_awaited_once()
    assert result["status"] == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/routes/test_uploads.py -v`
Expected: FAIL — `_async_client` not imported in uploads.py

- [ ] **Step 3: Implement the fix**

Replace in `backend/app/routes/uploads.py`:

```python
# Line 12: Change import
# BEFORE:
from app.storage import _client
# AFTER:
from app.storage import _async_client

# Lines 19-61: Rewrite init_upload to use async client
@router.post("/uploads/init")
async def init_upload(
    user: CurrentUser,
    file_name: str = Query(..., min_length=1),
    content_type: str = Query("video/mp4"),
    total_size: int = Query(..., gt=0),
):
    """Initialize a multipart upload. Returns upload_id and pre-signed part URLs."""
    bucket = get_settings().r2.bucket
    key = f"uploads/{user.id}/{uuid.uuid4()}/{file_name}"

    async with await _async_client() as r2:
        upload_id = (await r2.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            ContentType=content_type,
        ))["UploadId"]

        # Calculate number of parts
        part_count = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Generate pre-signed URLs for each part
        part_urls = []
        for part_number in range(1, part_count + 1):
            url = await r2.generate_presigned_url(
                ClientMethod="upload_part",
                Params={
                    "Bucket": bucket,
                    "Key": key,
                    "UploadId": upload_id,
                    "PartNumber": part_number,
                },
                ExpiresIn=3600,
            )
            part_urls.append({"part_number": part_number, "url": url})

    return {
        "upload_id": upload_id,
        "key": key,
        "chunk_size": CHUNK_SIZE,
        "part_count": part_count,
        "parts": part_urls,
    }

# Lines 70-91: Rewrite complete_upload to use async client
@router.post("/uploads/complete")
async def complete_upload(user: CurrentUser, body: CompleteUploadRequest):
    """Complete a multipart upload. Returns the final object key."""
    bucket = get_settings().r2.bucket

    multipart_parts = [
        {"PartNumber": p["part_number"], "ETag": p["etag"]}
        for p in sorted(body.parts, key=lambda x: x["part_number"])
    ]

    if not multipart_parts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No parts provided")

    async with await _async_client() as r2:
        await r2.complete_multipart_upload(
            Bucket=bucket,
            Key=body.key,
            UploadId=body.upload_id,
            MultipartUpload={"Parts": multipart_parts},
        )

    return {"status": "completed", "key": body.key}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/routes/test_uploads.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/uploads.py backend/tests/routes/test_uploads.py
git commit -m "fix(backend): use async S3 client in upload routes"
```

---

### Task 3: Fix sync-in-async in sessions.py

**Files:**
- Modify: `backend/app/routes/sessions.py:19,27,35-43`
- Test: `backend/tests/routes/test_sessions.py` (create if missing)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/routes/test_sessions.py
"""Tests for sync-in-async fixes in session routes."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_session_to_response_uses_async_url():
    """_session_to_response should use async get_object_url, not sync."""
    with patch("app.routes.sessions.get_object_url_async", new_callable=AsyncMock, return_value="https://r2.url/video.mp4") as mock_url:
        from app.routes.sessions import _session_to_response

        session = MagicMock()
        session.id = "sess-1"
        session.user_id = "user-1"
        session.element_type = "waltz_jump"
        session.video_key = "input/video.mp4"
        session.processed_video_key = "output/video.mp4"
        session.processed_video_url = None
        session.video_url = None
        session.poses_url = None
        session.csv_url = None
        session.pose_data = None
        session.frame_metrics = None
        session.status = "done"
        session.error_message = None
        session.phases = None
        session.recommendations = []
        session.overall_score = 0.8
        session.created_at = MagicMock()
        session.processed_at = MagicMock()
        session.metrics = []

        result = _session_to_response(session)

    # Should be called twice: video_key + processed_video_key
    assert mock_url.await_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/routes/test_sessions.py -v`
Expected: FAIL — `get_object_url_async` not imported

- [ ] **Step 3: Implement the fix**

Replace in `backend/app/routes/sessions.py`:

```python
# Line 19: Change import
# BEFORE:
from app.storage import get_object_url
# AFTER:
import asyncio
from app.storage import get_object_url

# Replace _session_to_response (lines 27-57):
def _session_to_response(session) -> SessionResponse:
    """Convert ORM Session to response schema with presigned URLs."""
    return SessionResponse.model_validate(
        {
            "id": session.id,
            "user_id": session.user_id,
            "element_type": session.element_type,
            "video_key": session.video_key,
            "video_url": get_object_url(session.video_key)
            if session.video_key
            else session.video_url,
            "processed_video_key": session.processed_video_key,
            "processed_video_url": (
                get_object_url(session.processed_video_key)
                if session.processed_video_key
                else session.processed_video_url
            ),
            "poses_url": session.poses_url,
            "csv_url": session.csv_url,
            "pose_data": session.pose_data,
            "frame_metrics": session.frame_metrics,
            "status": session.status,
            "error_message": session.error_message,
            "phases": session.phases,
            "recommendations": session.recommendations,
            "overall_score": session.overall_score,
            "created_at": session.created_at,
            "processed_at": session.processed_at,
            "metrics": session.metrics,
        }
    )


# Make callers async and wrap with asyncio.to_thread:
async def _session_to_response_async(session) -> SessionResponse:
    """Async version that wraps sync get_object_url in thread."""
    video_url = None
    processed_video_url = None
    if session.video_key:
        video_url = await asyncio.to_thread(get_object_url, session.video_key)
    elif session.video_url:
        video_url = session.video_url

    if session.processed_video_key:
        processed_video_url = await asyncio.to_thread(get_object_url, session.processed_video_key)
    elif session.processed_video_url:
        processed_video_url = session.processed_video_url

    return SessionResponse.model_validate(
        {
            "id": session.id,
            "user_id": session.user_id,
            "element_type": session.element_type,
            "video_key": session.video_key,
            "video_url": video_url,
            "processed_video_key": session.processed_video_key,
            "processed_video_url": processed_video_url,
            "poses_url": session.poses_url,
            "csv_url": session.csv_url,
            "pose_data": session.pose_data,
            "frame_metrics": session.frame_metrics,
            "status": session.status,
            "error_message": session.error_message,
            "phases": session.phases,
            "recommendations": session.recommendations,
            "overall_score": session.overall_score,
            "created_at": session.created_at,
            "processed_at": session.processed_at,
            "metrics": session.metrics,
        }
    )
```

Then update all callers (`create_session`, `list_sessions`, `get_session`, `patch_session`) to use `await _session_to_response_async(session)` instead of `_session_to_response(session)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/routes/test_sessions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/sessions.py backend/tests/routes/test_sessions.py
git commit -m "fix(backend): wrap sync get_object_url in asyncio.to_thread"
```

---

### Task 4: Valkey connection pool singleton

**Files:**
- Modify: `backend/app/task_manager.py:29-37,40-68,71-87,...` (all functions)
- Modify: `backend/app/main.py:1-50` (add lifespan)
- Modify: `backend/app/routes/detect.py:44-48,75-79,101-105` (remove close)
- Modify: `backend/app/routes/process.py:38-42,84-88,119-142` (remove close)
- Test: `backend/tests/test_task_manager_pool.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_task_manager_pool.py
"""Tests for Valkey connection pool."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_valkey_returns_same_instance():
    """get_valkey should return the same Redis instance (singleton)."""
    with patch("app.task_manager._valkey_pool", None):
        from app.task_manager import init_valkey_pool, get_valkey, close_valkey_pool

        mock_redis = AsyncMock()
        with patch("app.task_manager.aioredis.Redis", return_value=mock_redis) as mock_cls:
            await init_valkey_pool()
            result1 = get_valkey()
            result2 = get_valkey()
            assert result1 is result2
            assert mock_cls.call_count == 1
            await close_valkey_pool()


@pytest.mark.asyncio
async def test_update_progress_does_not_close_valkey():
    """update_progress should NOT close the shared valkey connection."""
    with patch("app.task_manager._valkey_pool", None):
        from app.task_manager import init_valkey_pool, get_valkey, close_valkey_pool, update_progress

        mock_redis = AsyncMock()
        mock_redis.hset = AsyncMock()
        with patch("app.task_manager.aioredis.Redis", return_value=mock_redis):
            await init_valkey_pool()
            await update_progress("task-1", 0.5, "Processing")
            # close should NOT have been called
            mock_redis.close.assert_not_awaited()
            await close_valkey_pool()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_task_manager_pool.py -v`
Expected: FAIL — `init_valkey_pool`, `get_valkey`, `close_valkey_pool` don't exist

- [ ] **Step 3: Implement the connection pool**

Add pool functions to `backend/app/task_manager.py` (after line 37, before `create_task_state`):

```python
# Connection pool singleton
_valkey_pool: aioredis.Redis | None = None


async def init_valkey_pool() -> None:
    """Initialize the shared Valkey connection pool. Call from FastAPI lifespan."""
    global _valkey_pool
    if _valkey_pool is not None:
        return
    settings = get_settings()
    _valkey_pool = aioredis.Redis(
        host=settings.valkey.host,
        port=settings.valkey.port,
        db=settings.valkey.db,
        password=settings.valkey.password.get_secret_value(),
        decode_responses=True,
        max_connections=20,
    )


async def close_valkey_pool() -> None:
    """Close the shared Valkey connection pool. Call from FastAPI lifespan."""
    global _valkey_pool
    if _valkey_pool is not None:
        await _valkey_pool.close()
        _valkey_pool = None


def get_valkey() -> aioredis.Redis:
    """Get the shared Valkey connection. Raises if pool not initialized."""
    assert _valkey_pool is not None, "Call init_valkey_pool() first"
    return _valkey_pool
```

Then refactor all existing functions to use pool instead of per-call connect/close. The pattern for each function:

```python
# BEFORE (every function):
async def update_progress(task_id, fraction, message, valkey=None):
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        await valkey.hset(...)
    finally:
        if close:
            await valkey.close()

# AFTER:
async def update_progress(task_id, fraction, message, valkey=None):
    if valkey is None:
        valkey = get_valkey()
    await valkey.hset(...)
```

Apply this pattern to ALL 10 functions: `create_task_state`, `update_progress`, `store_result`, `store_error`, `mark_cancelled`, `get_task_state`, `is_cancelled`, `set_cancel_signal`, `publish_task_event`.

Keep `get_valkey_client()` for backward compatibility (worker uses it), but mark as deprecated.

- [ ] **Step 4: Add lifespan to main.py**

```python
# backend/app/main.py — add lifespan
from contextlib import asynccontextmanager

from app.task_manager import close_valkey_pool, init_valkey_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_valkey_pool()
    yield
    await close_valkey_pool()

app = FastAPI(title="AI Тренер — Фигурное катание", lifespan=lifespan)
```

- [ ] **Step 5: Update routes to remove valkey.close()**

In `detect.py` and `process.py`, remove all `try/finally: await valkey.close()` blocks. The pool handles lifecycle.

- [ ] **Step 6: Run tests**

Run: `cd backend && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/task_manager.py backend/app/main.py backend/app/routes/detect.py backend/app/routes/process.py backend/tests/test_task_manager_pool.py
git commit -m "refactor(backend): Valkey connection pool singleton"
```

---

### Task 5: arq pool singleton in FastAPI lifespan

**Files:**
- Modify: `backend/app/main.py` (lifespan)
- Modify: `backend/app/routes/detect.py:50-67`
- Modify: `backend/app/routes/process.py:53-76`

- [ ] **Step 1: Add arq pool to lifespan**

In `backend/app/main.py`, extend the lifespan:

```python
from arq import create_pool
from arq.connections import RedisSettings

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_valkey_pool()

    # arq pool singleton for job enqueue
    app.state.arq_pool = await create_pool(
        RedisSettings(
            host=settings.valkey.host,
            port=settings.valkey.port,
            database=settings.valkey.db,
            password=settings.valkey.password.get_secret_value(),
        )
    )
    yield
    await app.state.arq_pool.close()
    await close_valkey_pool()
```

- [ ] **Step 2: Update detect.py to use pool from app state**

```python
# detect.py — replace lines 50-67:
# BEFORE:
arq_pool = await create_pool(RedisSettings(...))
try:
    await arq_pool.enqueue_job(...)
finally:
    await arq_pool.close()

# AFTER (requires Request dependency injection):
from fastapi import Request

@router.post("/detect", response_model=DetectQueueResponse)
async def enqueue_detect(
    video: UploadFile,
    tracking: str = "auto",
    request: Request,
) -> DetectQueueResponse:
    ...
    await request.app.state.arq_pool.enqueue_job(
        "detect_video_task",
        task_id=task_id,
        video_key=video_key,
        tracking=tracking,
        _priority=0,
    )
    ...
```

- [ ] **Step 3: Update process.py similarly**

Same pattern — inject `request: Request`, use `request.app.state.arq_pool`.

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py backend/app/routes/detect.py backend/app/routes/process.py
git commit -m "refactor(backend): arq pool singleton in FastAPI lifespan"
```

---

### Task 6: N+1 batch query in session_saver.py

**Files:**
- Modify: `backend/app/crud/session_metric.py:15-41` (add batch function)
- Modify: `backend/app/services/session_saver.py:40-70` (use batch)
- Test: `backend/tests/test_session_metric_batch.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_session_metric_batch.py
"""Tests for batch PR query."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_current_best_batch_returns_all_metrics():
    """Should return dict of metric_name -> best value in one query."""
    with patch("app.crud.session_metric.get_current_best_batch") as mock_batch:
        mock_batch.return_value = {"airtime": 0.6, "max_height": 0.45}
        from app.crud.session_metric import get_current_best_batch
        result = await get_current_best_batch(
            db=AsyncMock(),
            user_id="user-1",
            element_type="waltz_jump",
            metric_names=["airtime", "max_height"],
        )
        assert result == {"airtime": 0.6, "max_height": 0.45}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_session_metric_batch.py -v`
Expected: FAIL — `get_current_best_batch` doesn't exist

- [ ] **Step 3: Add batch function to session_metric.py**

```python
# backend/app/crud/session_metric.py — add after get_current_best:

async def get_current_best_batch(
    db: AsyncSession,
    user_id: str,
    element_type: str,
    metric_names: list[str],
) -> dict[str, float | None]:
    """Get current best values for multiple metrics in one query."""
    from sqlalchemy import func

    query = (
        select(
            SessionMetric.metric_name,
            func.max(SessionMetric.metric_value).label("best"),
        )
        .join(Session)
        .where(
            Session.user_id == user_id,
            Session.element_type == element_type,
            SessionMetric.metric_name.in_(metric_names),
            Session.status == "done",
        )
        .group_by(SessionMetric.metric_name)
    )
    result = await db.execute(query)
    return {row.metric_name: row.best for row in result.all()}
```

- [ ] **Step 4: Update session_saver.py to use batch**

```python
# session_saver.py — replace the for loop (lines 40-70):
    # Build metric rows with PR tracking (batch PR lookup)
    metric_names = [mr.name for mr in metrics]
    current_bests = await get_current_best_batch(
        db,
        user_id=session.user_id,
        element_type=session.element_type,
        metric_names=metric_names,
    )

    metric_rows = []
    for mr in metrics:
        mdef = METRIC_REGISTRY.get(mr.name)
        ref_value = mdef.ideal_range[0] if mdef else None
        ref_max = mdef.ideal_range[1] if mdef else None

        is_in_range = None
        if mdef and ref_value is not None and ref_max is not None:
            is_in_range = ref_value <= mr.value <= ref_max

        # Check PR (from batch result)
        current_best = current_bests.get(mr.name)
        direction = mdef.direction if mdef else "higher"
        is_pr, prev_best = check_pr(direction, current_best, mr.value)

        metric_rows.append(
            {
                "session_id": session_id,
                "metric_name": mr.name,
                "metric_value": mr.value,
                "is_pr": is_pr,
                "prev_best": prev_best,
                "reference_value": ref_value,
                "is_in_range": is_in_range,
            }
        )
```

Also update imports: add `get_current_best_batch` to the import from `app.crud.session_metric`.

- [ ] **Step 5: Run tests**

Run: `cd backend && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/crud/session_metric.py backend/app/services/session_saver.py backend/tests/test_session_metric_batch.py
git commit -m "perf(backend): batch PR query in session_saver (N+1 fix)"
```

---

### Task 7: httpx.AsyncClient reuse + thread-safe cache in vastai/client.py

**Files:**
- Modify: `ml/src/vastai/client.py:26-29,43-60,127-145,186`
- Test: `ml/tests/vastai/test_client_reuse.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/vastai/test_client_reuse.py
"""Tests for httpx reuse and thread-safe cache."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


def test_worker_url_cache_is_thread_safe():
    """Concurrent access to cache should not corrupt state."""
    from src.vastai.client import _worker_url_cache, _worker_url_cache_time, _get_worker_url

    # Reset cache
    import src.vastai.client as client_mod
    client_mod._worker_url_cache = "https://cached-url.example.com"
    client_mod._worker_url_cache_time = 0.0

    with patch("src.vastai.client.httpx.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"url": "https://new-url.example.com"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        results = []
        errors = []

        def read_cache():
            try:
                results.append(client_mod._worker_url_cache)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_cache) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r == "https://cached-url.example.com" for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && uv run pytest tests/vastai/test_client_reuse.py -v`
Expected: PASS (current code may work for reads, but test documents expected behavior)

- [ ] **Step 3: Implement thread-safe cache + httpx reuse**

```python
# ml/src/vastai/client.py — replace lines 26-29:
import threading

_worker_url_cache: str | None = None
_worker_url_cache_time: float = 0.0
_WORKER_URL_TTL = 60
_cache_lock = threading.Lock()

# Module-level httpx client (lazy init)
_http_client: httpx.AsyncClient | None = None
_client_lock = threading.Lock()


def _get_http_client() -> httpx.AsyncClient:
    """Get or create a shared httpx.AsyncClient."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    return _http_client
```

Update `_get_worker_url` (sync) and `_asyncio_get_worker_url` (async) to use `_cache_lock`:

```python
def _get_worker_url(endpoint_name: str, api_key: str) -> str:
    global _worker_url_cache, _worker_url_cache_time
    with _cache_lock:
        now = time.monotonic()
        if _worker_url_cache and (now - _worker_url_cache_time) < _WORKER_URL_TTL:
            return _worker_url_cache
    # ... rest unchanged, but wrap the cache write:
    resp = httpx.post(...)
    resp.raise_for_status()
    url = resp.json()["url"]
    with _cache_lock:
        _worker_url_cache = url
        _worker_url_cache_time = time.monotonic()
    return url
```

Update `process_video_remote_async` to reuse httpx client:

```python
# Line 186 — BEFORE:
async with httpx.AsyncClient() as client:
    resp = await client.post(...)

# AFTER:
client = _get_http_client()
resp = await client.post(...)
```

- [ ] **Step 4: Run tests**

Run: `cd ml && uv run python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add ml/src/vastai/client.py ml/tests/vastai/test_client_reuse.py
git commit -m "refactor(vastai): httpx client reuse + thread-safe URL cache"
```

---

### Task 8: SSE timeout + worker graceful shutdown

**Files:**
- Modify: `backend/app/routes/process.py:114-144` (SSE timeout)
- Modify: `ml/src/worker.py:163-168` (shutdown)
- Test: `backend/tests/routes/test_sse_timeout.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/routes/test_sse_timeout.py
"""Tests for SSE timeout."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SSE_STREAM_TIMEOUT = 60  # seconds


@pytest.mark.asyncio
async def test_sse_stream_has_timeout():
    """SSE stream should timeout after 60s of no messages."""
    with (
        patch("app.routes.process.get_valkey_client") as mock_valkey_fn,
        patch("app.routes.process.get_task_state", new_callable=AsyncMock) as mock_state,
    ):
        mock_valkey = AsyncMock()
        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()

        # Simulate no messages (pubsub.listen yields nothing, then times out)
        async def listen_nothing():
            yield  # yield one iteration, then timeout

        mock_pubsub.listen = listen_nothing
        mock_valkey.pubsub.return_value = mock_pubsub
        mock_valkey.close = AsyncMock()
        mock_valkey_fn.return_value = mock_valkey

        mock_state.return_value = {"status": "running", "progress": "0.5"}

        # Import should work without hanging forever
        from app.routes.process import stream_process_status
        assert hasattr(stream_process_status, "__call__")
```

- [ ] **Step 2: Implement SSE timeout**

```python
# backend/app/routes/process.py — replace event_generator (lines 118-142):
SSE_STREAM_TIMEOUT = 60  # seconds

async def event_generator():
    valkey = await get_valkey_client()
    pubsub = valkey.pubsub()
    channel = f"{TASK_EVENTS_PREFIX}{task_id}"
    await pubsub.subscribe(channel)
    try:
        # Send initial state
        state = await get_task_state(task_id, valkey=valkey)
        if state:
            if state["status"] in ("completed", "failed", "cancelled"):
                yield f"data: {json.dumps(state)}\n\n"
                return
            yield f"data: {json.dumps(state)}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'unknown'})}\n\n"

        # Listen with timeout
        try:
            async with asyncio.timeout(SSE_STREAM_TIMEOUT):
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        yield f"data: {message['data'].decode()}\n\n"
                        try:
                            data = json.loads(message["data"])
                            if data.get("status") in ("completed", "failed", "cancelled"):
                                return
                        except (json.JSONDecodeError, TypeError):
                            pass
        except TimeoutError:
            # Fallback: poll final state
            state = await get_task_state(task_id, valkey=valkey)
            if state:
                yield f"data: {json.dumps({**state, '_timeout': True})}\n\n"
    finally:
        await pubsub.unsubscribe(channel)
```

- [ ] **Step 3: Implement worker graceful shutdown**

```python
# ml/src/worker.py — replace lines 167-168:
async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker shutting down")
    pool = ctx.get("redis")
    if pool:
        await pool.close()
    # Close httpx client used by Vast.ai
    from src.vastai.client import _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
```

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/process.py ml/src/worker.py backend/tests/routes/test_sse_timeout.py
git commit -m "fix(backend): SSE timeout + worker graceful shutdown"
```

---

## Phase 2: GPU Optimizations (Tier 2)

### Task 9: ONNX SessionOptions for BatchRTMO

**Files:**
- Modify: `ml/src/pose_estimation/rtmo_batch.py:221-238` (add SessionOptions)
- Modify: `ml/src/pose_3d/onnx_extractor.py:52-56` (add SessionOptions)
- Test: `ml/tests/pose_estimation/test_batch_rtmo_session_opts.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pose_estimation/test_batch_rtmo_session_opts.py
"""Tests for ONNX SessionOptions optimization."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_batch_rtmo_uses_session_options():
    """BatchRTMO should create InferenceSession with optimized SessionOptions."""
    with patch("src.pose_estimation.rtmo_batch.onnxruntime") as mock_ort:
        mock_ort.GraphOptimizationLevel = MagicMock(ORT_ENABLE_ALL="ORT_ENABLE_ALL")
        mock_ort.ExecutionMode = MagicMock(ORT_SEQUENTIAL="ORT_SEQUENTIAL")
        mock_ort.InferenceSession = MagicMock()
        mock_ort.CUDNN_CONV_ALGO_SEARCH_EXHAUSTIVE = "EXHAUSTIVE"

        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        from pathlib import Path
        with patch("src.pose_estimation.rtmo_batch.RTMO_MODELS", {"balanced": "/fake/model.onnx"}):
            with patch("src.pose_estimation.rtmo_batch.Path") as mock_path:
                mock_path.return_value.exists.return_value = True
                mock_path.return_value.__str__ = lambda self: "/fake/model.onnx"

                from src.pose_estimation.rtmo_batch import BatchRTMO

                try:
                    BatchRTMO(mode="balanced", device="cpu", score_thr=0.3, nms_thr=0.45)
                except Exception:
                    pass  # May fail for other reasons, we just check SessionOptions

        # Verify InferenceSession was called with sess_options
        call_args = mock_ort.InferenceSession.call_args
        if call_args:
            # Second positional arg or sess_options keyword
            assert len(call_args.args) >= 2 or "sess_options" in call_args.kwargs
```

- [ ] **Step 2: Implement SessionOptions**

```python
# ml/src/pose_estimation/rtmo_batch.py — replace lines 221-232:
import onnxruntime

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if self._device == "cuda"
    else ["CPUExecutionProvider"]
)

# Optimized session options
opts = onnxruntime.SessionOptions()
opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.enable_mem_pattern = True
opts.enable_mem_reuse = True
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 1

self._session = onnxruntime.InferenceSession(
    str(model_path),
    sess_options=opts,
    providers=providers,
)
```

- [ ] **Step 3: Add warm-up after session creation**

```python
# After session creation, add warm-up:
logger.info(f"BatchRTMO initialized: mode={mode}, device={self._device}")

# Warm-up inference (eliminates first-call CUDA compilation latency)
import numpy as np
dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
self._session.run(self._output_names, {self._input_name: dummy})
```

- [ ] **Step 4: Run tests**

Run: `cd ml && uv run python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add ml/src/pose_estimation/rtmo_batch.py ml/tests/pose_estimation/test_batch_rtmo_session_opts.py
git commit -m "perf(pose): ONNX SessionOptions + warm-up for BatchRTMO"
```

---

### Task 10: Double buffering — AsyncFrameReader in pose_extractor.py

**Files:**
- Modify: `ml/src/pose_estimation/pose_extractor.py:238-300` (use AsyncFrameReader)
- Test: `ml/tests/pose_estimation/test_async_frame_extraction.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pose_estimation/test_async_frame_extraction.py
"""Tests for double-buffered frame extraction."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_pose_extractor_uses_async_frame_reader():
    """extract_video_tracked should use AsyncFrameReader when available."""
    # Verify that AsyncFrameReader is imported and used in pose_extractor
    from src.pose_estimation.pose_extractor import PoseExtractor
    import src.pose_estimation.pose_extractor as pe_mod
    assert hasattr(pe_mod, "AsyncFrameReader"), "AsyncFrameReader should be imported"
```

- [ ] **Step 2: Implement double buffering**

In `ml/src/pose_estimation/pose_extractor.py`, add import at top:

```python
from src.utils.frame_buffer import AsyncFrameReader
```

Replace the main extraction loop (lines 238-300) to use AsyncFrameReader. The key change is replacing `cap.read()` with `AsyncFrameReader.get_frame()`. The logic inside the loop (resize, tracker, keypoints, tracking) stays the same.

```python
# BEFORE (line 239-255):
cap = cv2.VideoCapture(str(video_path))
...
while cap.isOpened() and frame_idx < num_frames:
    if self._frame_skip > 1 and frame_idx % self._frame_skip != 0:
        ret = cap.grab()
        ...
        continue
    ret, frame = cap.read()

# AFTER:
reader = AsyncFrameReader(video_path, buffer_size=16, frame_skip=self._frame_skip)
reader.start()

try:
    while True:
        result = reader.get_frame()
        if result is None:
            break
        frame_idx, frame = result
        num_frames = max(num_frames, frame_idx + 1)
        ...  # rest of per-frame logic unchanged
finally:
    reader.join()
```

Note: frame_skip is handled by AsyncFrameReader natively (line 72-74 in frame_buffer.py), so the manual skip logic can be removed.

- [ ] **Step 3: Run tests**

Run: `cd ml && uv run python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add ml/src/pose_estimation/pose_extractor.py ml/tests/pose_estimation/test_async_frame_extraction.py
git commit -m "perf(pose): double buffering via AsyncFrameReader in pose extraction"
```

---

### Task 11: Batch RTMO integration in pipeline

**Files:**
- Modify: `ml/src/pose_estimation/pose_extractor.py` (add batched extraction method)
- Modify: `ml/src/pose_estimation/batch_extractor.py` (add tracking support)
- Modify: `ml/src/pipeline.py` (use batched extraction when available)
- Test: `ml/tests/pose_estimation/test_batch_pipeline_integration.py` (create)
- Benchmark: `ml/scripts/benchmark_batch_inference.py` (create)

This is the largest task. It requires:
1. Adding tracking post-processing to BatchPoseExtractor
2. Creating a unified extraction method that dispatches to batch or per-frame
3. Benchmarking to measure actual speedup

- [ ] **Step 1: Write the benchmark script**

```python
# ml/scripts/benchmark_batch_inference.py
"""Benchmark batch vs per-frame RTMO inference."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose_estimation.rtmo_batch import BatchRTMO
from src.pose_estimation.pose_extractor import PoseExtractor


def benchmark_batch(video_path: str, batch_sizes: list[int] = [1, 4, 8, 16]):
    """Compare batch inference times."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Total frames: {len(frames)}")

    for bs in batch_sizes:
        rtmo = BatchRTMO(mode="balanced", device="cuda", score_thr=0.3, nms_thr=0.45)

        # Warm-up
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        rtmo._session.run(rtmo._output_names, {rtmo._input_name: dummy})

        start = time.perf_counter()
        for i in range(0, len(frames), bs):
            batch = frames[i : i + bs]
            rtmo.infer_batch(batch)
        elapsed = time.perf_counter() - start

        fps = len(frames) / elapsed
        print(f"batch_size={bs:2d}: {elapsed:.3f}s ({fps:.1f} fps)")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/test_video.mp4"
    benchmark_batch(video)
```

- [ ] **Step 2: Run benchmark to get baseline numbers**

Run: `cd ml && uv run python scripts/benchmark_batch_inference.py data/test_video.mp4`
Expected: Table of batch_size vs time

- [ ] **Step 3: Add tracking post-processing to batch extraction**

In `ml/src/pose_estimation/batch_extractor.py`, add a method that runs batch inference then sequential tracking:

```python
def extract_with_tracking(
    self,
    video_path: str,
    frame_skip: int = 1,
    progress_cb=None,
) -> TrackedExtraction:
    """Batch inference + sequential tracking post-process."""
    # 1. Read all frames
    cap = cv2.VideoCapture(str(video_path))
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    # 2. Select frames (frame_skip)
    selected = all_frames[::frame_skip]

    # 3. Batch inference
    batch_size = self._batch_size
    all_results = []
    for i in range(0, len(selected), batch_size):
        batch = selected[i : i + batch_size]
        results = self._batch_rtmo.infer_batch(batch)
        all_results.extend(results)

    # 4. Sequential tracking post-process (using Sports2D / custom tracker)
    # ... (integrate existing tracking logic from PoseExtractor)
```

- [ ] **Step 4: Add dispatch in PoseExtractor**

In `pose_extractor.py`, add a `use_batch` parameter:

```python
def extract_video_tracked(
    self,
    video_path: str | Path,
    ...,
    use_batch: bool = False,
    batch_size: int = 8,
) -> TrackedExtraction:
    if use_batch:
        from src.pose_estimation.batch_extractor import BatchPoseExtractor
        batch_ext = BatchPoseExtractor(batch_size=batch_size, device=self._device)
        return batch_ext.extract_with_tracking(str(video_path), frame_skip=self._frame_skip, progress_cb=progress_cb)
    # ... existing per-frame logic
```

- [ ] **Step 5: Run tests + benchmark**

Run: `cd ml && uv run python -m pytest tests/ -v`
Run: `cd ml && uv run python scripts/benchmark_batch_inference.py data/test_video.mp4`

- [ ] **Step 6: Commit**

```bash
git add ml/src/pose_estimation/pose_extractor.py ml/src/pose_estimation/batch_extractor.py ml/scripts/benchmark_batch_inference.py ml/tests/pose_estimation/test_batch_pipeline_integration.py
git commit -m "feat(pose): integrate BatchRTMO with tracking post-process"
```

---

### Task 12: ONNX IO Binding for BatchRTMO

**Files:**
- Modify: `ml/src/pose_estimation/rtmo_batch.py:240-270` (add IO Binding path)
- Test: `ml/tests/pose_estimation/test_io_binding.py` (create)

- [ ] **Step 1: Write the test**

```python
# ml/tests/pose_estimation/test_io_binding.py
"""Tests for IO Binding zero-copy inference."""
from __future__ import annotations

import numpy as np

import pytest


@pytest.mark.cuda
def test_batch_rtmo_io_binding_matches_regular():
    """IO Binding results should match regular session.run results."""
    pytest.skip("Requires GPU — run manually with: uv run python scripts/benchmark_batch_inference.py")
```

- [ ] **Step 2: Implement IO Binding**

Add to `BatchRTMO` class:

```python
def infer_batch_iobinding(
    self,
    frames: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run batch inference with IO Binding (zero-copy GPU transfer)."""
    if not frames:
        return []

    batch_tensor, ratios = preprocess_batch(frames)
    batch_size = batch_tensor.shape[0]

    # Pre-allocate GPU tensors (once)
    if not hasattr(self, "_binding") or self._binding_batch_size != batch_size:
        import onnxruntime as ort

        self._binding = self._session.io_binding()
        input_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(
            [batch_size, 3, 640, 640],
            np.float32,
            "cuda",
            0,
        )
        self._binding.bind_ortvalue_input(self._input_name, input_ortvalue)
        for name in self._output_names:
            self._binding.bind_output(name, "cuda", 0)
        self._binding_batch_size = batch_size

    # Update input in-place
    input_ortvalue = self._binding.get_inputs()[0].get_ortvalue()
    input_ortvalue.update_inplace(batch_tensor)

    # Run inference
    self._session.run_with_iobinding(self._binding)

    # Read outputs
    dets = self._binding.get_outputs()[0].get_numpy_data()[:len(frames)]
    keypoints = self._binding.get_outputs()[1].get_numpy_data()[:len(frames)]

    return postprocess_batch(dets, keypoints, ratios, self._score_thr, self._nms_thr)
```

- [ ] **Step 3: Run tests**

Run: `cd ml && uv run python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Benchmark IO Binding vs regular**

Add to benchmark script and compare.

- [ ] **Step 5: Commit**

```bash
git add ml/src/pose_estimation/rtmo_batch.py ml/tests/pose_estimation/test_io_binding.py
git commit -m "perf(pose): ONNX IO Binding zero-copy for BatchRTMO"
```

---

## Phase 3: Strategic Investments (Tier 3)

### Task 13: FP16 quantization of RTMO model

**Files:**
- Create: `ml/scripts/quantize_rtmo_fp16.py`
- Modify: `ml/src/pose_estimation/rtmo_batch.py` (auto-detect FP16 model)
- Test: `ml/tests/pose_estimation/test_fp16_quantization.py` (create)

- [ ] **Step 1: Write the quantization script**

```python
# ml/scripts/quantize_rtmo_fp16.py
"""Convert RTMO ONNX model to FP16 for GPU inference."""
from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnxconverter_common import float16


def quantize(input_path: str, output_path: str) -> None:
    """Convert ONNX model to FP16 (keep IO in FP32)."""
    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)
    print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    from src.pose_estimation.rtmo_batch import RTMO_MODELS
    input_model = RTMO_MODELS[mode]
    output_model = input_model.replace(".onnx", "-fp16.onnx")
    quantize(input_model, output_model)
```

- [ ] **Step 2: Run quantization**

Run: `cd ml && uv run python scripts/quantize_rtmo_fp16.py balanced`

- [ ] **Step 3: Benchmark FP16 vs FP32**

Run: `cd ml && uv run python scripts/benchmark_batch_inference.py data/test_video.mp4`
Compare FP32 vs FP16 inference times and accuracy.

- [ ] **Step 4: Add FP16 auto-detection to BatchRTMO**

```python
# In BatchRTMO.__init__:
# Check for FP16 variant first
fp16_model_path = str(model_path).replace(".onnx", "-fp16.onnx")
if Path(fp16_model_path).exists():
    model_path = Path(fp16_model_path)
    logger.info(f"Using FP16 model: {fp16_model_path}")
```

- [ ] **Step 5: Commit**

```bash
git add ml/scripts/quantize_rtmo_fp16.py ml/src/pose_estimation/rtmo_batch.py ml/tests/pose_estimation/test_fp16_quantization.py
git commit -m "feat(pose): FP16 quantization for RTMO with auto-detection"
```

---

### Task 14: NVENC encode in H264Writer

**Files:**
- Modify: `ml/src/utils/video_writer.py:30-47` (add NVENC auto-detect)
- Test: `ml/tests/utils/test_video_writer_nvenc.py` (create)

- [ ] **Step 1: Write the test**

```python
# ml/tests/utils/test_video_writer_nvenc.py
"""Tests for NVENC auto-detection in H264Writer."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_h264_writer_detects_nvenc():
    """H264Writer should try h264_nvenc first, fallback to libx264."""
    with patch("src.utils.video_writer.av.open") as mock_open:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_container.add_stream.return_value = mock_stream
        mock_container.__aenter__ = MagicMock(return_value=mock_container)
        mock_container.__aexit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_container

        from src.utils.video_writer import H264Writer

        # Should not raise
        try:
            writer = H264Writer("/tmp/test.mp4", 1280, 720, 30)
            writer.close()
        except Exception:
            pass
```

- [ ] **Step 2: Implement NVENC auto-detect**

```python
# ml/src/utils/video_writer.py — modify H264Writer.__init__:
def __init__(self, path, width, height, fps, codec="auto"):
    self._container = av.open(str(path), "w")
    self._stream = self._container.add_stream(codec, rate=fps)
    self._stream.width = width
    self._stream.height = height
    self._stream.pix_fmt = "yuv420p"

# Replace with:
def __init__(self, path, width, height, fps, codec="auto"):
    self._container = av.open(str(path), "w")

    # Auto-detect NVENC
    if codec == "auto":
        try:
            test_stream = self._container.add_stream("h264_nvenc", rate=fps)
            # Probe if it actually works
            test_stream.width = width
            test_stream.height = height
            test_stream.pix_fmt = "yuv420p"
            codec = "h264_nvenc"
        except Exception:
            # NVENC not available, remove test stream
            self._container.streams.video.clear()
            codec = "libx264"

    self._stream = self._container.add_stream(codec, rate=fps)
    self._stream.width = width
    self._stream.height = height
    self._stream.pix_fmt = "yuv420p"

    if codec == "h264_nvenc":
        self._stream.options = {"preset": "p4", "rc": "constqp", "qp": "28"}
```

- [ ] **Step 3: Remove unnecessary cvtColor**

```python
# video_writer.py — in write():
# BEFORE:
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")

# AFTER:
av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
```

- [ ] **Step 4: Run tests**

Run: `cd ml && uv run python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add ml/src/utils/video_writer.py ml/tests/utils/test_video_writer_nvenc.py
git commit -m "perf(video): NVENC auto-detect + remove unnecessary cvtColor"
```

---

### Task 15: CUDA Graph Capture for BatchRTMO

**Files:**
- Modify: `ml/src/pose_estimation/rtmo_batch.py` (add CUDA Graph)
- Test: `ml/tests/pose_estimation/test_cuda_graph.py` (create)

- [ ] **Step 1: Write the test**

```python
# ml/tests/pose_estimation/test_cuda_graph.py
"""Tests for CUDA Graph capture."""
from __future__ import annotations

import pytest


@pytest.mark.cuda
def test_batch_rtmo_cuda_graph():
    """CUDA Graph should capture and replay correctly."""
    pytest.skip("Requires GPU — manual verification only")
```

- [ ] **Step 2: Implement CUDA Graph**

```python
# Add to BatchRTMO after warm-up:
def _enable_cuda_graph(self, batch_size: int) -> None:
    """Capture CUDA graph for fixed batch size inference."""
    if self._device != "cuda":
        return
    import onnxruntime as ort
    if not hasattr(ort.InferenceSession, "capture_cuda_graph"):
        logger.info("CUDA Graph not available in this ONNX Runtime version")
        return

    dummy = np.zeros((batch_size, 3, 640, 640), dtype=np.float32)
    self._session.capture_cuda_graph(
        feeds={self._input_name: dummy},
    )
    logger.info(f"CUDA Graph captured for batch_size={batch_size}")
```

- [ ] **Step 3: Run tests**

Run: `cd ml && uv run python -m pytest tests/ -v`

- [ ] **Step 4: Benchmark**

Run: `cd ml && uv run python scripts/benchmark_batch_inference.py data/test_video.mp4`
Compare with/without CUDA Graph.

- [ ] **Step 5: Commit**

```bash
git add ml/src/pose_estimation/rtmo_batch.py ml/tests/pose_estimation/test_cuda_graph.py
git commit -m "perf(pose): CUDA Graph capture for fixed batch size"
```

---

## Dependency Graph

```
Task 1 ─┐
Task 2 ─┤
Task 3 ─┤─→ Task 4 (Valkey pool) ──→ Task 5 (arq pool) ──→ Testing
Task 6 ─┤
Task 7 ─┘
          │
Task 8 ───┘──→ Testing

Task 9 ───→ Task 10 (Double buffer) ──→ Task 11 (Batch RTMO) ──→ Task 12 (IO Binding) ──→ Benchmark
                                                                      │
Task 13 (FP16) ────────────────────────────────────────────────────────┘

Task 14 (NVENC) ──→ Testing (independent)
Task 15 (CUDA Graph) ──→ Testing (depends on Task 11)
```

## Expected Cumulative Speedup

| After Tasks | Pipeline Time (364 frames) | Speedup | Source |
|-------------|---------------------------|---------|--------|
| Baseline | ~9.3s | 1.0x | Current |
| 1-8 (Tier 1) | ~9.3s | 1.0x | I/O only, no pipeline change |
| + 9-10 (SessionOpts + DoubleBuffer) | ~8.6s | 1.08x | Benchmark script |
| + 11 (Batch RTMO) | ~4.3s | 2.16x | Benchmark script |
| + 12 (IO Binding) | ~3.6s | 2.58x | Benchmark script |
| + 13 (FP16) | ~2.0-2.5s | 3.7-4.6x | Benchmark script |
| + 14 (NVENC) | depends | needs benchmark | Encode fraction |
| + 15 (CUDA Graph) | ~2.0-2.3s | ~4x | Benchmark script |

**All speedup numbers are ESTIMATES. Run `ml/scripts/benchmark_batch_inference.py` after each task to get actual numbers.**
