# S3-Only Storage Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate `outputs_dir` (local `data/uploads/`). All files stored in S3-compatible storage (R2 now, MinIO later). Backend streams files from R2 to frontend. Remove SSE endpoint, keep queue-only processing.

**Architecture:** Backend receives uploads, stores in R2 with `input/` prefix. Processing dispatches to Vast.ai which reads/writes R2 directly. Results served via backend streaming proxy (`StreamingResponse` from boto3). No local file storage for outputs.

**Tech Stack:** FastAPI, boto3 (S3-compatible), Cloudflare R2, arq/Valkey, Next.js

---

### Task 1: Expand storage.py with new helpers

**Files:**
- Modify: `src/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing tests for new storage functions**

Add these tests to `tests/test_storage.py`:

```python
from src.storage import (
    delete_object,
    download_file,
    get_object_url,
    list_objects,
    object_exists,
    stream_object,
    upload_bytes,
    upload_file,
)


@patch("src.storage._client")
def test_upload_bytes_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_client.return_value = mock_s3
    with patch("src.storage.get_settings"):
        result = upload_bytes(b"hello world", "input/test.txt")
    assert result == "input/test.txt"
    mock_s3.put_object.assert_called_once_with(
        Bucket=mock.ANY, Key="input/test.txt", Body=b"hello world"
    )


@patch("src.storage._client")
def test_stream_object_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.iter_chunks.return_value = [b"chunk1", b"chunk2"]
    mock_body.content_length = 12
    mock_body.content_type = "video/mp4"
    mock_s3.get_object.return_value = {"Body": mock_body, "ContentLength": 12, "ContentType": "video/mp4"}
    mock_client.return_value = mock_s3
    with patch("src.storage.get_settings"):
        body, length, ctype = stream_object("output/test.mp4")
    mock_s3.get_object.assert_called_once_with(Bucket=mock.ANY, Key="output/test.mp4")
    assert length == 12
    assert ctype == "video/mp4"
    assert list(body.iter_chunks()) == [b"chunk1", b"chunk2"]


@patch("src.storage._client")
def test_object_exists_true(mock_client):
    mock_s3 = MagicMock()
    mock_s3.head_object.return_value = {"ContentLength": 100}
    mock_client.return_value = mock_s3
    with patch("src.storage.get_settings"):
        assert object_exists("output/test.mp4") is True
    mock_s3.head_object.assert_called_once()


@patch("src.storage._client")
def test_object_exists_false(mock_client):
    from botocore.exceptions import ClientError

    mock_s3 = MagicMock()
    mock_s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "404"}}, "head_object"
    )
    mock_client.return_value = mock_s3
    with patch("src.storage.get_settings"):
        assert object_exists("output/missing.mp4") is False


@patch("src.storage._client")
def test_get_object_url_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://r2.example.com/output/test.mp4?sig=abc"
    mock_client.return_value = mock_s3
    with patch("src.storage.get_settings"):
        url = get_object_url("output/test.mp4", expires=1800)
    assert "sig=abc" in url
    mock_s3.generate_presigned_url.assert_called_once()


@patch("src.storage._client")
def test_list_objects(mock_client):
    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {
        "Contents": [{"Key": "input/a.mp4"}, {"Key": "input/b.mp4"}]
    }
    mock_client.return_value = mock_s3
    with patch("src.storage.get_settings"):
        keys = list_objects("input/")
    assert keys == ["input/a.mp4", "input/b.mp4"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL — `ImportError: cannot import name 'upload_bytes'`

- [ ] **Step 3: Implement new storage functions**

Add to `src/storage.py` after the existing `delete_object` function:

```python
def upload_bytes(data: bytes, key: str) -> str:
    """Upload bytes to R2. Returns the key."""
    bucket = get_settings().r2.bucket
    logger.info("Uploading %d bytes -> s3://%s/%s", len(data), bucket, key)
    _client().put_object(Bucket=bucket, Key=key, Body=data)
    return key


def stream_object(key: str) -> tuple:
    """Stream object from R2. Returns (body, content_length, content_type)."""
    bucket = get_settings().r2.bucket
    logger.info("Streaming s3://%s/%s", bucket, key)
    resp = _client().get_object(Bucket=bucket, Key=key)
    body = resp["Body"]
    length = resp.get("ContentLength", 0)
    ctype = resp.get("ContentType", "application/octet-stream")
    return body, length, ctype


def object_exists(key: str) -> bool:
    """Check if object exists in R2."""
    from botocore.exceptions import ClientError

    try:
        _client().head_object(Bucket=get_settings().r2.bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def get_object_url(key: str, expires: int = 3600) -> str:
    """Generate a presigned URL for an object."""
    bucket = get_settings().r2.bucket
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def list_objects(prefix: str) -> list[str]:
    """List object keys with given prefix."""
    bucket = get_settings().r2.bucket
    resp = _client().list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_storage.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage.py tests/test_storage.py
git commit -m "feat(storage): add upload_bytes, stream_object, object_exists, get_object_url, list_objects"
```

---

### Task 2: Update config — remove outputs_dir, add presign_expires

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`
- Modify: `.envrc`

- [ ] **Step 1: Remove outputs_dir from AppConfig**

In `src/config.py`, remove line 115 (`outputs_dir: str = "data/uploads"`) from `AppConfig`.

- [ ] **Step 2: Add presign_expires to R2Config**

In `src/config.py`, add after line 93 (`bucket: str = "skating-ml-pipeline"`):

```python
    presign_expires: int = 3600
```

- [ ] **Step 3: Remove from .env.example**

In `.env.example`, remove line 29 (`APP_OUTPUTS_DIR=data/uploads`).

- [ ] **Step 4: Remove from .envrc**

In `.envrc`, remove line 18 (`export OUTPUTS_DIR=${OUTPUTS_DIR:-data/uploads}`).

- [ ] **Step 5: Commit**

```bash
git add src/config.py .env.example .envrc
git commit -m "refactor(config): remove outputs_dir, add r2.presign_expires"
```

---

### Task 3: Update schemas — video_path → video_key

**Files:**
- Modify: `src/backend/schemas.py`

- [ ] **Step 1: Update ProcessRequest**

In `src/backend/schemas.py`, change `ProcessRequest` (line 100):

```python
class ProcessRequest(BaseModel):
    video_key: str
    person_click: PersonClick
    frame_skip: int = 1
    layer: int = 3
    tracking: str = "auto"
    export: bool = True
    depth: bool = False
    optical_flow: bool = False
    segment: bool = False
    foot_track: bool = False
    matting: bool = False
    inpainting: bool = False
```

- [ ] **Step 2: Update DetectResponse**

In `src/backend/schemas.py`, change `DetectResponse` (line 94):

```python
class DetectResponse(BaseModel):
    persons: list[PersonInfo]
    preview_image: str
    video_key: str
    auto_click: PersonClick | None = None
    status: str
```

- [ ] **Step 3: Commit**

```bash
git add src/backend/schemas.py
git commit -m "refactor(schemas): rename video_path to video_key in ProcessRequest and DetectResponse"
```

---

### Task 4: Simplify Vast.ai client — R2 key in, R2 keys out

**Files:**
- Modify: `src/vastai/client.py`
- Test: `tests/test_vastai_client.py`

- [ ] **Step 1: Write failing test for new VastResult fields**

Replace the existing `test_vast_result_fields` in `tests/test_vastai_client.py`:

```python
def test_vast_result_fields():
    r = VastResult(
        video_key="output/test_analyzed.mp4",
        poses_key="output/test_poses.npy",
        csv_key=None,
        stats={"frames": 100},
    )
    assert r.video_key == "output/test_analyzed.mp4"
    assert r.csv_key is None
    assert r.stats == {"frames": 100}


@patch("src.vastai.client.httpx.post")
def test_process_video_remote_passes_r2_key(mock_post):
    mock_route_resp = MagicMock()
    mock_route_resp.status_code = 200
    mock_route_resp.json.return_value = {"url": "https://worker.vast.ai:8000"}
    mock_route_resp.raise_for_status = MagicMock()

    mock_process_resp = MagicMock()
    mock_process_resp.status_code = 200
    mock_process_resp.json.return_value = {
        "video_r2_key": "output/test_analyzed.mp4",
        "poses_r2_key": "output/test_poses.npy",
        "csv_r2_key": None,
        "stats": {"total_frames": 100, "valid_frames": 90, "fps": 30.0, "resolution": "1920x1080"},
    }
    mock_process_resp.raise_for_status = MagicMock()

    mock_post.side_effect = [mock_route_resp, mock_process_resp]

    with patch("src.vastai.client.get_settings") as mock_settings:
        s = MagicMock()
        s.vastai.api_key.get_secret_value.return_value = "test-key"
        s.vastai.endpoint_name = "skating-ml-gpu"
        s.r2.endpoint_url = "https://r2.example.com"
        s.r2.access_key_id.get_secret_value.return_value = "key-id"
        s.r2.secret_access_key.get_secret_value.return_value = "secret"
        s.r2.bucket = "test-bucket"
        mock_settings.return_value = s

        result = process_video_remote(
            video_key="input/test.mp4",
            person_click={"x": 100, "y": 200},
        )

    assert result.video_key == "output/test_analyzed.mp4"
    assert result.poses_key == "output/test_poses.npy"
    assert result.csv_key is None
    # Two calls: route + process
    assert mock_post.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vastai_client.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'video_key'`

- [ ] **Step 3: Rewrite Vast.ai client**

Replace the entire `src/vastai/client.py` with:

```python
"""Client for calling Vast.ai Serverless GPU endpoint.

Flow:
  1. POST /route to get worker URL from Vast.ai
  2. POST /process to the worker with R2 key + credentials
  3. Worker processes and uploads results to R2
  4. Return R2 keys (no local download)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

ROUTE_URL = "https://run.vast.ai/route/"
REQUEST_TIMEOUT = 600  # 10 min for video processing
ROUTE_TIMEOUT = 30


@dataclass
class VastResult:
    video_key: str
    poses_key: str | None
    csv_key: str | None
    stats: dict


def _get_worker_url(endpoint_name: str, api_key: str) -> str:
    """Route request to get a ready worker URL."""
    resp = httpx.post(
        ROUTE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"endpoint": endpoint_name},
        timeout=ROUTE_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["url"]


def process_video_remote(
    video_key: str,
    person_click: dict[str, int] | None = None,
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    ml_flags: dict[str, bool] | None = None,
) -> VastResult:
    """Send video processing to Vast.ai Serverless GPU.

    Video must already be in R2 at `video_key`.
    Returns R2 keys for results (no local download).

    Raises httpx.HTTPStatusError on routing/processing failures.
    """
    settings = get_settings()
    if ml_flags is None:
        ml_flags = {}

    api_key = settings.vastai.api_key.get_secret_value()
    endpoint_name = settings.vastai.endpoint_name

    # 1. Route to worker
    logger.info("Routing to Vast.ai endpoint: %s", endpoint_name)
    worker_url = _get_worker_url(endpoint_name, api_key)
    logger.info("Worker URL: %s", worker_url)

    # 2. Send processing request (video is already in R2)
    payload = {
        "video_r2_key": video_key,
        "person_click": person_click,
        "frame_skip": frame_skip,
        "layer": layer,
        "tracking": tracking,
        "export": export,
        "ml_flags": ml_flags,
        "r2_endpoint_url": settings.r2.endpoint_url,
        "r2_access_key_id": settings.r2.access_key_id.get_secret_value(),
        "r2_secret_access_key": settings.r2.secret_access_key.get_secret_value(),
        "r2_bucket": settings.r2.bucket,
    }
    resp = httpx.post(
        f"{worker_url}/process",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json()

    # 3. Return R2 keys directly (no download)
    return VastResult(
        video_key=result["video_r2_key"],
        poses_key=result.get("poses_r2_key"),
        csv_key=result.get("csv_r2_key"),
        stats=result["stats"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vastai_client.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/vastai/client.py tests/test_vastai_client.py
git commit -m "refactor(vastai): simplify client — R2 key in, R2 keys out, no upload/download"
```

---

### Task 5: Update detect route — upload to R2, use /tmp, return video_key

**Files:**
- Modify: `src/backend/routes/detect.py`

- [ ] **Step 1: Rewrite detect route**

Replace the entire `src/backend/routes/detect.py` with:

```python
"""POST /api/detect — detect persons in an uploaded video."""

from __future__ import annotations

import base64
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile

from src.backend.schemas import DetectResponse, PersonClick, PersonInfo
from src.device import DeviceConfig
from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
from src.storage import upload_bytes
from src.utils.video import get_video_meta
from src.web_helpers import (
    render_person_preview,
)

router = APIRouter()


def _create_extractor(tracking: str) -> RTMPoseExtractor:
    cfg = DeviceConfig.default()
    return RTMPoseExtractor(
        mode="balanced",
        tracking_backend="rtmlib",
        tracking_mode=tracking,
        conf_threshold=0.3,
        output_format="normalized",
        device=cfg.device,
    )


def _encode_frame_bgr(frame: np.ndarray) -> str:
    """Encode BGR frame to base64 PNG string."""
    success, buf = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError("Failed to encode preview image")
    return base64.b64encode(buf).decode("ascii")


@router.post("/detect", response_model=DetectResponse)
async def detect_persons(
    video: UploadFile,
    tracking: str = "auto",
) -> DetectResponse:
    """Detect all persons in the uploaded video and return annotated preview."""
    suffix = Path(video.filename or "video.mp4").suffix
    video_key = f"input/{uuid.uuid4().hex}{suffix}"

    content = await video.read()

    # Upload to R2
    upload_bytes(content, video_key)

    # Save to /tmp for local RTMPose processing
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            tmp_file = f.name

        video_path = Path(tmp_file)

        extractor = _create_extractor(tracking)
        persons, _ = extractor.preview_persons(video_path, num_frames=30)

        if not persons:
            return DetectResponse(
                persons=[],
                preview_image="",
                video_key=video_key,
                status="Люди не найдены. Попробуйте другое видео.",
            )

        # Read first frame for annotated preview
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(status_code=500, detail="Failed to read video frame")

        meta = get_video_meta(video_path)
        w, h = meta.width, meta.height

        annotated = render_person_preview(frame, persons, selected_idx=None)
        preview_b64 = _encode_frame_bgr(annotated)

        # Auto-select if only one person
        auto_click = None
        status: str
        if len(persons) == 1:
            mid_hip = persons[0]["mid_hip"]
            auto_click = PersonClick(
                x=int(mid_hip[0] * w),
                y=int(mid_hip[1] * h),
            )
            status = "Обнаружен 1 человек — выбран автоматически"
        else:
            status = f"Обнаружено {len(persons)} человек. Выберите на превью или из списка."

        persons_out = [
            PersonInfo(
                track_id=p["track_id"],
                hits=p["hits"],
                bbox=p["bbox"],
                mid_hip=p["mid_hip"],
            )
            for p in persons
        ]

        return DetectResponse(
            persons=persons_out,
            preview_image=preview_b64,
            video_key=video_key,
            auto_click=auto_click,
            status=status,
        )

    finally:
        # Cleanup /tmp file
        if tmp_file:
            Path(tmp_file).unlink(missing_ok=True)
```

- [ ] **Step 2: Commit**

```bash
git add src/backend/routes/detect.py
git commit -m "refactor(detect): upload to R2, use /tmp for RTMPose, return video_key"
```

---

### Task 6: Remove SSE endpoint from process route

**Files:**
- Modify: `src/backend/routes/process.py`

- [ ] **Step 1: Rewrite process route — queue only**

Replace the entire `src/backend/routes/process.py` with:

```python
"""POST /api/process/queue — enqueue video processing job."""

from __future__ import annotations

import uuid

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, HTTPException

from src.backend.schemas import (
    ProcessRequest,
    ProcessResponse,
    ProcessStats,
    QueueProcessResponse,
    TaskStatusResponse,
)
from src.config import get_settings
from src.task_manager import (
    create_task_state,
    get_task_state,
    get_valkey_client,
    set_cancel_signal,
)
from src.worker import MLModelFlags

router = APIRouter()


@router.post("/process/queue", response_model=QueueProcessResponse)
async def enqueue_process(req: ProcessRequest):
    """Enqueue video processing job and return task_id immediately."""
    settings = get_settings()
    task_id = f"proc_{uuid.uuid4().hex[:12]}"

    valkey = await get_valkey_client()
    try:
        await create_task_state(task_id, video_key=req.video_key, valkey=valkey)
    finally:
        await valkey.close()

    ml_flags = MLModelFlags(
        depth=req.depth,
        optical_flow=req.optical_flow,
        segment=req.segment,
        foot_track=req.foot_track,
        matting=req.matting,
        inpainting=req.inpainting,
    )

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
            "process_video_task",
            task_id=task_id,
            video_key=req.video_key,
            person_click={"x": req.person_click.x, "y": req.person_click.y},
            frame_skip=req.frame_skip,
            layer=req.layer,
            tracking=req.tracking,
            export=req.export,
            ml_flags=ml_flags,
        )
    finally:
        await arq_pool.close()

    return QueueProcessResponse(task_id=task_id)


@router.get("/process/{task_id}/status", response_model=TaskStatusResponse)
async def get_process_status(task_id: str):
    """Poll task status."""
    valkey = await get_valkey_client()
    try:
        state = await get_task_state(task_id, valkey=valkey)
    finally:
        await valkey.close()

    if state is None:
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


@router.post("/process/{task_id}/cancel")
async def cancel_queued_process(task_id: str):
    """Cancel a queued or running task via Valkey signal."""
    await set_cancel_signal(task_id)
    return {"status": "cancel_requested", "task_id": task_id}
```

- [ ] **Step 2: Commit**

```bash
git add src/backend/routes/process.py
git commit -m "refactor(process): remove SSE endpoint, keep queue-only processing"
```

---

### Task 7: Update misc route — stream from R2 instead of disk

**Files:**
- Modify: `src/backend/routes/misc.py`

- [ ] **Step 1: Rewrite misc route**

Replace the entire `src/backend/routes/misc.py` with:

```python
"""Health check and file serving routes (R2 streaming proxy)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.storage import object_exists, stream_object

router = APIRouter(tags=["misc"])

# Content-type mapping by extension
_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".npy": "application/octet-stream",
    ".csv": "text/csv",
}


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/outputs/{key:path}")
async def serve_output(key: str):
    """Stream file from R2 as a proxy (frontend never talks to R2 directly)."""
    if not object_exists(key):
        raise HTTPException(status_code=404, detail="File not found")

    body, length, ctype = stream_object(key)
    # Prefer extension-based content type over what S3 reports
    from pathlib import Path

    ext = Path(key).suffix.lower()
    if ext in _CONTENT_TYPES:
        ctype = _CONTENT_TYPES[ext]

    return StreamingResponse(
        content=body.iter_chunks(chunk_size=8192),
        media_type=ctype,
        headers={"Content-Length": str(length)},
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/backend/routes/misc.py
git commit -m "refactor(misc): serve outputs from R2 via streaming proxy"
```

---

### Task 8: Update worker — thin dispatcher, no local GPU

**Files:**
- Modify: `src/worker.py`

- [ ] **Step 1: Rewrite worker**

Replace the entire `src/worker.py` with:

```python
"""arq worker for video processing pipeline.

Run with: uv run python -m src.worker

Dispatches all processing to Vast.ai Serverless GPU.
No local GPU fallback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar

from arq import Retry
from arq.connections import RedisSettings

from src.config import get_settings
from src.task_manager import (
    TaskStatus,
    get_valkey_client,
    is_cancelled,
    mark_cancelled,
    store_error,
    store_result,
    update_progress,
)

logger = logging.getLogger(__name__)


@dataclass
class MLModelFlags:
    """ML model feature flags for video processing."""

    depth: bool = False
    optical_flow: bool = False
    segment: bool = False
    foot_track: bool = False
    matting: bool = False
    inpainting: bool = False


async def startup(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker starting up")


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker shutting down")


async def process_video_task(
    ctx: dict[str, Any],
    *,
    task_id: str,
    video_key: str,
    person_click: dict[str, int],
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    ml_flags: MLModelFlags | None = None,
) -> dict[str, Any]:
    """arq task: dispatch video processing to Vast.ai Serverless GPU."""
    if ml_flags is None:
        ml_flags = MLModelFlags()
    settings = get_settings()
    valkey = await get_valkey_client()

    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"task:{task_id}",
            mapping={"status": TaskStatus.RUNNING, "started_at": now},
        )

        from src.vastai.client import process_video_remote

        logger.info("Dispatching task %s to Vast.ai (video_key=%s)", task_id, video_key)
        vast_result = await asyncio.to_thread(
            process_video_remote,
            video_key=video_key,
            person_click={"x": person_click["x"], "y": person_click["y"]},
            frame_skip=frame_skip,
            layer=layer,
            tracking=tracking,
            export=export,
            ml_flags={
                "depth": ml_flags.depth,
                "optical_flow": ml_flags.optical_flow,
                "segment": ml_flags.segment,
                "foot_track": ml_flags.foot_track,
                "matting": ml_flags.matting,
                "inpainting": ml_flags.inpainting,
            },
        )
        logger.info("Vast.ai processing complete for task %s", task_id)

        response_data = {
            "video_path": vast_result.video_key,
            "poses_path": vast_result.poses_key,
            "csv_path": vast_result.csv_key,
            "stats": vast_result.stats,
            "status": "Analysis complete!",
        }
        await store_result(task_id, response_data, valkey=valkey)
        return response_data

    except Exception as e:
        logger.exception("Pipeline task %s failed", task_id)
        await store_error(task_id, str(e), valkey=valkey)
        error_msg = str(e).lower()
        if any(term in error_msg for term in ["timeout", "connection", "network"]):
            raise Retry(defer=ctx.get("job_try", 1) * 10) from e
        raise

    finally:
        await valkey.close()


_settings = get_settings()


class WorkerSettings:
    """arq worker configuration."""

    queue_name: str = "skating:queue"
    max_jobs: int = _settings.app.worker_max_jobs
    retry_jobs: bool = True
    retry_delays: ClassVar[list[int]] = _settings.app.worker_retry_delays

    on_startup = startup
    on_shutdown = shutdown

    functions: ClassVar[list] = [process_video_task]
    cron_jobs: ClassVar[list] = []

    redis_settings = RedisSettings(
        host=_settings.valkey.host,
        port=_settings.valkey.port,
        database=_settings.valkey.db,
        password=_settings.valkey.password.get_secret_value(),
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/worker.py
git commit -m "refactor(worker): remove local GPU fallback, dispatch only to Vast.ai"
```

---

### Task 9: Update frontend types and schemas

**Files:**
- Modify: `src/frontend/src/types/index.ts`
- Modify: `src/frontend/src/lib/schemas.ts`
- Modify: `src/frontend/src/lib/api.ts`
- Modify: `src/frontend/src/test/handlers.ts`

- [ ] **Step 1: Update TypeScript types**

In `src/frontend/src/types/index.ts`, change `DetectResponse`:

```typescript
export interface DetectResponse {
  persons: PersonInfo[]
  preview_image: string // base64 PNG
  video_key: string
  auto_click: PersonClick | null
  status: string
}
```

- [ ] **Step 2: Update Zod schemas**

In `src/frontend/src/lib/schemas.ts`, change `DetectResponseSchema`:

```typescript
export const DetectResponseSchema = z.object({
  persons: z.array(PersonInfoSchema),
  preview_image: z.string().min(1),
  video_key: z.string().min(1),
  auto_click: PersonClickSchema.nullable(),
  status: z.string(),
})
```

- [ ] **Step 3: Update ProcessRequestSchema**

In `src/frontend/src/lib/schemas.ts`, change `ProcessRequestSchema`:

```typescript
export const ProcessRequestSchema = z.object({
  video_key: z.string().min(1),
  person_click: PersonClickSchema,
  frame_skip: z.number().int().positive().default(1),
  layer: z.number().int().min(0).max(3).default(3),
  tracking: z.enum(["auto", "manual"]).default("auto"),
  export: z.boolean().default(true),
  depth: z.boolean().default(false),
  optical_flow: z.boolean().default(false),
  segment: z.boolean().default(false),
  foot_track: z.boolean().default(false),
  matting: z.boolean().default(false),
  inpainting: z.boolean().default(false),
})
```

- [ ] **Step 4: Update ProcessRequest TypeScript type**

In `src/frontend/src/types/index.ts`, change `ProcessRequest`:

```typescript
export interface ProcessRequest {
  video_key: string
  person_click: PersonClick
  frame_skip: number
  layer: number
  tracking: string
  export: boolean
}
```

- [ ] **Step 5: Remove SSE processVideo function from api.ts**

In `src/frontend/src/lib/api.ts`, remove the entire `processVideo` function and its `SSECallbacks` interface (lines 104-162). Also remove the `cancelProcessing` function (lines 67-71). The imports from schemas stay the same.

- [ ] **Step 6: Update mock handlers**

Replace `src/frontend/src/test/handlers.ts`:

```typescript
import { HttpResponse, http } from "msw"

// Mock API handlers for /api/detect and /api/process
export const handlers = [
  // POST /api/detect
  http.post("/api/detect", async ({ request }) => {
    const formData = await request.formData()
    const file = formData.get("video") as File

    if (!file) {
      return HttpResponse.json({ detail: "No video file provided" }, { status: 400 })
    }

    // Mock response
    return HttpResponse.json({
      persons: [
        {
          track_id: 1,
          hits: 100,
          bbox: [0.3, 0.2, 0.7, 0.8],
          mid_hip: [0.5, 0.5],
        },
      ],
      preview_image: "data:image/png;base64,mock",
      video_key: "input/mock-video.mp4",
      auto_click: null,
      status: "success",
    })
  }),
]
```

- [ ] **Step 7: Commit**

```bash
git add src/frontend/src/types/index.ts src/frontend/src/lib/schemas.ts src/frontend/src/lib/api.ts src/frontend/src/test/handlers.ts
git commit -m "refactor(frontend): video_path → video_key, remove SSE processVideo"
```

---

### Task 10: Update frontend pages — queue-based processing

**Files:**
- Modify: `src/frontend/src/app/page.tsx`
- Modify: `src/frontend/src/app/analyze/page.tsx`

- [ ] **Step 1: Update home page — use video_key**

In `src/frontend/src/app/page.tsx`, change the `handleAnalyze` function (line 150-161):

```typescript
  const handleAnalyze = () => {
    if (!detectResult || !clickCoord) return
    const params = new URLSearchParams({
      video_key: detectResult.video_key,
      person_click: `${clickCoord.x},${clickCoord.y}`,
      frame_skip: String(frameSkip),
      layer: String(layer),
      tracking,
      export: String(doExport),
    })
    router.push(`/analyze?${params.toString()}`)
  }
```

- [ ] **Step 2: Rewrite analyze page — queue-based instead of SSE**

Replace the entire `src/frontend/src/app/analyze/page.tsx`:

```tsx
"use client"

import { AlertCircle, Loader2 } from "lucide-react"
import { useRouter, useSearchParams } from "next/navigation"
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { DownloadSection } from "@/components/dashboard/download-section"
import { StatsCards } from "@/components/dashboard/stats-cards"
import { VideoPlayer } from "@/components/dashboard/video-player"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { useTranslations } from "@/i18n"
import { cancelQueuedProcess, enqueueProcess, pollTaskStatus } from "@/lib/api"
import { toastError, toastSuccess } from "@/lib/toast"
import type { PersonClick, ProcessResponse } from "@/types"

type Phase = "processing" | "done" | "error"

function AnalyzeContent() {
  const params = useSearchParams()
  const router = useRouter()
  const t = useTranslations("analyze")

  const [phase, setPhase] = useState<Phase>("processing")
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState(t("starting"))
  const [result, setResult] = useState<ProcessResponse | null>(null)
  const [error, setError] = useState("")
  const [taskId, setTaskId] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const videoKey = params.get("video_key") || ""
  const clickParts = (params.get("person_click") || "0,0").split(",")
  const personClick: PersonClick = useMemo(
    () => ({
      x: Number(clickParts[0]),
      y: Number(clickParts[1]),
    }),
    [clickParts],
  )
  const frameSkip = Number(params.get("frame_skip") || 1)
  const layer = Number(params.get("layer") || 3)
  const tracking = params.get("tracking") || "auto"
  const doExport = params.get("export") !== "false"

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startProcessing = useCallback(async () => {
    setPhase("processing")
    setProgress(0)
    setMessage(t("preparing"))

    try {
      // Enqueue the job
      const { task_id } = await enqueueProcess({
        video_key: videoKey,
        person_click: personClick,
        frame_skip: frameSkip,
        layer: layer,
        tracking: tracking,
        export: doExport,
      })
      setTaskId(task_id)

      // Poll for results
      pollRef.current = setInterval(async () => {
        try {
          const status = await pollTaskStatus(task_id)

          if (status.progress) {
            setProgress(Math.round(status.progress * 100))
          }
          if (status.message) {
            setMessage(status.message)
          }

          if (status.status === "completed" && status.result) {
            stopPolling()
            setResult(status.result as ProcessResponse)
            setPhase("done")
            toastSuccess(t("complete"))
          } else if (status.status === "failed") {
            stopPolling()
            setError(status.error || "Processing failed")
            setPhase("error")
            toastError(status.error || "Processing failed")
          } else if (status.status === "cancelled") {
            stopPolling()
            setError("Cancelled")
            setPhase("error")
          }
        } catch (err) {
          console.error("Poll error:", err)
        }
      }, 1000)
    } catch (err) {
      setError(String(err))
      setPhase("error")
      toastError(String(err))
    }
  }, [videoKey, personClick, frameSkip, layer, tracking, doExport, t, stopPolling])

  const handleCancel = useCallback(async () => {
    stopPolling()
    if (taskId) {
      await cancelQueuedProcess(taskId).catch(() => {})
    }
    setError("Cancelled")
    setPhase("error")
  }, [taskId, stopPolling])

  useEffect(() => {
    if (videoKey) startProcessing()
    return () => stopPolling()
  }, [videoKey, startProcessing, stopPolling])

  const videoUrl = result ? `/api/v1/outputs/${result.video_path}` : ""
  const posesUrl = result?.poses_path ? `/api/v1/outputs/${result.poses_path}` : null
  const csvUrl = result?.csv_path ? `/api/v1/outputs/${result.csv_path}` : null

  return (
    <div>
      {/* Processing */}
      {phase === "processing" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <h2 className="text-lg font-medium">{t("title")}</h2>
            <Progress value={progress} className="w-full max-w-md" />
            <p className="text-sm text-muted-foreground">{message}</p>
            <p className="text-xs text-muted-foreground">{progress}%</p>
            <Button variant="outline" size="sm" onClick={handleCancel}>
              Cancel
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Done — dashboard */}
      {phase === "done" && result && (
        <div className="space-y-6">
          <StatsCards stats={result.stats} />
          <VideoPlayer src={videoUrl} />
          <DownloadSection videoUrl={videoUrl} posesUrl={posesUrl} csvUrl={csvUrl} />
        </div>
      )}

      {/* Error */}
      {phase === "error" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <AlertCircle className="h-8 w-8 text-destructive" />
            <p className="text-destructive">{error}</p>
            <div className="flex gap-2">
              <Button onClick={startProcessing}>{t("retry")}</Button>
              <Button variant="outline" onClick={() => router.push("/")}>
                {t("back")}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default function AnalyzePage() {
  return (
    <Suspense>
      <AnalyzeContent />
    </Suspense>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/app/page.tsx src/frontend/src/app/analyze/page.tsx
git commit -m "refactor(frontend): switch analyze page from SSE to queue-based processing"
```

---

### Task 11: Update task_manager — video_path → video_key

**Files:**
- Modify: `src/task_manager.py`

- [ ] **Step 1: Update create_task_state signature and hash field**

In `src/task_manager.py`, change the `create_task_state` function (lines 40-68):

Line 42: change parameter name from `video_path` to `video_key`:

```python
async def create_task_state(
    task_id: str,
    video_key: str,
    valkey: aioredis.Redis | None = None,
) -> None:
```

Line 56: change hash field name:

```python
                "video_key": video_key,
```

- [ ] **Step 2: Commit**

```bash
git add src/task_manager.py
git commit -m "refactor(task_manager): video_path → video_key"
```

---

### Task 12: Verify backend starts and run tests

**Files:**
- None (verification only)

- [ ] **Step 1: Run Python tests**

Run: `uv run pytest tests/ -v --timeout=30`
Expected: All tests PASS (existing tests should still work)

- [ ] **Step 2: Check for stale outputs_dir references**

Run: `grep -rn "outputs_dir\|OUTPUTS_DIR" src/ --include="*.py"`
Expected: No matches (all references removed)

- [ ] **Step 3: Check frontend compiles**

Run: `cd src/frontend && npx next build 2>&1 | head -50`
Expected: No type errors

- [ ] **Step 4: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix(s3-migration): cleanup stale references"
```
