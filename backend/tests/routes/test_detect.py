"""Tests for detect routes (enqueue, status, result)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Create test FastAPI app with detect routes and mock arq_pool on state."""
    from app.routes.detect import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.arq_pool = AsyncMock()
    return app


@pytest.fixture
async def client(app):
    """Create test HTTP client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_detect(client: AsyncClient, app):
    """POST /detect uploads video, creates task state, enqueues job."""
    video_content = b"fake-video-bytes"

    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.create_task_state", new_callable=AsyncMock),
        patch("app.routes.detect.upload_bytes_async", new_callable=AsyncMock),
    ):
        response = await client.post(
            "/api/v1/detect",
            files={"video": ("test.mp4", BytesIO(video_content), "video/mp4")},
            data={"tracking": "auto"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"].startswith("det_")
    assert data["video_key"].startswith("input/")
    assert data["video_key"].endswith(".mp4")
    assert data["status"] == "pending"

    app.state.arq_pool.enqueue_job.assert_awaited_once_with(
        "detect_video_task",
        task_id=data["task_id"],
        video_key=data["video_key"],
        tracking="auto",
        _queue_name="skating:queue:fast",
    )


@pytest.mark.asyncio
async def test_enqueue_detect_custom_tracking(client: AsyncClient, app):
    """POST /detect passes the tracking query parameter to the job."""
    video_content = b"fake-video-bytes"

    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.create_task_state", new_callable=AsyncMock),
        patch("app.routes.detect.upload_bytes_async", new_callable=AsyncMock),
    ):
        response = await client.post(
            "/api/v1/detect?tracking=manual",
            files={"video": ("test.webm", BytesIO(video_content), "video/webm")},
        )

    assert response.status_code == 200

    call_kwargs = app.state.arq_pool.enqueue_job.call_args.kwargs
    assert call_kwargs["tracking"] == "manual"


# ---------------------------------------------------------------------------
# GET /detect/{task_id}/status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_status(client: AsyncClient):
    """GET /detect/{task_id}/status returns task state without result."""
    fake_state = {
        "task_id": "det_abc123",
        "status": "running",
        "progress": 0.5,
        "message": "Processing",
        "result": None,
        "error": "",
    }

    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.get_task_state", new_callable=AsyncMock, return_value=fake_state),
    ):
        response = await client.get("/api/v1/detect/det_abc123/status")

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "det_abc123"
    assert data["status"] == "running"
    assert data["progress"] == 0.5
    assert data["message"] == "Processing"
    assert data["result"] is None
    assert data["error"] == ""


@pytest.mark.asyncio
async def test_detect_status_not_found(client: AsyncClient):
    """GET /detect/{task_id}/status returns 404 when task not found."""
    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.get_task_state", new_callable=AsyncMock, return_value=None),
    ):
        response = await client.get("/api/v1/detect/det_nonexist/status")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


@pytest.mark.asyncio
async def test_detect_status_with_error(client: AsyncClient):
    """GET /detect/{task_id}/status returns error field when present."""
    fake_state = {
        "task_id": "det_fail",
        "status": "failed",
        "progress": 0.3,
        "message": "Error",
        "result": None,
        "error": "CUDA out of memory",
    }

    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.get_task_state", new_callable=AsyncMock, return_value=fake_state),
    ):
        response = await client.get("/api/v1/detect/det_fail/status")

    assert response.status_code == 200
    data = response.json()
    assert data["error"] == "CUDA out of memory"


@pytest.mark.asyncio
async def test_detect_status_with_result_type_mismatch(client: AsyncClient):
    """GET /detect/{task_id}/status raises ValidationError when result is DetectResultResponse.

    TaskStatusResponse.result is typed as ProcessResponse | None, so embedding
    a DetectResultResponse causes a Pydantic validation error at serialization.
    This is a known type mismatch in the detect route (see # type: ignore on line 73).
    In production, ServerErrorMiddleware converts this to 500.
    """
    from pydantic import ValidationError

    fake_result = {
        "persons": [
            {
                "track_id": 1,
                "hits": 50,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "mid_hip": [0.5, 0.6],
            }
        ],
        "preview_image": "base64data",
        "video_key": "input/abc.mp4",
        "auto_click": {"x": 100, "y": 200},
        "status": "completed",
    }
    fake_state = {
        "task_id": "det_abc123",
        "status": "completed",
        "progress": 1.0,
        "message": "Done",
        "result": fake_result,
        "error": "",
    }

    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.get_task_state", new_callable=AsyncMock, return_value=fake_state),
        pytest.raises(ValidationError, match="ProcessResponse"),
    ):
        await client.get("/api/v1/detect/det_abc123/status")


# ---------------------------------------------------------------------------
# GET /detect/{task_id}/result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_result(client: AsyncClient):
    """GET /detect/{task_id}/result returns DetectResultResponse for completed task."""
    fake_result = {
        "persons": [
            {
                "track_id": 2,
                "hits": 100,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "mid_hip": [0.5, 0.5],
            }
        ],
        "preview_image": "abc123",
        "video_key": "input/test.webm",
        "auto_click": None,
        "status": "completed",
    }
    fake_state = {
        "task_id": "det_done",
        "status": "completed",
        "progress": 1.0,
        "message": "Done",
        "result": fake_result,
        "error": "",
    }

    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.get_task_state", new_callable=AsyncMock, return_value=fake_state),
    ):
        response = await client.get("/api/v1/detect/det_done/result")

    assert response.status_code == 200
    data = response.json()
    assert data["video_key"] == "input/test.webm"
    assert len(data["persons"]) == 1
    assert data["persons"][0]["track_id"] == 2
    assert data["auto_click"] is None


@pytest.mark.asyncio
async def test_detect_result_with_auto_click(client: AsyncClient):
    """GET /detect/{task_id}/result returns auto_click when present."""
    fake_result = {
        "persons": [],
        "preview_image": "img",
        "video_key": "input/x.mp4",
        "auto_click": {"x": 320, "y": 240},
        "status": "completed",
    }
    fake_state = {
        "task_id": "det_auto",
        "status": "completed",
        "progress": 1.0,
        "message": "Done",
        "result": fake_result,
        "error": "",
    }

    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.get_task_state", new_callable=AsyncMock, return_value=fake_state),
    ):
        response = await client.get("/api/v1/detect/det_auto/result")

    assert response.status_code == 200
    data = response.json()
    assert data["auto_click"] == {"x": 320, "y": 240}


@pytest.mark.asyncio
async def test_detect_result_not_found(client: AsyncClient):
    """GET /detect/{task_id}/result returns 404 when task not found."""
    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.get_task_state", new_callable=AsyncMock, return_value=None),
    ):
        response = await client.get("/api/v1/detect/det_ghost/result")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


@pytest.mark.asyncio
async def test_detect_result_not_completed(client: AsyncClient):
    """GET /detect/{task_id}/result returns 400 when task not completed."""
    fake_state = {
        "task_id": "det_running",
        "status": "running",
        "progress": 0.5,
        "message": "Processing",
        "result": None,
        "error": "",
    }

    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.get_task_state", new_callable=AsyncMock, return_value=fake_state),
    ):
        response = await client.get("/api/v1/detect/det_running/result")

    assert response.status_code == 400
    assert response.json()["detail"] == "Task not completed yet"


@pytest.mark.asyncio
async def test_detect_result_no_result(client: AsyncClient):
    """GET /detect/{task_id}/result returns 500 when task completed but no result stored."""
    fake_state = {
        "task_id": "det_empty",
        "status": "completed",
        "progress": 1.0,
        "message": "Done",
        "result": None,
        "error": "",
    }

    with (
        patch("app.routes.detect.get_valkey", return_value=MagicMock()),
        patch("app.routes.detect.get_task_state", new_callable=AsyncMock, return_value=fake_state),
    ):
        response = await client.get("/api/v1/detect/det_empty/result")

    assert response.status_code == 500
    assert response.json()["detail"] == "No result stored"
