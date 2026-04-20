"""Tests for misc routes (health check, R2 output streaming)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Create test FastAPI app with misc routes."""
    from app.routes.misc import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
async def client(app):
    """Create test HTTP client (no DB needed for misc routes)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient):
    """GET /health returns {"status": "ok"} without authentication."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_serve_output_not_found(client: AsyncClient):
    """GET /outputs/{key} returns 404 when object does not exist in R2."""
    with patch("app.routes.misc.object_exists_async", new_callable=AsyncMock, return_value=False):
        response = await client.get("/api/v1/outputs/nonexistent/video.mp4")
    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


@pytest.mark.asyncio
async def test_serve_output_streams_file(client: AsyncClient):
    """GET /outputs/{key} streams the file from R2 with correct content-type."""
    fake_chunks = [b"chunk1", b"chunk2", b"chunk3"]

    async def fake_iter_chunks(*, chunk_size):
        for chunk in fake_chunks:
            yield chunk

    mock_body = MagicMock()
    mock_body.iter_chunks = fake_iter_chunks

    with (
        patch("app.routes.misc.object_exists_async", new_callable=AsyncMock, return_value=True),
        patch(
            "app.routes.misc.stream_object_async",
            new_callable=AsyncMock,
            return_value=(mock_body, 999, "application/octet-stream"),
        ),
    ):
        response = await client.get("/api/v1/outputs/session123/result.mp4")

    assert response.status_code == 200
    assert response.content == b"chunk1chunk2chunk3"
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-length"] == "999"


@pytest.mark.asyncio
async def test_serve_output_content_type_by_extension(client: AsyncClient):
    """GET /outputs/{key} overrides S3 content-type with extension-based type."""

    async def fake_iter_chunks(*, chunk_size):
        yield b"csv,data"

    mock_body = MagicMock()
    mock_body.iter_chunks = fake_iter_chunks

    with (
        patch("app.routes.misc.object_exists_async", new_callable=AsyncMock, return_value=True),
        patch(
            "app.routes.misc.stream_object_async",
            new_callable=AsyncMock,
            return_value=(mock_body, 9, "application/octet-stream"),
        ),
    ):
        response = await client.get("/api/v1/outputs/session123/metrics.csv")

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_serve_output_preserves_unknown_extension(client: AsyncClient):
    """GET /outputs/{key} uses S3-reported content-type for unknown extensions."""

    async def fake_iter_chunks(*, chunk_size):
        yield b"bin"

    mock_body = MagicMock()
    mock_body.iter_chunks = fake_iter_chunks

    with (
        patch("app.routes.misc.object_exists_async", new_callable=AsyncMock, return_value=True),
        patch(
            "app.routes.misc.stream_object_async",
            new_callable=AsyncMock,
            return_value=(mock_body, 3, "application/special-type"),
        ),
    ):
        response = await client.get("/api/v1/outputs/session123/data.xyz")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/special-type"


@pytest.mark.asyncio
async def test_serve_output_path_with_slashes(client: AsyncClient):
    """GET /outputs/{key:path} handles nested paths with slashes."""
    with patch("app.routes.misc.object_exists_async", new_callable=AsyncMock, return_value=False):
        response = await client.get("/api/v1/outputs/user/42/session/7/video.webm")

    assert response.status_code == 404
