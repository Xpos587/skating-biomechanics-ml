"""Tests for models route (list available ML models on disk)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Create test FastAPI app with models routes."""
    from app.routes.models import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
async def client(app):
    """Create test HTTP client (no DB needed for models route)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def fake_models_dir(tmp_path):
    """Create a temp directory with some model files present and some missing."""
    # Create files that "exist"
    (tmp_path / "depth_anything_v2_small.onnx").write_bytes(b"\x00" * (5 * 1024 * 1024))  # 5MB
    (tmp_path / "neuflowv2_mixed.onnx").write_bytes(
        b"\x00" * (12 * 1024 * 1024 + 500_000)
    )  # ~12.5MB -> rounds to 12.5

    # Create subdirectory for segment model
    sam2_dir = tmp_path / "sam2"
    sam2_dir.mkdir()
    (sam2_dir / "vision_encoder.onnx").write_bytes(b"\x00" * (45 * 1024 * 1024))  # 45MB

    # foot_tracker.onnx, rvm_mobilenetv3.onnx, lama_fp32.onnx are NOT created
    return tmp_path


@pytest.mark.asyncio
async def test_list_models_returns_six_entries(client: AsyncClient, fake_models_dir: Path):
    """GET /models returns exactly 6 model entries."""
    with patch("app.routes.models._MODELS_DIR", fake_models_dir):
        response = await client.get("/api/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 6


@pytest.mark.asyncio
async def test_list_models_available_flags(client: AsyncClient, fake_models_dir: Path):
    """GET /models correctly reports available=True/False based on file existence."""
    with patch("app.routes.models._MODELS_DIR", fake_models_dir):
        response = await client.get("/api/v1/models")

    data = response.json()
    by_id = {m["id"]: m for m in data}

    # These files were created
    assert by_id["depth"]["available"] is True
    assert by_id["optical_flow"]["available"] is True
    assert by_id["segment"]["available"] is True

    # These files were NOT created
    assert by_id["foot_track"]["available"] is False
    assert by_id["matting"]["available"] is False
    assert by_id["inpainting"]["available"] is False


@pytest.mark.asyncio
async def test_list_models_size_mb_when_available(client: AsyncClient, fake_models_dir: Path):
    """GET /models returns size_mb for available models, None for missing ones."""
    with patch("app.routes.models._MODELS_DIR", fake_models_dir):
        response = await client.get("/api/v1/models")

    data = response.json()
    by_id = {m["id"]: m for m in data}

    # Available models have size_mb set
    assert by_id["depth"]["size_mb"] == 5.0
    assert by_id["optical_flow"]["size_mb"] == 12.5
    assert by_id["segment"]["size_mb"] == 45.0

    # Missing models have size_mb=None
    assert by_id["foot_track"]["size_mb"] is None
    assert by_id["matting"]["size_mb"] is None
    assert by_id["inpainting"]["size_mb"] is None


@pytest.mark.asyncio
async def test_list_models_all_missing(client: AsyncClient, tmp_path: Path):
    """GET /models with no files returns all available=False."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()

    with patch("app.routes.models._MODELS_DIR", empty_dir):
        response = await client.get("/api/v1/models")

    data = response.json()
    assert len(data) == 6
    for model in data:
        assert model["available"] is False
        assert model["size_mb"] is None


@pytest.mark.asyncio
async def test_list_models_response_schema(client: AsyncClient, fake_models_dir: Path):
    """GET /models response matches ModelStatus schema (id, available, size_mb)."""
    with patch("app.routes.models._MODELS_DIR", fake_models_dir):
        response = await client.get("/api/v1/models")

    data = response.json()
    for model in data:
        assert "id" in model
        assert "available" in model
        assert "size_mb" in model
        assert isinstance(model["available"], bool)
