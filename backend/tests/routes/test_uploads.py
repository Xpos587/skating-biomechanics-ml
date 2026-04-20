"""Tests for uploads multipart route."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from app.auth.security import create_access_token, hash_password
from app.models.user import User
from app.routes.uploads import CHUNK_SIZE
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    from app.routes.uploads import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
async def client(app, db_session):
    from app.database import get_db

    app.dependency_overrides[get_db] = lambda: db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def authed_user(db_session):
    user = User(email="user@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(authed_user):
    token = create_access_token(user_id=authed_user.id)
    return {"Authorization": f"Bearer {token}"}


def _mock_r2():
    r2 = MagicMock()
    r2.create_multipart_upload.return_value = {"UploadId": "up_123"}
    r2.generate_presigned_url.return_value = "https://presigned.url/part"
    r2.complete_multipart_upload.return_value = {}
    return r2


def _mock_settings():
    cfg = MagicMock()
    cfg.r2.bucket = "test-bucket"
    return cfg


@pytest.mark.asyncio
async def test_init_upload(client: AsyncClient, auth_headers):
    """POST /uploads/init returns upload_id, key, chunk_size, part_count, parts."""
    with (
        patch("app.routes.uploads._client") as mock_s3_client,
        patch("app.routes.uploads.get_settings") as mock_settings,
    ):
        mock_s3_client.return_value = _mock_r2()
        mock_settings.return_value = _mock_settings()

        response = await client.post(
            "/api/v1/uploads/init",
            params={"file_name": "video.mp4", "total_size": 10000000},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["upload_id"] == "up_123"
    assert data["key"].startswith("uploads/")
    assert data["key"].endswith("/video.mp4")
    assert data["chunk_size"] == CHUNK_SIZE
    assert isinstance(data["part_count"], int)
    assert isinstance(data["parts"], list)
    assert len(data["parts"]) == data["part_count"]


@pytest.mark.asyncio
async def test_init_upload_part_count(client: AsyncClient, auth_headers):
    """15MB file / 5MB chunk = 3 parts."""
    with (
        patch("app.routes.uploads._client") as mock_s3_client,
        patch("app.routes.uploads.get_settings") as mock_settings,
    ):
        mock_s3_client.return_value = _mock_r2()
        mock_settings.return_value = _mock_settings()

        total_size = 15 * 1024 * 1024  # exactly 3 chunks
        response = await client.post(
            "/api/v1/uploads/init",
            params={"file_name": "big.mp4", "total_size": total_size},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["part_count"] == 3
    assert len(data["parts"]) == 3


@pytest.mark.asyncio
async def test_init_upload_single_part(client: AsyncClient, auth_headers):
    """3MB file = 1 part."""
    with (
        patch("app.routes.uploads._client") as mock_s3_client,
        patch("app.routes.uploads.get_settings") as mock_settings,
    ):
        mock_s3_client.return_value = _mock_r2()
        mock_settings.return_value = _mock_settings()

        total_size = 3 * 1024 * 1024  # fits in 1 chunk
        response = await client.post(
            "/api/v1/uploads/init",
            params={"file_name": "small.mp4", "total_size": total_size},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["part_count"] == 1
    assert len(data["parts"]) == 1


@pytest.mark.asyncio
async def test_complete_upload(client: AsyncClient, auth_headers):
    """POST /uploads/complete calls complete_multipart_upload with sorted parts."""
    with (
        patch("app.routes.uploads._client") as mock_s3_client,
        patch("app.routes.uploads.get_settings") as mock_settings,
    ):
        r2 = _mock_r2()
        mock_s3_client.return_value = r2
        mock_settings.return_value = _mock_settings()

        response = await client.post(
            "/api/v1/uploads/complete",
            json={
                "upload_id": "up_123",
                "key": "uploads/user-id/uuid/video.mp4",
                "parts": [
                    {"part_number": 1, "etag": '"etag1"'},
                    {"part_number": 2, "etag": '"etag2"'},
                ],
            },
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["key"] == "uploads/user-id/uuid/video.mp4"

    r2.complete_multipart_upload.assert_called_once()
    call_kwargs = r2.complete_multipart_upload.call_args
    parts = call_kwargs.kwargs["MultipartUpload"]["Parts"]
    assert parts == [
        {"PartNumber": 1, "ETag": '"etag1"'},
        {"PartNumber": 2, "ETag": '"etag2"'},
    ]


@pytest.mark.asyncio
async def test_complete_upload_empty_parts(client: AsyncClient, auth_headers):
    """POST /uploads/complete with no parts returns 400."""
    with (
        patch("app.routes.uploads._client") as mock_s3_client,
        patch("app.routes.uploads.get_settings") as mock_settings,
    ):
        mock_s3_client.return_value = _mock_r2()
        mock_settings.return_value = _mock_settings()

        response = await client.post(
            "/api/v1/uploads/complete",
            json={
                "upload_id": "up_123",
                "key": "uploads/user-id/uuid/video.mp4",
                "parts": [],
            },
            headers=auth_headers,
        )

    assert response.status_code == 400
    assert "No parts provided" in response.json()["detail"]


@pytest.mark.asyncio
async def test_complete_upload_parts_sorted(client: AsyncClient, auth_headers):
    """Parts are sorted by part_number regardless of input order."""
    with (
        patch("app.routes.uploads._client") as mock_s3_client,
        patch("app.routes.uploads.get_settings") as mock_settings,
    ):
        r2 = _mock_r2()
        mock_s3_client.return_value = r2
        mock_settings.return_value = _mock_settings()

        response = await client.post(
            "/api/v1/uploads/complete",
            json={
                "upload_id": "up_123",
                "key": "uploads/user-id/uuid/video.mp4",
                "parts": [
                    {"part_number": 3, "etag": '"etag3"'},
                    {"part_number": 1, "etag": '"etag1"'},
                    {"part_number": 2, "etag": '"etag2"'},
                ],
            },
            headers=auth_headers,
        )

    assert response.status_code == 200

    call_kwargs = r2.complete_multipart_upload.call_args
    parts = call_kwargs.kwargs["MultipartUpload"]["Parts"]
    assert parts == [
        {"PartNumber": 1, "ETag": '"etag1"'},
        {"PartNumber": 2, "ETag": '"etag2"'},
        {"PartNumber": 3, "ETag": '"etag3"'},
    ]


@pytest.mark.asyncio
async def test_init_upload_auth_required(client: AsyncClient):
    """POST /uploads/init without auth returns 401."""
    with (
        patch("app.routes.uploads._client") as mock_s3_client,
        patch("app.routes.uploads.get_settings") as mock_settings,
    ):
        mock_s3_client.return_value = _mock_r2()
        mock_settings.return_value = _mock_settings()

        response = await client.post(
            "/api/v1/uploads/init",
            params={"file_name": "video.mp4", "total_size": 10000000},
        )

    assert response.status_code == 401
