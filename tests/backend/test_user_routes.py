"""Tests for user API routes."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.auth.security import create_access_token, hash_password
from src.backend.models import User


@pytest.fixture
def app():
    from fastapi import FastAPI

    from src.backend.routes.users import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/users")
    return app


@pytest.fixture
async def client(app, db_session: AsyncSession):
    from src.backend.database import get_db

    app.dependency_overrides[get_db] = lambda: db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def authed_user(db_session: AsyncSession) -> User:
    user = User(
        email="user@example.com",
        hashed_password=hash_password("pass"),
        display_name="Test User",
        bio="Skater",
        height_cm=175,
        weight_kg=70.0,
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(authed_user):
    token = create_access_token(user_id=authed_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient, auth_headers):
    """Test GET /api/users/me returns current user."""
    response = await client.get("/api/v1/users/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "user@example.com"
    assert data["display_name"] == "Test User"
    assert data["bio"] == "Skater"
    assert data["height_cm"] == 175
    assert data["weight_kg"] == 70.0


@pytest.mark.asyncio
async def test_get_me_unauthorized(client: AsyncClient):
    """Test GET /api/users/me without auth returns 401."""
    response = await client.get("/api/v1/users/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_update_profile(client: AsyncClient, auth_headers):
    """Test PATCH /api/users/me updates profile fields."""
    response = await client.patch(
        "/api/v1/users/me",
        json={"display_name": "New Name", "bio": "Updated bio", "height_cm": 180},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["display_name"] == "New Name"
    assert data["bio"] == "Updated bio"
    assert data["height_cm"] == 180


@pytest.mark.asyncio
async def test_update_settings(client: AsyncClient, auth_headers):
    """Test PATCH /api/users/me/settings updates preferences."""
    response = await client.patch(
        "/api/v1/users/me/settings",
        json={"language": "en", "theme": "dark"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "en"
    assert data["theme"] == "dark"
