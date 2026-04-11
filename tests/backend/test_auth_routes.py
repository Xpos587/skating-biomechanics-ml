"""Tests for auth API routes."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.auth.security import hash_password
from src.backend.models.user import User


@pytest.fixture
def app():
    """Create test FastAPI app with auth routes."""
    from fastapi import FastAPI

    from src.backend.routes.auth import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/auth")
    return app


@pytest.fixture
async def client(app, db_session: AsyncSession):
    """Create test HTTP client with DB override."""
    from src.backend.database import get_db

    app.dependency_overrides[get_db] = lambda: db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_register(client: AsyncClient, db_session: AsyncSession):
    """Test successful registration."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "new@example.com", "password": "securepass123"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

    # Verify user was created
    from sqlalchemy import select

    result = await db_session.execute(select(User).where(User.email == "new@example.com"))
    user = result.scalar_one_or_none()
    assert user is not None
    assert user.hashed_password != "securepass123"


async def test_register_duplicate_email(client: AsyncClient, db_session: AsyncSession):
    """Test registration with duplicate email returns 409."""
    user = User(email="exists@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()

    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "exists@example.com", "password": "securepass123"},
    )
    assert response.status_code == 409


async def test_register_short_password(client: AsyncClient):
    """Test registration with short password returns 422."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "new@example.com", "password": "short"},
    )
    assert response.status_code == 422


async def test_login(client: AsyncClient, db_session: AsyncSession):
    """Test successful login."""
    user = User(email="login@example.com", hashed_password=hash_password("pass123"))
    db_session.add(user)
    await db_session.flush()

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": "pass123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


async def test_login_wrong_password(client: AsyncClient, db_session: AsyncSession):
    """Test login with wrong password returns 401."""
    user = User(email="login@example.com", hashed_password=hash_password("correct"))
    db_session.add(user)
    await db_session.flush()

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": "wrong"},
    )
    assert response.status_code == 401


async def test_login_nonexistent_email(client: AsyncClient):
    """Test login with nonexistent email returns 401."""
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "pass123"},
    )
    assert response.status_code == 401


async def test_refresh_tokens(client: AsyncClient, db_session: AsyncSession):
    """Test refresh token rotation."""
    user = User(email="refresh@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    # Login to get tokens
    login_resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "refresh@example.com", "password": "pass"},
    )
    tokens = login_resp.json()
    old_refresh = tokens["refresh_token"]

    # Use refresh token
    refresh_resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": old_refresh},
    )
    assert refresh_resp.status_code == 200
    new_tokens = refresh_resp.json()
    assert new_tokens["refresh_token"] != old_refresh
    assert "access_token" in new_tokens

    # Old refresh token should be revoked
    second_refresh = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": old_refresh},
    )
    assert second_refresh.status_code == 401
