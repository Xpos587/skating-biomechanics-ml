"""Tests for auth FastAPI dependencies."""

import pytest
from app.auth.deps import get_current_user
from app.auth.security import create_access_token, hash_password
from app.models import User
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_get_current_user_valid(db_session: AsyncSession):
    """Test get_current_user returns user for valid token."""
    user = User(email="test@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    token = create_access_token(user_id=user.id)
    result = await get_current_user(token=token, db=db_session)

    assert result.id == user.id
    assert result.email == "test@example.com"


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(db_session: AsyncSession):
    """Test get_current_user raises 401 for invalid token."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token="invalid.jwt.token", db=db_session)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_nonexistent_user(db_session: AsyncSession):
    """Test get_current_user raises 401 if user doesn't exist."""
    token = create_access_token(user_id="nonexistent-id")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token, db=db_session)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_inactive_user(db_session: AsyncSession):
    """Test get_current_user raises 401 for inactive user."""
    user = User(email="test@example.com", hashed_password="hash", is_active=False)
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    token = create_access_token(user_id=user.id)
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token, db=db_session)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_no_sub_claim(db_session: AsyncSession):
    """Test get_current_user raises 401 when token has no 'sub' claim."""
    import jwt as pyjwt
    from app.config import get_settings

    settings = get_settings()
    # Create a token without 'sub' claim
    payload = {"type": "access", "exp": int(__import__("time").time() + 3600)}
    token = pyjwt.encode(payload, settings.jwt.secret_key.get_secret_value(), algorithm="HS256")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token, db=db_session)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_dev_user_returns_first_active(db_session: AsyncSession):
    """Test _get_dev_user returns first active user."""
    from app.auth.deps import _get_dev_user

    u1 = User(id="user-1", email="first@example.com", hashed_password="hash", is_active=True)
    u2 = User(id="user-2", email="second@example.com", hashed_password="hash", is_active=True)
    db_session.add(u1)
    db_session.add(u2)
    await db_session.flush()

    result = await _get_dev_user(db_session)
    assert result.id == "user-1"


@pytest.mark.asyncio
async def test_get_dev_user_no_active_users(db_session: AsyncSession):
    """Test _get_dev_user raises 500 when no active users exist."""
    from app.auth.deps import _get_dev_user

    with pytest.raises(HTTPException) as exc_info:
        await _get_dev_user(db_session)
    assert exc_info.value.status_code == 500
    assert "No active users" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_dev_user_skips_inactive(db_session: AsyncSession):
    """Test _get_dev_user skips inactive users."""
    from app.auth.deps import _get_dev_user

    u1 = User(id="user-1", email="inactive@example.com", hashed_password="hash", is_active=False)
    u2 = User(id="user-2", email="active@example.com", hashed_password="hash", is_active=True)
    db_session.add(u1)
    db_session.add(u2)
    await db_session.flush()

    result = await _get_dev_user(db_session)
    assert result.id == "user-2"


@pytest.mark.asyncio
async def test_get_current_user_skip_auth_mode(db_session: AsyncSession):
    """Test get_current_user returns dev user when skip_auth is enabled."""
    from unittest.mock import patch

    user = User(id="dev-user", email="dev@example.com", hashed_password="hash", is_active=True)
    db_session.add(user)
    await db_session.flush()

    with patch("app.auth.deps.settings.app.skip_auth", True):
        result = await get_current_user(token="", db=db_session)
    assert result.id == "dev-user"
