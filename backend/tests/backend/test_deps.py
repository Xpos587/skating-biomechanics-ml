"""Tests for auth FastAPI dependencies."""

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.deps import get_current_user
from backend.app.auth.security import create_access_token, hash_password
from backend.app.models import User


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
