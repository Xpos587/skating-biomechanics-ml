"""Tests for ORM models."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models import RefreshToken, User


@pytest.mark.asyncio
async def test_create_user(db_session: AsyncSession):
    """Test creating a user and reading it back."""
    user = User(
        email="test@example.com",
        hashed_password="hashed",
        display_name="Test User",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    result = await db_session.execute(select(User).where(User.email == "test@example.com"))
    fetched = result.scalar_one()

    assert fetched.email == "test@example.com"
    assert fetched.display_name == "Test User"
    assert fetched.is_active is True
    assert fetched.language == "ru"
    assert fetched.theme == "system"
    assert fetched.id is not None


@pytest.mark.asyncio
async def test_create_refresh_token(db_session: AsyncSession):
    """Test creating a refresh token linked to a user."""
    user = User(email="test@example.com", hashed_password="hashed")
    db_session.add(user)
    await db_session.flush()

    token = RefreshToken(
        user_id=user.id,
        token_hash="a" * 64,
        family_id="b" * 36,
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
    )
    db_session.add(token)
    await db_session.flush()
    await db_session.refresh(token)

    assert token.user_id == user.id
    assert token.is_revoked is False


@pytest.mark.asyncio
async def test_user_timestamps(db_session: AsyncSession):
    """Test that created_at is set automatically."""
    user = User(email="test@example.com", hashed_password="hashed")
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    assert user.created_at is not None
    assert user.updated_at is not None
