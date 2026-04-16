"""Tests for ORM models."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models import RefreshToken, Session, User


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


@pytest.mark.asyncio
async def test_session_json_columns(db_session: AsyncSession):
    """Test that pose_data and frame_metrics JSON columns work."""
    user = User(email="test@example.com", hashed_password="hashed")
    db_session.add(user)
    await db_session.flush()

    session = Session(
        id=str(uuid.uuid4()),
        user_id=user.id,
        element_type="waltz_jump",
        pose_data={"frames": [0, 10, 20], "poses": [[[0.5, 0.5, 0.9]]], "fps": 30.0},
        frame_metrics={"knee_angles_r": [120.5, 115.0, 130.0]},
    )
    db_session.add(session)
    await db_session.flush()
    await db_session.refresh(session)

    assert session.pose_data == {"frames": [0, 10, 20], "poses": [[[0.5, 0.5, 0.9]]], "fps": 30.0}
    assert session.frame_metrics == {"knee_angles_r": [120.5, 115.0, 130.0]}


@pytest.mark.asyncio
async def test_session_backward_compatibility(db_session: AsyncSession):
    """Test that old poses_url and csv_url columns still work (deprecated)."""
    user = User(email="test@example.com", hashed_password="hashed")
    db_session.add(user)
    await db_session.flush()

    # Old-style session with URLs instead of JSON
    session = Session(
        id=str(uuid.uuid4()),
        user_id=user.id,
        element_type="waltz_jump",
        poses_url="https://r2.example.com/poses.npy",
        csv_url="https://r2.example.com/metrics.csv",
    )
    db_session.add(session)
    await db_session.flush()
    await db_session.refresh(session)

    assert session.poses_url == "https://r2.example.com/poses.npy"
    assert session.csv_url == "https://r2.example.com/metrics.csv"
    assert session.pose_data is None
    assert session.frame_metrics is None
