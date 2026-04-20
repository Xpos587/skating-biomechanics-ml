"""Tests for User CRUD operations."""

from app.crud.user import create, get_by_email, get_by_id, update
from app.models.user import User


async def test_create_user(db_session):
    user = await create(
        db_session,
        email="test@example.com",
        hashed_password="hash123",
    )
    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.hashed_password == "hash123"
    assert user.is_active is True


async def test_create_user_with_kwargs(db_session):
    user = await create(
        db_session,
        email="test@example.com",
        hashed_password="hash123",
        display_name="Test User",
        height_cm=175,
        weight_kg=70.0,
        language="en",
    )
    assert user.display_name == "Test User"
    assert user.height_cm == 175
    assert user.weight_kg == 70.0
    assert user.language == "en"


async def test_get_by_id(db_session):
    user = await create(
        db_session,
        email="test@example.com",
        hashed_password="hash123",
    )
    fetched = await get_by_id(db_session, user.id)
    assert fetched is not None
    assert fetched.email == "test@example.com"


async def test_get_by_id_not_found(db_session):
    fetched = await get_by_id(db_session, "nonexistent")
    assert fetched is None


async def test_get_by_email(db_session):
    await create(
        db_session,
        email="findme@example.com",
        hashed_password="hash123",
    )
    fetched = await get_by_email(db_session, "findme@example.com")
    assert fetched is not None
    assert fetched.email == "findme@example.com"


async def test_get_by_email_not_found(db_session):
    fetched = await get_by_email(db_session, "nobody@example.com")
    assert fetched is None


async def test_update_user(db_session):
    user = await create(
        db_session,
        email="test@example.com",
        hashed_password="hash123",
        display_name="Old Name",
    )
    updated = await update(
        db_session,
        user,
        display_name="New Name",
        bio="Updated bio",
    )
    assert updated.display_name == "New Name"
    assert updated.bio == "Updated bio"


async def test_update_user_ignores_none(db_session):
    user = await create(
        db_session,
        email="test@example.com",
        hashed_password="hash123",
        display_name="Original",
    )
    updated = await update(
        db_session,
        user,
        display_name=None,
        bio="New bio",
    )
    # display_name should remain unchanged because None was passed
    assert updated.display_name == "Original"
    assert updated.bio == "New bio"
