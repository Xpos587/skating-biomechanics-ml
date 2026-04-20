"""Tests for RefreshToken CRUD operations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.crud.refresh_token import create, get_active_by_hash, get_by_hash, revoke, revoke_family
from app.models.user import User


def _make_user(db, user_id: str = "user-1") -> User:
    user = User(id=user_id, email=f"{user_id}@test.com", hashed_password="hash")
    db.add(user)
    return user


async def test_create_and_get_by_hash(db_session):
    _make_user(db_session)
    await db_session.flush()

    token = await create(
        db_session,
        user_id="user-1",
        token_hash="hash1",
        family_id="family-a",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    assert token.id is not None
    assert token.is_revoked is False

    fetched = await get_by_hash(db_session, "hash1")
    assert fetched is not None
    assert fetched.id == token.id


async def test_get_by_hash_not_found(db_session):
    fetched = await get_by_hash(db_session, "nonexistent")
    assert fetched is None


async def test_revoke_single_token(db_session):
    _make_user(db_session)
    await db_session.flush()

    token = await create(
        db_session,
        user_id="user-1",
        token_hash="hash1",
        family_id="family-a",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    assert token.is_revoked is False

    await revoke(db_session, token)
    assert token.is_revoked is True

    # Should still be in DB but revoked
    fetched = await get_by_hash(db_session, "hash1")
    assert fetched is not None
    assert fetched.is_revoked is True


async def test_revoke_family(db_session):
    _make_user(db_session)
    await db_session.flush()

    # Create 3 tokens in the same family
    await create(
        db_session,
        user_id="user-1",
        token_hash="hash1",
        family_id="family-a",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    await create(
        db_session,
        user_id="user-1",
        token_hash="hash2",
        family_id="family-a",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    await create(
        db_session,
        user_id="user-1",
        token_hash="hash3",
        family_id="family-b",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )

    count = await revoke_family(db_session, "family-a")
    assert count == 2

    # Tokens in family-a should be revoked
    assert (await get_by_hash(db_session, "hash1")).is_revoked is True
    assert (await get_by_hash(db_session, "hash2")).is_revoked is True

    # Token in family-b should NOT be revoked
    assert (await get_by_hash(db_session, "hash3")).is_revoked is False


async def test_revoke_family_skips_already_revoked(db_session):
    _make_user(db_session)
    await db_session.flush()

    t1 = await create(
        db_session,
        user_id="user-1",
        token_hash="hash1",
        family_id="family-a",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    # Manually revoke one before batch revoke
    t1.is_revoked = True
    await db_session.flush()

    await create(
        db_session,
        user_id="user-1",
        token_hash="hash2",
        family_id="family-a",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )

    count = await revoke_family(db_session, "family-a")
    # Only 1 non-revoked token should be revoked
    assert count == 1


async def test_revoke_family_empty(db_session):
    count = await revoke_family(db_session, "nonexistent-family")
    assert count == 0


async def test_get_active_by_hash_valid(db_session):
    _make_user(db_session)
    await db_session.flush()

    await create(
        db_session,
        user_id="user-1",
        token_hash="hash1",
        family_id="family-a",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )

    active = await get_active_by_hash(db_session, "hash1")
    assert active is not None
    assert active.is_revoked is False


async def test_get_active_by_hash_revoked(db_session):
    _make_user(db_session)
    await db_session.flush()

    token = await create(
        db_session,
        user_id="user-1",
        token_hash="hash1",
        family_id="family-a",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    await revoke(db_session, token)

    active = await get_active_by_hash(db_session, "hash1")
    assert active is None


async def test_get_active_by_hash_expired(db_session):
    _make_user(db_session)
    await db_session.flush()

    await create(
        db_session,
        user_id="user-1",
        token_hash="hash1",
        family_id="family-a",
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    )

    active = await get_active_by_hash(db_session, "hash1")
    assert active is None


async def test_get_active_by_hash_not_found(db_session):
    active = await get_active_by_hash(db_session, "nonexistent")
    assert active is None
