"""Tests for Connection CRUD operations."""

from app.crud.connection import (
    create,
    get_active,
    get_by_id,
    is_connected_as,
    list_for_user,
    list_pending_for_user,
)
from app.models.connection import Connection, ConnectionStatus, ConnectionType
from app.models.user import User


def _make_user(db, user_id: str) -> User:
    """Create a minimal User row for FK references."""
    user = User(
        id=user_id,
        email=f"{user_id}@test.com",
        hashed_password="hash",
    )
    db.add(user)
    return user


async def test_create_connection(db_session):
    """Verify all fields are set correctly on creation."""
    u1 = _make_user(db_session, "user-1")
    u2 = _make_user(db_session, "user-2")
    await db_session.flush()

    conn = await create(
        db_session,
        from_user_id=u1.id,
        to_user_id=u2.id,
        connection_type=ConnectionType.COACHING,
        initiated_by=u1.id,
    )
    assert conn.id is not None
    assert conn.from_user_id == u1.id
    assert conn.to_user_id == u2.id
    assert conn.connection_type == ConnectionType.COACHING
    assert conn.status == ConnectionStatus.INVITED
    assert conn.initiated_by == u1.id


async def test_get_by_id(db_session):
    _make_user(db_session, "user-1")
    _make_user(db_session, "user-2")
    await db_session.flush()

    conn = await create(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.COACHING,
        initiated_by="user-1",
    )
    fetched = await get_by_id(db_session, conn.id)
    assert fetched is not None
    assert fetched.id == conn.id


async def test_get_by_id_not_found(db_session):
    fetched = await get_by_id(db_session, "nonexistent-id")
    assert fetched is None


async def test_get_active(db_session):
    """INVITED status counts as active (not ended)."""
    _make_user(db_session, "user-1")
    _make_user(db_session, "user-2")
    await db_session.flush()

    await create(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.CHOREOGRAPHY,
        initiated_by="user-1",
    )

    result = await get_active(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.CHOREOGRAPHY,
    )
    assert result is not None
    assert result.status == ConnectionStatus.INVITED


async def test_get_active_ended_returns_none(db_session):
    """ENDED connections should not be returned by get_active."""
    _make_user(db_session, "user-1")
    _make_user(db_session, "user-2")
    await db_session.flush()

    conn = await create(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.COACHING,
        initiated_by="user-1",
    )
    conn.status = ConnectionStatus.ENDED
    await db_session.flush()

    result = await get_active(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.COACHING,
    )
    assert result is None


async def test_list_for_user(db_session):
    """Both parties should see the connection in their list."""
    _make_user(db_session, "user-1")
    _make_user(db_session, "user-2")
    _make_user(db_session, "user-3")
    await db_session.flush()

    await create(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.COACHING,
        initiated_by="user-1",
    )
    await create(
        db_session,
        from_user_id="user-3",
        to_user_id="user-2",
        connection_type=ConnectionType.CHOREOGRAPHY,
        initiated_by="user-3",
    )

    list_1 = await list_for_user(db_session, "user-1")
    assert len(list_1) == 1

    list_2 = await list_for_user(db_session, "user-2")
    assert len(list_2) == 2

    list_3 = await list_for_user(db_session, "user-3")
    assert len(list_3) == 1


async def test_list_pending_for_user(db_session):
    """Only to_user with INVITED status should see pending."""
    _make_user(db_session, "user-1")
    _make_user(db_session, "user-2")
    _make_user(db_session, "user-3")
    await db_session.flush()

    # INVITED connection where user-2 is the recipient
    await create(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.COACHING,
        initiated_by="user-1",
    )
    # ACTIVE connection where user-2 is the recipient
    conn = await create(
        db_session,
        from_user_id="user-3",
        to_user_id="user-2",
        connection_type=ConnectionType.CHOREOGRAPHY,
        initiated_by="user-3",
    )
    conn.status = ConnectionStatus.ACTIVE
    await db_session.flush()

    pending = await list_pending_for_user(db_session, "user-2")
    assert len(pending) == 1
    assert pending[0].from_user_id == "user-1"


async def test_is_connected_as_true(db_session):
    """ACTIVE connection grants access."""
    _make_user(db_session, "user-1")
    _make_user(db_session, "user-2")
    await db_session.flush()

    conn = await create(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.COACHING,
        initiated_by="user-1",
    )
    conn.status = ConnectionStatus.ACTIVE
    await db_session.flush()

    assert await is_connected_as(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.COACHING,
    ) is True


async def test_is_connected_as_false(db_session):
    """No connection returns False."""
    _make_user(db_session, "user-1")
    _make_user(db_session, "user-2")
    await db_session.flush()

    assert await is_connected_as(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.COACHING,
    ) is False


async def test_is_connected_as_wrong_type(db_session):
    """Coaching connection does not grant choreography access."""
    _make_user(db_session, "user-1")
    _make_user(db_session, "user-2")
    await db_session.flush()

    conn = await create(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.COACHING,
        initiated_by="user-1",
    )
    conn.status = ConnectionStatus.ACTIVE
    await db_session.flush()

    assert await is_connected_as(
        db_session,
        from_user_id="user-1",
        to_user_id="user-2",
        connection_type=ConnectionType.CHOREOGRAPHY,
    ) is False
