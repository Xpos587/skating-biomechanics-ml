"""Direct handler tests for connections routes (coverage-tracked)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.models.connection import ConnectionStatus, ConnectionType
from app.routes.connections import (
    _conn_to_response,
    accept_invite,
    end_connection,
    invite,
    list_connections,
    list_pending,
)
from app.schemas import ConnectionResponse, InviteRequest
from fastapi import HTTPException


def _mock_user(user_id: str = "user_1", display_name: str = "Alice") -> MagicMock:
    u = MagicMock()
    u.id = user_id
    u.display_name = display_name
    return u


def _mock_conn_response(
    conn_id: str = "conn_1",
    from_user_id: str = "user_1",
    to_user_id: str = "user_2",
    connection_type: str = "coaching",
    status: str = "invited",
    from_user_name: str = "Alice",
    to_user_name: str = "Bob",
) -> ConnectionResponse:
    return ConnectionResponse(
        id=conn_id,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        connection_type=connection_type,
        status=status,
        initiated_by=from_user_id,
        created_at=datetime.now(UTC).isoformat(),
        ended_at=None,
        from_user_name=from_user_name,
        to_user_name=to_user_name,
    )


def _mock_conn(
    conn_id: str = "conn_1",
    from_user_id: str = "user_1",
    to_user_id: str = "user_2",
    connection_type: ConnectionType = ConnectionType.COACHING,
    status: ConnectionStatus = ConnectionStatus.INVITED,
) -> MagicMock:
    conn = MagicMock()
    conn.id = conn_id
    conn.from_user_id = from_user_id
    conn.to_user_id = to_user_id
    conn.connection_type = connection_type
    conn.status = status
    conn.initiated_by = from_user_id
    conn.ended_at = None
    conn.created_at = datetime.now(UTC)
    conn.from_user = MagicMock(display_name="Alice")
    conn.to_user = MagicMock(display_name="Bob")
    return conn


# ---------------------------------------------------------------------------
# invite
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_invite_success():
    body = InviteRequest(to_user_email="bob@test.com", connection_type="coaching")
    mock_user = _mock_user()
    mock_db = AsyncMock()
    target_user = MagicMock()
    target_user.id = "user_2"
    new_conn = _mock_conn()
    resp = _mock_conn_response()

    with (
        patch(
            "app.routes.connections.get_by_email", new_callable=AsyncMock, return_value=target_user
        ),
        patch("app.routes.connections.get_active_conn", new_callable=AsyncMock, return_value=None),
        patch("app.routes.connections.create_conn", new_callable=AsyncMock, return_value=new_conn),
        patch("app.routes.connections._conn_to_response", return_value=resp),
    ):
        result = await invite(body, mock_user, mock_db)

    assert result.from_user_id == "user_1"
    assert result.to_user_id == "user_2"
    assert result.from_user_name == "Alice"
    assert result.to_user_name == "Bob"


@pytest.mark.anyio
async def test_invite_user_not_found():
    body = InviteRequest(to_user_email="nobody@test.com", connection_type="coaching")
    mock_user = _mock_user()
    mock_db = AsyncMock()

    with (
        patch("app.routes.connections.get_by_email", new_callable=AsyncMock, return_value=None),
        pytest.raises(HTTPException, match="User not found"),
    ):
        await invite(body, mock_user, mock_db)


@pytest.mark.anyio
async def test_invite_connection_exists():
    body = InviteRequest(to_user_email="bob@test.com", connection_type="coaching")
    mock_user = _mock_user()
    mock_db = AsyncMock()
    target_user = MagicMock()
    target_user.id = "user_2"

    with (
        patch(
            "app.routes.connections.get_by_email", new_callable=AsyncMock, return_value=target_user
        ),
        patch(
            "app.routes.connections.get_active_conn",
            new_callable=AsyncMock,
            return_value=_mock_conn(),
        ),
        pytest.raises(HTTPException, match="Connection already exists"),
    ):
        await invite(body, mock_user, mock_db)


@pytest.mark.anyio
async def test_invite_choreography_type():
    body = InviteRequest(to_user_email="bob@test.com", connection_type="choreography")
    mock_user = _mock_user()
    mock_db = AsyncMock()
    target_user = MagicMock()
    target_user.id = "user_2"
    new_conn = _mock_conn(connection_type=ConnectionType.CHOREOGRAPHY)
    resp = _mock_conn_response(connection_type="choreography")

    with (
        patch(
            "app.routes.connections.get_by_email", new_callable=AsyncMock, return_value=target_user
        ),
        patch("app.routes.connections.get_active_conn", new_callable=AsyncMock, return_value=None),
        patch("app.routes.connections.create_conn", new_callable=AsyncMock, return_value=new_conn),
        patch("app.routes.connections._conn_to_response", return_value=resp),
    ):
        result = await invite(body, mock_user, mock_db)

    assert result.connection_type == "choreography"


# ---------------------------------------------------------------------------
# accept_invite
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_accept_invite_success():
    mock_user = _mock_user("user_2")
    mock_db = AsyncMock()
    conn = _mock_conn(status=ConnectionStatus.INVITED)
    resp = _mock_conn_response(status="active")

    with (
        patch("app.routes.connections.get_conn_by_id", new_callable=AsyncMock, return_value=conn),
        patch("app.routes.connections._conn_to_response", return_value=resp),
    ):
        result = await accept_invite("conn_1", mock_user, mock_db)

    assert conn.status == ConnectionStatus.ACTIVE
    mock_db.add.assert_called_once_with(conn)
    mock_db.flush.assert_called_once()
    mock_db.refresh.assert_called_once_with(conn)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "conn_status,user_id,error_match",
    [
        pytest.param(None, "user_2", "Connection not found", id="not_found"),
        pytest.param(ConnectionStatus.INVITED, "user_3", "Not authorized", id="wrong_user"),
        pytest.param(
            ConnectionStatus.ACTIVE, "user_2", "Not an active invite", id="already_active"
        ),
        pytest.param(ConnectionStatus.ENDED, "user_2", "Not an active invite", id="already_ended"),
    ],
)
async def test_accept_invite_error_cases(conn_status, user_id, error_match):
    mock_user = _mock_user(user_id)
    mock_db = AsyncMock()
    conn = None if conn_status is None else _mock_conn(status=conn_status, to_user_id="user_2")

    with (
        patch("app.routes.connections.get_conn_by_id", new_callable=AsyncMock, return_value=conn),
        pytest.raises(HTTPException, match=error_match),
    ):
        await accept_invite("conn_1", mock_user, mock_db)


# ---------------------------------------------------------------------------
# end_connection
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_end_connection_as_from_user():
    mock_user = _mock_user("user_1")
    mock_db = AsyncMock()
    conn = _mock_conn(status=ConnectionStatus.ACTIVE)
    resp = _mock_conn_response(status="ended")

    with (
        patch("app.routes.connections.get_conn_by_id", new_callable=AsyncMock, return_value=conn),
        patch("app.routes.connections._conn_to_response", return_value=resp),
    ):
        result = await end_connection("conn_1", mock_user, mock_db)

    assert conn.status == ConnectionStatus.ENDED
    assert conn.ended_at is not None
    mock_db.add.assert_called_once_with(conn)
    mock_db.flush.assert_called_once()


@pytest.mark.anyio
async def test_end_connection_as_to_user():
    mock_user = _mock_user("user_2")
    mock_db = AsyncMock()
    conn = _mock_conn(status=ConnectionStatus.ACTIVE)
    resp = _mock_conn_response(status="ended")

    with (
        patch("app.routes.connections.get_conn_by_id", new_callable=AsyncMock, return_value=conn),
        patch("app.routes.connections._conn_to_response", return_value=resp),
    ):
        result = await end_connection("conn_1", mock_user, mock_db)

    assert conn.status == ConnectionStatus.ENDED


@pytest.mark.anyio
@pytest.mark.parametrize(
    "conn_status,user_id,error_match",
    [
        pytest.param(None, "user_1", "Connection not found", id="not_found"),
        pytest.param(ConnectionStatus.ACTIVE, "user_3", "Not authorized", id="not_participant"),
        pytest.param(ConnectionStatus.ENDED, "user_1", "Already ended", id="already_ended"),
    ],
)
async def test_end_connection_error_cases(conn_status, user_id, error_match):
    mock_user = _mock_user(user_id)
    mock_db = AsyncMock()
    conn = None if conn_status is None else _mock_conn(status=conn_status)

    with (
        patch("app.routes.connections.get_conn_by_id", new_callable=AsyncMock, return_value=conn),
        pytest.raises(HTTPException, match=error_match),
    ):
        await end_connection("conn_1", mock_user, mock_db)


# ---------------------------------------------------------------------------
# list_connections
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_connections():
    mock_user = _mock_user()
    mock_db = AsyncMock()
    conns = [_mock_conn(conn_id="c1"), _mock_conn(conn_id="c2")]
    resp1 = _mock_conn_response(conn_id="c1")
    resp2 = _mock_conn_response(conn_id="c2")

    with (
        patch("app.routes.connections.list_for_user", new_callable=AsyncMock, return_value=conns),
        patch("app.routes.connections._conn_to_response", side_effect=[resp1, resp2]),
    ):
        result = await list_connections(mock_user, mock_db)

    assert len(result.connections) == 2
    assert result.connections[0].id == "c1"


@pytest.mark.anyio
async def test_list_connections_empty():
    mock_user = _mock_user()
    mock_db = AsyncMock()

    with patch("app.routes.connections.list_for_user", new_callable=AsyncMock, return_value=[]):
        result = await list_connections(mock_user, mock_db)

    assert result.connections == []


# ---------------------------------------------------------------------------
# list_pending
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_pending():
    mock_user = _mock_user()
    mock_db = AsyncMock()
    conns = [_mock_conn(conn_id="p1", status=ConnectionStatus.INVITED)]
    resp = _mock_conn_response(conn_id="p1", status="invited")

    with (
        patch(
            "app.routes.connections.list_pending_for_user",
            new_callable=AsyncMock,
            return_value=conns,
        ),
        patch("app.routes.connections._conn_to_response", return_value=resp),
    ):
        result = await list_pending(mock_user, mock_db)

    assert len(result.connections) == 1


@pytest.mark.anyio
async def test_list_pending_empty():
    mock_user = _mock_user()
    mock_db = AsyncMock()

    with patch(
        "app.routes.connections.list_pending_for_user", new_callable=AsyncMock, return_value=[]
    ):
        result = await list_pending(mock_user, mock_db)

    assert result.connections == []


# ---------------------------------------------------------------------------
# _conn_to_response (helper)
# ---------------------------------------------------------------------------


def test_conn_to_response_with_users():
    from_user = MagicMock()
    from_user.display_name = "Alice"
    to_user = MagicMock()
    to_user.display_name = "Bob"

    conn = type(
        "Connection",
        (),
        {
            "id": "conn_1",
            "from_user_id": "user_1",
            "to_user_id": "user_2",
            "connection_type": "coaching",
            "status": "active",
            "initiated_by": "user_1",
            "ended_at": None,
            "created_at": datetime.now(UTC),
            "from_user": from_user,
            "to_user": to_user,
        },
    )()

    result = _conn_to_response(conn)

    assert result.id == "conn_1"
    assert result.from_user_name == "Alice"
    assert result.to_user_name == "Bob"


def test_conn_to_response_without_users():
    conn = type(
        "Connection",
        (),
        {
            "id": "conn_2",
            "from_user_id": "user_1",
            "to_user_id": "user_2",
            "connection_type": "coaching",
            "status": "invited",
            "initiated_by": "user_1",
            "ended_at": None,
            "created_at": datetime.now(UTC),
            "from_user": None,
            "to_user": None,
        },
    )()

    result = _conn_to_response(conn)

    assert result.from_user_name is None
    assert result.to_user_name is None
