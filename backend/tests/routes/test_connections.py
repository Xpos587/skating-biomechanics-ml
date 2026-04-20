"""Tests for connections API routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.auth.security import create_access_token, hash_password
from app.models.user import User
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
def app():
    from app.routes.connections import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
async def client(app, db_session: AsyncSession):
    from app.database import get_db

    app.dependency_overrides[get_db] = lambda: db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def user_a(db_session: AsyncSession) -> User:
    user = User(email="a@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def user_b(db_session: AsyncSession) -> User:
    user = User(email="b@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers_a(user_a):
    token = create_access_token(user_id=user_a.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_b(user_b):
    token = create_access_token(user_id=user_b.id)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# POST /connections/invite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invite_creates_connection(
    client: AsyncClient, user_a: User, user_b: User, auth_headers_a
):
    """Inviting an existing user creates a connection with INVITED status."""
    response = await client.post(
        "/api/v1/connections/invite",
        json={"to_user_email": "b@example.com", "connection_type": "coaching"},
        headers=auth_headers_a,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["from_user_id"] == user_a.id
    assert data["to_user_id"] == user_b.id
    assert data["connection_type"] == "coaching"
    assert data["status"] == "invited"
    assert data["initiated_by"] == user_a.id
    assert data["ended_at"] is None or data["ended_at"] == "None"


@pytest.mark.asyncio
async def test_invite_user_not_found(client: AsyncClient, auth_headers_a):
    """Inviting a nonexistent email returns 404."""
    response = await client.post(
        "/api/v1/connections/invite",
        json={"to_user_email": "nobody@example.com", "connection_type": "coaching"},
        headers=auth_headers_a,
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"


@pytest.mark.asyncio
async def test_invite_duplicate(client: AsyncClient, user_a: User, user_b: User, auth_headers_a):
    """Inviting the same user twice returns 409."""
    payload = {"to_user_email": "b@example.com", "connection_type": "coaching"}
    first = await client.post("/api/v1/connections/invite", json=payload, headers=auth_headers_a)
    assert first.status_code == 201

    second = await client.post("/api/v1/connections/invite", json=payload, headers=auth_headers_a)
    assert second.status_code == 409
    assert second.json()["detail"] == "Connection already exists"


# ---------------------------------------------------------------------------
# POST /connections/{conn_id}/accept
# ---------------------------------------------------------------------------


@pytest.fixture
async def invited_connection(client, user_a, user_b, auth_headers_a):
    """Create an INVITED connection from user_a to user_b and return its ID."""
    resp = await client.post(
        "/api/v1/connections/invite",
        json={"to_user_email": "b@example.com", "connection_type": "coaching"},
        headers=auth_headers_a,
    )
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_accept_invite(client: AsyncClient, invited_connection: str, auth_headers_b):
    """Invitee accepts the connection, status becomes ACTIVE."""
    response = await client.post(
        f"/api/v1/connections/{invited_connection}/accept",
        headers=auth_headers_b,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"


@pytest.mark.asyncio
async def test_accept_invite_not_found(client: AsyncClient, auth_headers_b):
    """Accepting a nonexistent connection returns 404."""
    response = await client.post(
        "/api/v1/connections/nonexistent-id/accept",
        headers=auth_headers_b,
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Connection not found"


@pytest.mark.asyncio
async def test_accept_invite_wrong_user(
    client: AsyncClient, invited_connection: str, auth_headers_a
):
    """Non-invitee (the sender) accepting returns 403."""
    response = await client.post(
        f"/api/v1/connections/{invited_connection}/accept",
        headers=auth_headers_a,
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized"


@pytest.mark.asyncio
async def test_accept_invite_already_active(
    client: AsyncClient, invited_connection: str, auth_headers_b
):
    """Accepting a connection that is already ACTIVE returns 400."""
    # First accept succeeds
    await client.post(
        f"/api/v1/connections/{invited_connection}/accept",
        headers=auth_headers_b,
    )
    # Second accept fails
    response = await client.post(
        f"/api/v1/connections/{invited_connection}/accept",
        headers=auth_headers_b,
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Not an active invite"


# ---------------------------------------------------------------------------
# POST /connections/{conn_id}/end
# ---------------------------------------------------------------------------


@pytest.fixture
async def active_connection(client, user_a, user_b, auth_headers_a, auth_headers_b):
    """Create an ACTIVE connection from user_a to user_b and return its ID."""
    invite_resp = await client.post(
        "/api/v1/connections/invite",
        json={"to_user_email": "b@example.com", "connection_type": "coaching"},
        headers=auth_headers_a,
    )
    conn_id = invite_resp.json()["id"]
    await client.post(
        f"/api/v1/connections/{conn_id}/accept",
        headers=auth_headers_b,
    )
    return conn_id


@pytest.mark.asyncio
async def test_end_connection_by_sender(
    client: AsyncClient, active_connection: str, auth_headers_a
):
    """The connection initiator can end the connection."""
    response = await client.post(
        f"/api/v1/connections/{active_connection}/end",
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ended"
    assert data["ended_at"] is not None


@pytest.mark.asyncio
async def test_end_connection_by_receiver(
    client: AsyncClient, active_connection: str, auth_headers_b
):
    """The connection receiver can also end the connection."""
    response = await client.post(
        f"/api/v1/connections/{active_connection}/end",
        headers=auth_headers_b,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ended"


@pytest.mark.asyncio
async def test_end_connection_not_found(client: AsyncClient, auth_headers_a):
    """Ending a nonexistent connection returns 404."""
    response = await client.post(
        "/api/v1/connections/nonexistent-id/end",
        headers=auth_headers_a,
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Connection not found"


@pytest.mark.asyncio
async def test_end_connection_not_party(
    client: AsyncClient, active_connection: str, db_session: AsyncSession
):
    """A third user who is not part of the connection gets 403."""
    third = User(email="third@example.com", hashed_password=hash_password("pass"))
    db_session.add(third)
    await db_session.flush()
    await db_session.refresh(third)
    token = create_access_token(user_id=third.id)
    headers = {"Authorization": f"Bearer {token}"}

    response = await client.post(
        f"/api/v1/connections/{active_connection}/end",
        headers=headers,
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized"


@pytest.mark.asyncio
async def test_end_connection_already_ended(
    client: AsyncClient, active_connection: str, auth_headers_a
):
    """Ending an already-ended connection returns 400."""
    await client.post(
        f"/api/v1/connections/{active_connection}/end",
        headers=auth_headers_a,
    )
    response = await client.post(
        f"/api/v1/connections/{active_connection}/end",
        headers=auth_headers_a,
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Already ended"


# ---------------------------------------------------------------------------
# GET /connections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_connections(client: AsyncClient, active_connection: str, auth_headers_a):
    """User can list their connections."""
    response = await client.get(
        "/api/v1/connections",
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert "connections" in data
    assert len(data["connections"]) >= 1
    conn = data["connections"][0]
    assert conn["status"] == "active"


# ---------------------------------------------------------------------------
# GET /connections/pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_pending(client: AsyncClient, invited_connection: str, auth_headers_b):
    """User can list pending invites they received."""
    response = await client.get(
        "/api/v1/connections/pending",
        headers=auth_headers_b,
    )
    assert response.status_code == 200
    data = response.json()
    assert "connections" in data
    assert len(data["connections"]) >= 1
    conn_ids = [c["id"] for c in data["connections"]]
    assert invited_connection in conn_ids


@pytest.mark.asyncio
async def test_list_pending_empty_for_sender(
    client: AsyncClient, invited_connection: str, auth_headers_a
):
    """The sender has no pending invites (they are the inviter, not invitee)."""
    response = await client.get(
        "/api/v1/connections/pending",
        headers=auth_headers_a,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["connections"] == []
