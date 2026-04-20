"""Tests for sessions CRUD routes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from app.auth.security import create_access_token, hash_password
from app.models.user import User
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
def app():
    from app.routes.sessions import router
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
async def authed_user(db_session: AsyncSession) -> User:
    user = User(email="user@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(authed_user):
    token = create_access_token(user_id=authed_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def other_user(db_session: AsyncSession) -> User:
    user = User(email="other@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
def other_headers(other_user):
    token = create_access_token(user_id=other_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_create_session(client: AsyncClient, auth_headers, db_session: AsyncSession):
    """POST /sessions with element_type returns 201 and response fields."""
    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.post(
            "/api/v1/sessions",
            json={"element_type": "waltz_jump"},
            headers=auth_headers,
        )

    assert response.status_code == 201
    data = response.json()
    assert data["element_type"] == "waltz_jump"
    assert data["status"] == "uploading"
    assert data["user_id"]
    assert data["id"]
    assert data["created_at"]


@pytest.mark.asyncio
async def test_create_session_with_video_key(
    client: AsyncClient, auth_headers, db_session: AsyncSession
):
    """POST /sessions with video_key sets status to 'queued'."""
    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.post(
            "/api/v1/sessions",
            json={"element_type": "toe_loop", "video_key": "uploads/user123/video.mp4"},
            headers=auth_headers,
        )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "queued"
    assert data["video_key"] == "uploads/user123/video.mp4"
    assert data["video_url"] == "https://fake.url"


@pytest.mark.asyncio
async def test_create_session_without_video_key(
    client: AsyncClient, auth_headers, db_session: AsyncSession
):
    """POST /sessions without video_key sets status to 'uploading'."""
    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.post(
            "/api/v1/sessions",
            json={"element_type": "flip"},
            headers=auth_headers,
        )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "uploading"
    assert data["video_key"] is None


@pytest.mark.asyncio
async def test_list_sessions(
    client: AsyncClient, auth_headers, authed_user, db_session: AsyncSession
):
    """GET /sessions returns list with total count."""
    from app.crud.session import create as crud_create

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        await crud_create(db_session, user_id=authed_user.id, element_type="waltz_jump")
        await crud_create(db_session, user_id=authed_user.id, element_type="toe_loop")

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.get("/api/v1/sessions", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["sessions"]) == 2


@pytest.mark.asyncio
async def test_list_sessions_filter_element_type(
    client: AsyncClient, auth_headers, authed_user, db_session: AsyncSession
):
    """GET /sessions?element_type=... filters by element type."""
    from app.crud.session import create as crud_create

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        await crud_create(db_session, user_id=authed_user.id, element_type="waltz_jump")
        await crud_create(db_session, user_id=authed_user.id, element_type="toe_loop")

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.get(
            "/api/v1/sessions",
            params={"element_type": "waltz_jump"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["sessions"][0]["element_type"] == "waltz_jump"


@pytest.mark.asyncio
async def test_list_sessions_coach_access_denied(
    client: AsyncClient, auth_headers, other_user, db_session: AsyncSession
):
    """GET /sessions?user_id=other without coaching connection returns 403."""
    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.get(
            "/api/v1/sessions",
            params={"user_id": other_user.id},
            headers=auth_headers,
        )

    assert response.status_code == 403
    assert "Not a coach" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_session(
    client: AsyncClient, auth_headers, authed_user, db_session: AsyncSession
):
    """GET /sessions/{id} returns the session."""
    from app.crud.session import create as crud_create

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        session = await crud_create(db_session, user_id=authed_user.id, element_type="axel")

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.get(f"/api/v1/sessions/{session.id}", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session.id
    assert data["element_type"] == "axel"


@pytest.mark.asyncio
async def test_get_session_not_found(client: AsyncClient, auth_headers):
    """GET /sessions/{id} with nonexistent id returns 404."""
    response = await client.get("/api/v1/sessions/nonexistent-id", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_session_forbidden(
    client: AsyncClient, other_headers, authed_user, db_session: AsyncSession
):
    """GET /sessions/{id} for another user's session returns 403."""
    from app.crud.session import create as crud_create

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        session = await crud_create(db_session, user_id=authed_user.id, element_type="lutz")

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.get(f"/api/v1/sessions/{session.id}", headers=other_headers)

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_patch_session(
    client: AsyncClient, auth_headers, authed_user, db_session: AsyncSession
):
    """PATCH /sessions/{id} updates element_type."""
    from app.crud.session import create as crud_create

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        session = await crud_create(db_session, user_id=authed_user.id, element_type="flip")

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.patch(
            f"/api/v1/sessions/{session.id}",
            json={"element_type": "lutz"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["element_type"] == "lutz"


@pytest.mark.asyncio
async def test_patch_session_not_found(client: AsyncClient, auth_headers):
    """PATCH /sessions/{id} with nonexistent id returns 404."""
    response = await client.patch(
        "/api/v1/sessions/nonexistent-id",
        json={"element_type": "lutz"},
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_session(
    client: AsyncClient, auth_headers, authed_user, db_session: AsyncSession
):
    """DELETE /sessions/{id} returns 204 and soft-deletes the session."""
    from app.crud.session import create as crud_create
    from app.crud.session import get_by_id

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        session = await crud_create(db_session, user_id=authed_user.id, element_type="salchow")

    response = await client.delete(f"/api/v1/sessions/{session.id}", headers=auth_headers)
    assert response.status_code == 204

    # Verify soft-deleted (status changed to "deleted")
    await db_session.refresh(session)
    assert session.status == "deleted"


@pytest.mark.asyncio
async def test_delete_session_not_found(client: AsyncClient, auth_headers):
    """DELETE /sessions/{id} with nonexistent id returns 404."""
    response = await client.delete("/api/v1/sessions/nonexistent-id", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_forbidden(
    client: AsyncClient, other_headers, authed_user, db_session: AsyncSession
):
    """DELETE /sessions/{id} for another user's session returns 403."""
    from app.crud.session import create as crud_create

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        session = await crud_create(db_session, user_id=authed_user.id, element_type="loop")

    response = await client.delete(f"/api/v1/sessions/{session.id}", headers=other_headers)
    assert response.status_code == 403


@pytest.fixture
async def coach_user(db_session: AsyncSession) -> User:
    user = User(email="coach@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def skater_user(db_session: AsyncSession) -> User:
    user = User(email="skater@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
def coach_headers(coach_user):
    token = create_access_token(user_id=coach_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def skater_headers(skater_user):
    token = create_access_token(user_id=skater_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def coaching_connection(db_session: AsyncSession, coach_user, skater_user):
    """Create an active coaching connection: coach -> skater."""
    from app.models.connection import Connection, ConnectionStatus, ConnectionType

    conn = Connection(
        from_user_id=coach_user.id,
        to_user_id=skater_user.id,
        connection_type=ConnectionType.COACHING,
        status=ConnectionStatus.ACTIVE,
        initiated_by=coach_user.id,
    )
    db_session.add(conn)
    await db_session.flush()
    await db_session.refresh(conn)
    return conn


@pytest.mark.asyncio
async def test_list_sessions_coach_allowed(
    client: AsyncClient, coach_headers, skater_user, coaching_connection, db_session: AsyncSession
):
    """GET /sessions?user_id=skater with active coaching connection returns 200."""
    from app.crud.session import create as crud_create

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        await crud_create(db_session, user_id=skater_user.id, element_type="waltz_jump")

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.get(
            "/api/v1/sessions",
            params={"user_id": skater_user.id},
            headers=coach_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["sessions"][0]["element_type"] == "waltz_jump"


@pytest.mark.asyncio
async def test_get_session_coach_allowed(
    client: AsyncClient, coach_headers, skater_user, coaching_connection, db_session: AsyncSession
):
    """GET /sessions/{id} for student's session returns 200 when coach."""
    from app.crud.session import create as crud_create

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        session = await crud_create(db_session, user_id=skater_user.id, element_type="axel")

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://fake.url",
    ):
        response = await client.get(f"/api/v1/sessions/{session.id}", headers=coach_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session.id
