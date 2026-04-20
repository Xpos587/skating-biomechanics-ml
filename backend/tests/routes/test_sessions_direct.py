"""Direct handler tests for sessions routes (coverage-tracked)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.routes.sessions import (
    _session_to_response,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    patch_session,
)
from app.schemas import CreateSessionRequest, PatchSessionRequest, SessionResponse
from fastapi import HTTPException


def _mock_user(user_id: str = "user_1") -> MagicMock:
    u = MagicMock()
    u.id = user_id
    return u


def _mock_session(
    session_id: str = "sess_1",
    user_id: str = "user_1",
    element_type: str = "waltz_jump",
    status: str = "done",
    video_key: str | None = "videos/test.mp4",
    processed_video_key: str | None = None,
    video_url: str | None = None,
    processed_video_url: str | None = None,
) -> MagicMock:
    s = MagicMock()
    s.id = session_id
    s.user_id = user_id
    s.element_type = element_type
    s.status = status
    s.video_key = video_key
    s.video_url = video_url
    s.processed_video_key = processed_video_key
    s.processed_video_url = processed_video_url
    s.poses_url = None
    s.csv_url = None
    s.pose_data = None
    s.frame_metrics = None
    s.error_message = None
    s.phases = None
    s.recommendations = None
    s.overall_score = None
    s.created_at = datetime.now(UTC)
    s.processed_at = None
    s.metrics = []
    return s


def _mock_session_response(session_id: str = "sess_1", user_id: str = "user_1") -> SessionResponse:
    """Create a real SessionResponse for mocking _session_to_response."""
    return SessionResponse(
        id=session_id,
        user_id=user_id,
        element_type="waltz_jump",
        video_key="videos/test.mp4",
        video_url="https://example.com/video.mp4",
        processed_video_key=None,
        processed_video_url=None,
        poses_url=None,
        csv_url=None,
        pose_data=None,
        frame_metrics=None,
        status="done",
        error_message=None,
        phases=None,
        recommendations=None,
        overall_score=None,
        created_at=datetime.now(UTC).isoformat(),
        processed_at=None,
        metrics=[],
    )


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_session_with_video_key():
    body = CreateSessionRequest(element_type="waltz_jump", video_key="videos/test.mp4")
    mock_user = _mock_user()
    mock_db = AsyncMock()
    session = _mock_session()
    resp = _mock_session_response()

    with (
        patch(
            "app.routes.sessions.create", new_callable=AsyncMock, return_value=session
        ) as mock_create,
        patch(
            "app.routes.sessions._session_to_response", new_callable=AsyncMock, return_value=resp
        ),
    ):
        result = await create_session(body, mock_user, mock_db)

    mock_create.assert_called_once_with(
        mock_db,
        user_id="user_1",
        element_type="waltz_jump",
        video_key="videos/test.mp4",
        status="queued",
    )


@pytest.mark.anyio
async def test_create_session_without_video_key():
    body = CreateSessionRequest(element_type="three_turn", video_key=None)
    mock_user = _mock_user()
    mock_db = AsyncMock()
    session = _mock_session(video_key=None)
    resp = _mock_session_response()

    with (
        patch(
            "app.routes.sessions.create", new_callable=AsyncMock, return_value=session
        ) as mock_create,
        patch(
            "app.routes.sessions._session_to_response", new_callable=AsyncMock, return_value=resp
        ),
    ):
        result = await create_session(body, mock_user, mock_db)

    mock_create.assert_called_once_with(
        mock_db, user_id="user_1", element_type="three_turn", video_key=None, status="uploading"
    )


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_sessions_own():
    mock_user = _mock_user()
    mock_db = AsyncMock()

    with (
        patch("app.routes.sessions.list_by_user", new_callable=AsyncMock, return_value=[]),
        patch("app.routes.sessions.count_by_user", new_callable=AsyncMock, return_value=0),
    ):
        result = await list_sessions(mock_user, mock_db)

    assert result.total == 0
    assert result.sessions == []


@pytest.mark.anyio
async def test_list_sessions_with_filters():
    mock_user = _mock_user()
    mock_db = AsyncMock()
    sessions = [_mock_session(session_id="s1")]
    resp = _mock_session_response(session_id="s1")

    with (
        patch("app.routes.sessions.list_by_user", new_callable=AsyncMock, return_value=sessions),
        patch("app.routes.sessions.count_by_user", new_callable=AsyncMock, return_value=1),
        patch(
            "app.routes.sessions._session_to_response", new_callable=AsyncMock, return_value=resp
        ),
    ):
        result = await list_sessions(
            mock_user, mock_db, element_type="toe_loop", limit=10, offset=5, sort="overall_score"
        )

    assert result.total == 1


@pytest.mark.anyio
async def test_list_sessions_coach_access_allowed():
    mock_user = _mock_user("coach_1")
    mock_db = AsyncMock()

    with (
        patch("app.routes.sessions.is_connected_as", new_callable=AsyncMock, return_value=True),
        patch("app.routes.sessions.list_by_user", new_callable=AsyncMock, return_value=[]),
        patch("app.routes.sessions.count_by_user", new_callable=AsyncMock, return_value=0),
    ):
        result = await list_sessions(mock_user, mock_db, user_id="student_1")

    assert result.total == 0


@pytest.mark.anyio
async def test_list_sessions_coach_access_denied():
    mock_user = _mock_user("coach_1")
    mock_db = AsyncMock()

    with (
        patch("app.routes.sessions.is_connected_as", new_callable=AsyncMock, return_value=False),
        pytest.raises(HTTPException, match="Not a coach"),
    ):
        await list_sessions(mock_user, mock_db, user_id="student_1")


@pytest.mark.anyio
async def test_list_sessions_coach_viewing_own_sessions():
    """Coach requesting own user_id should work without connection check."""
    mock_user = _mock_user("coach_1")
    mock_db = AsyncMock()

    with (
        patch("app.routes.sessions.list_by_user", new_callable=AsyncMock, return_value=[]),
        patch("app.routes.sessions.count_by_user", new_callable=AsyncMock, return_value=0),
    ):
        result = await list_sessions(mock_user, mock_db, user_id="coach_1")

    assert result.total == 0


# ---------------------------------------------------------------------------
# get_session / patch_session / delete_session — not-found & wrong-user
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    "handler,call_args",
    [
        pytest.param(get_session, ("sess_1",), id="get_session"),
        pytest.param(
            patch_session, ("sess_1", PatchSessionRequest(element_type="lutz")), id="patch_session"
        ),
        pytest.param(delete_session, ("sess_1",), id="delete_session"),
    ],
)
async def test_session_not_found(handler, call_args):
    mock_user = _mock_user()
    mock_db = AsyncMock()

    with (
        patch("app.routes.sessions.get_by_id", new_callable=AsyncMock, return_value=None),
        pytest.raises(HTTPException, match="Session not found"),
    ):
        await handler(*call_args, mock_user, mock_db)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "handler,call_args",
    [
        pytest.param(
            patch_session, ("sess_1", PatchSessionRequest(element_type="lutz")), id="patch_session"
        ),
        pytest.param(delete_session, ("sess_1",), id="delete_session"),
    ],
)
async def test_session_wrong_user(handler, call_args):
    mock_user = _mock_user("user_2")
    mock_db = AsyncMock()
    session = _mock_session(user_id="user_1")

    with (
        patch("app.routes.sessions.get_by_id", new_callable=AsyncMock, return_value=session),
        pytest.raises(HTTPException, match="Not authorized"),
    ):
        await handler(*call_args, mock_user, mock_db)


# ---------------------------------------------------------------------------
# get_session — specific tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_session_own():
    mock_user = _mock_user()
    mock_db = AsyncMock()
    session = _mock_session()
    resp = _mock_session_response()

    with (
        patch("app.routes.sessions.get_by_id", new_callable=AsyncMock, return_value=session),
        patch(
            "app.routes.sessions._session_to_response", new_callable=AsyncMock, return_value=resp
        ),
    ):
        result = await get_session("sess_1", mock_user, mock_db)

    assert result is resp


@pytest.mark.anyio
async def test_get_session_not_found():
    mock_user = _mock_user()
    mock_db = AsyncMock()

    with (
        patch("app.routes.sessions.get_by_id", new_callable=AsyncMock, return_value=None),
        pytest.raises(HTTPException, match="Session not found"),
    ):
        await get_session("bad_id", mock_user, mock_db)


@pytest.mark.anyio
async def test_get_session_wrong_user():
    mock_user = _mock_user("user_2")
    mock_db = AsyncMock()
    session = _mock_session(user_id="user_1")

    with (
        patch("app.routes.sessions.get_by_id", new_callable=AsyncMock, return_value=session),
        patch("app.routes.sessions.is_connected_as", new_callable=AsyncMock, return_value=False),
        pytest.raises(HTTPException, match="Not authorized"),
    ):
        await get_session("sess_1", mock_user, mock_db)


@pytest.mark.anyio
async def test_get_session_coach_access():
    """Coach can view student's session when connected."""
    mock_user = _mock_user("coach_1")
    mock_db = AsyncMock()
    session = _mock_session(user_id="student_1")
    resp = _mock_session_response(user_id="student_1")

    with (
        patch("app.routes.sessions.get_by_id", new_callable=AsyncMock, return_value=session),
        patch("app.routes.sessions.is_connected_as", new_callable=AsyncMock, return_value=True),
        patch(
            "app.routes.sessions._session_to_response", new_callable=AsyncMock, return_value=resp
        ),
    ):
        result = await get_session("sess_1", mock_user, mock_db)

    assert result is resp


# ---------------------------------------------------------------------------
# patch_session — specific tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_patch_session_success():
    body = PatchSessionRequest(element_type="lutz")
    mock_user = _mock_user()
    mock_db = AsyncMock()
    session = _mock_session()
    resp = _mock_session_response()

    with (
        patch("app.routes.sessions.get_by_id", new_callable=AsyncMock, return_value=session),
        patch("app.routes.sessions.update", new_callable=AsyncMock, return_value=session),
        patch(
            "app.routes.sessions._session_to_response", new_callable=AsyncMock, return_value=resp
        ),
    ):
        result = await patch_session("sess_1", body, mock_user, mock_db)

    assert result is resp


@pytest.mark.anyio
async def test_patch_session_empty_body():
    """Patch with no fields set should still work."""
    body = PatchSessionRequest()
    mock_user = _mock_user()
    mock_db = AsyncMock()
    session = _mock_session()
    resp = _mock_session_response()

    with (
        patch("app.routes.sessions.get_by_id", new_callable=AsyncMock, return_value=session),
        patch(
            "app.routes.sessions.update", new_callable=AsyncMock, return_value=session
        ) as mock_update,
        patch(
            "app.routes.sessions._session_to_response", new_callable=AsyncMock, return_value=resp
        ),
    ):
        result = await patch_session("sess_1", body, mock_user, mock_db)

    mock_update.assert_called_once()


# ---------------------------------------------------------------------------
# delete_session — specific tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_session_success():
    mock_user = _mock_user()
    mock_db = AsyncMock()
    session = _mock_session()

    with (
        patch("app.routes.sessions.get_by_id", new_callable=AsyncMock, return_value=session),
        patch("app.routes.sessions.soft_delete", new_callable=AsyncMock),
    ):
        result = await delete_session("sess_1", mock_user, mock_db)

    assert result is None  # 204 No Content


# ---------------------------------------------------------------------------
# _session_to_response (helper)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_session_to_response_with_video_key():
    session = _mock_session(video_key="videos/test.mp4", video_url=None)

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://r2.example.com/videos/test.mp4",
    ):
        result = await _session_to_response(session)

    assert result.video_url == "https://r2.example.com/videos/test.mp4"
    assert result.video_key == "videos/test.mp4"


@pytest.mark.anyio
async def test_session_to_response_without_video_key():
    session = _mock_session(video_key=None, video_url="https://direct.example.com/video.mp4")

    result = await _session_to_response(session)

    assert result.video_url == "https://direct.example.com/video.mp4"


@pytest.mark.anyio
async def test_session_to_response_with_processed_video_key():
    session = _mock_session(processed_video_key="processed/test.mp4", processed_video_url=None)

    with patch(
        "app.routes.sessions.get_object_url_async",
        new_callable=AsyncMock,
        return_value="https://r2.example.com/processed/test.mp4",
    ):
        result = await _session_to_response(session)

    assert result.processed_video_url == "https://r2.example.com/processed/test.mp4"


@pytest.mark.anyio
async def test_session_to_response_without_processed_video_key():
    session = _mock_session(
        video_key=None,
        video_url="https://direct.example.com/video.mp4",
        processed_video_key=None,
        processed_video_url="https://direct.example.com/processed.mp4",
    )

    result = await _session_to_response(session)

    assert result.processed_video_url == "https://direct.example.com/processed.mp4"
