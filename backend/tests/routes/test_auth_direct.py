"""Direct handler tests for auth routes (coverage-tracked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.routes.auth import _issue_token_pair, login, logout, refresh, register
from app.schemas import LoginRequest, RefreshRequest, RegisterRequest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_register_success():
    body = RegisterRequest(email="new@test.com", password="securepass123", display_name="New User")
    mock_db = AsyncMock()
    new_user = MagicMock()
    new_user.id = "user_new"

    with (
        patch("app.routes.auth.get_by_email", new_callable=AsyncMock, return_value=None),
        patch("app.routes.auth.create_user", new_callable=AsyncMock, return_value=new_user),
        patch("app.routes.auth._issue_token_pair", new_callable=AsyncMock) as mock_issue,
    ):
        mock_issue.return_value = MagicMock(access_token="at", refresh_token="rt")
        result = await register(body, mock_db)

    assert result.access_token == "at"
    assert result.refresh_token == "rt"


@pytest.mark.anyio
async def test_register_email_taken():
    body = RegisterRequest(email="taken@test.com", password="securepass123")
    mock_db = AsyncMock()

    with (
        patch("app.routes.auth.get_by_email", new_callable=AsyncMock, return_value=MagicMock()),
        pytest.raises(HTTPException, match="Email already registered"),
    ):
        await register(body, mock_db)


@pytest.mark.anyio
async def test_register_without_display_name():
    body = RegisterRequest(email="new@test.com", password="securepass123", display_name=None)
    mock_db = AsyncMock()
    new_user = MagicMock()
    new_user.id = "user_new"

    with (
        patch("app.routes.auth.get_by_email", new_callable=AsyncMock, return_value=None),
        patch("app.routes.auth.create_user", new_callable=AsyncMock, return_value=new_user),
        patch("app.routes.auth._issue_token_pair", new_callable=AsyncMock) as mock_issue,
    ):
        mock_issue.return_value = MagicMock(access_token="at", refresh_token="rt")
        result = await register(body, mock_db)

    assert result.access_token == "at"


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_login_success():
    body = LoginRequest(email="user@test.com", password="correctpass")
    mock_db = AsyncMock()
    user = MagicMock()
    user.id = "user_1"
    user.hashed_password = "$2b$12$fakehashfortest"

    with (
        patch("app.routes.auth.get_by_email", new_callable=AsyncMock, return_value=user),
        patch("app.routes.auth.verify_password", return_value=True),
        patch("app.routes.auth._issue_token_pair", new_callable=AsyncMock) as mock_issue,
    ):
        mock_issue.return_value = MagicMock(access_token="at", refresh_token="rt")
        result = await login(body, mock_db)

    assert result.access_token == "at"


@pytest.mark.anyio
async def test_login_user_not_found():
    body = LoginRequest(email="nobody@test.com", password="whatever")
    mock_db = AsyncMock()

    with (
        patch("app.routes.auth.get_by_email", new_callable=AsyncMock, return_value=None),
        patch("app.routes.auth.verify_password", return_value=False),
        pytest.raises(HTTPException, match="Invalid email or password"),
    ):
        await login(body, mock_db)


@pytest.mark.anyio
async def test_login_wrong_password():
    body = LoginRequest(email="user@test.com", password="wrongpass")
    mock_db = AsyncMock()
    user = MagicMock()
    user.hashed_password = "$2b$12$fakehashfortest"

    with (
        patch("app.routes.auth.get_by_email", new_callable=AsyncMock, return_value=user),
        patch("app.routes.auth.verify_password", return_value=False),
        pytest.raises(HTTPException, match="Invalid email or password"),
    ):
        await login(body, mock_db)


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_refresh_success():
    body = RefreshRequest(refresh_token="old_rt")
    mock_db = AsyncMock()
    existing_token = MagicMock()
    existing_token.user_id = "user_1"
    existing_token.family_id = "fam_1"

    with (
        patch("app.routes.auth.hash_token", return_value="hashed"),
        patch(
            "app.routes.auth.get_active_by_hash",
            new_callable=AsyncMock,
            return_value=existing_token,
        ),
        patch("app.routes.auth.revoke", new_callable=AsyncMock),
        patch("app.routes.auth._issue_token_pair", new_callable=AsyncMock) as mock_issue,
    ):
        mock_issue.return_value = MagicMock(access_token="at_new", refresh_token="rt_new")
        result = await refresh(body, mock_db)

    assert result.access_token == "at_new"
    mock_issue.assert_called_once_with(mock_db, "user_1", family_id="fam_1")


@pytest.mark.anyio
async def test_refresh_invalid_token():
    body = RefreshRequest(refresh_token="bad_token")
    mock_db = AsyncMock()

    with (
        patch("app.routes.auth.hash_token", return_value="hashed"),
        patch("app.routes.auth.get_active_by_hash", new_callable=AsyncMock, return_value=None),
        pytest.raises(HTTPException, match="Invalid or expired refresh token"),
    ):
        await refresh(body, mock_db)


# ---------------------------------------------------------------------------
# logout
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_logout_success():
    body = RefreshRequest(refresh_token="some_rt")
    mock_db = AsyncMock()
    existing_token = MagicMock()

    with (
        patch("app.routes.auth.hash_token", return_value="hashed"),
        patch(
            "app.routes.auth.get_active_by_hash",
            new_callable=AsyncMock,
            return_value=existing_token,
        ),
        patch("app.routes.auth.revoke", new_callable=AsyncMock) as mock_revoke,
    ):
        result = await logout(body, mock_db)

    assert result is None  # 204 No Content
    mock_revoke.assert_called_once_with(mock_db, existing_token)


@pytest.mark.anyio
async def test_logout_no_matching_token():
    """Logout with invalid/expired token should succeed silently (no error)."""
    body = RefreshRequest(refresh_token="expired_rt")
    mock_db = AsyncMock()

    with (
        patch("app.routes.auth.hash_token", return_value="hashed"),
        patch("app.routes.auth.get_active_by_hash", new_callable=AsyncMock, return_value=None),
    ):
        result = await logout(body, mock_db)

    assert result is None


# ---------------------------------------------------------------------------
# _issue_token_pair (internal helper)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_issue_token_pair():
    mock_db = AsyncMock()

    with (
        patch("app.routes.auth.create_access_token", return_value="at"),
        patch("app.routes.auth.create_refresh_token", return_value="rt"),
        patch("app.routes.auth.hash_token", return_value="hashed_rt"),
        patch("app.routes.auth.create_refresh_token_crud", new_callable=AsyncMock),
    ):
        result = await _issue_token_pair(mock_db, "user_1")

    assert result.access_token == "at"
    assert result.refresh_token == "rt"


@pytest.mark.anyio
async def test_issue_token_pair_with_family_id():
    """Passing a family_id preserves the token family across rotations."""
    mock_db = AsyncMock()

    with (
        patch("app.routes.auth.create_access_token", return_value="at"),
        patch("app.routes.auth.create_refresh_token", return_value="rt"),
        patch("app.routes.auth.hash_token", return_value="hashed_rt"),
        patch("app.routes.auth.create_refresh_token_crud", new_callable=AsyncMock) as mock_create,
    ):
        result = await _issue_token_pair(mock_db, "user_1", family_id="existing_fam")

    # Verify family_id is passed to the CRUD create call
    call_kwargs = mock_create.call_args
    assert (
        call_kwargs.kwargs.get("family_id") == "existing_fam"
        or call_kwargs[1].get("family_id") == "existing_fam"
    )
