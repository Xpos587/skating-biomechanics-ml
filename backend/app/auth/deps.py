"""FastAPI dependencies for authentication."""

from __future__ import annotations

import logging
from typing import Annotated

import jwt as pyjwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

settings = get_settings()
_log = logging.getLogger(__name__)


async def _get_dev_user(db: AsyncSession) -> User:
    """Return the first active user from DB for dev mode (SKIP_AUTH)."""
    result = await db.execute(
        select(User).where(User.is_active.is_(True)).order_by(User.created_at).limit(1)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No active users in database. Create one first.",
        )
    return user


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Validate JWT and return the current user.

    When APP_SKIP_AUTH=true, returns the first active user from DB.
    """
    if settings.app.skip_auth:
        return await _get_dev_user(db)

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = pyjwt.decode(
            token, settings.jwt.secret_key.get_secret_value(), algorithms=["HS256"]
        )
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except (pyjwt.InvalidTokenError, ValueError):
        raise credentials_exception from None

    from app.crud.user import get_by_id

    user = await get_by_id(db, user_id)
    if user is None or not user.is_active:
        raise credentials_exception
    return user


# Type aliases for clean injection
DbDep = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
