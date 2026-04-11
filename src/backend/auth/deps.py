"""FastAPI dependencies for authentication."""

from __future__ import annotations

from typing import Annotated

import jwt as pyjwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.database import get_db
from src.backend.models.user import User
from src.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

settings = get_settings()


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Validate JWT and return the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = pyjwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except (pyjwt.InvalidTokenError, ValueError):
        raise credentials_exception from None

    from src.backend.crud.user import get_by_id

    user = await get_by_id(db, user_id)
    if user is None or not user.is_active:
        raise credentials_exception
    return user


# Type aliases for clean injection
DbDep = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
