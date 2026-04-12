"""Auth API routes: register, login, refresh, logout."""

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, status

from backend.app.auth.deps import DbDep
from backend.app.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)
from backend.app.config import get_settings
from backend.app.crud.refresh_token import create as create_refresh_token_crud
from backend.app.crud.refresh_token import get_active_by_hash, revoke
from backend.app.crud.user import create as create_user
from backend.app.crud.user import get_by_email
from backend.app.schemas import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)

router = APIRouter(tags=["auth"])
settings = get_settings()


async def _issue_token_pair(db: DbDep, user_id: str, family_id: str | None = None) -> TokenResponse:
    """Create and persist a new access + refresh token pair."""
    access = create_access_token(user_id=user_id)
    refresh_str = create_refresh_token()
    fam = family_id or str(uuid.uuid4())
    await create_refresh_token_crud(
        db,
        user_id=user_id,
        token_hash=hash_token(refresh_str),
        family_id=fam,
        expires_at=datetime.now(UTC) + timedelta(days=settings.jwt.refresh_token_expire_days),
    )
    return TokenResponse(access_token=access, refresh_token=refresh_str)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: DbDep):
    """Register a new user."""
    existing = await get_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = await create_user(
        db,
        email=body.email,
        hashed_password=hash_password(body.password),
        display_name=body.display_name,
    )
    return await _issue_token_pair(db, user.id)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DbDep):
    """Authenticate and return tokens."""
    user = await get_by_email(db, body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )
    return await _issue_token_pair(db, user.id)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: DbDep):
    """Rotate refresh token and issue new token pair."""
    token_hash = hash_token(body.refresh_token)
    existing = await get_active_by_hash(db, token_hash)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        )
    await revoke(db, existing)
    return await _issue_token_pair(db, existing.user_id, family_id=existing.family_id)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, db: DbDep):
    """Revoke a refresh token (client should also discard access token)."""
    token_hash = hash_token(body.refresh_token)
    existing = await get_active_by_hash(db, token_hash)
    if existing:
        await revoke(db, existing)
