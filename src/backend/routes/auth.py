"""Auth API routes: register, login, refresh, logout."""

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, status

from src.backend.auth.deps import DbDep
from src.backend.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)
from src.backend.crud.refresh_token import create as create_refresh_token_crud
from src.backend.crud.refresh_token import get_active_by_hash, revoke
from src.backend.crud.user import create as create_user
from src.backend.crud.user import get_by_email
from src.backend.schemas_auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)
from src.config import get_settings

router = APIRouter(tags=["auth"])
settings = get_settings()


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

    access_token = create_access_token(user_id=user.id)
    refresh_token_str = create_refresh_token()
    family_id = str(uuid.uuid4())

    await create_refresh_token_crud(
        db,
        user_id=user.id,
        token_hash=hash_token(refresh_token_str),
        family_id=family_id,
        expires_at=datetime.now(UTC) + timedelta(days=settings.jwt_refresh_token_expire_days),
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token_str)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DbDep):
    """Authenticate and return tokens."""
    user = await get_by_email(db, body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    access_token = create_access_token(user_id=user.id)
    refresh_token_str = create_refresh_token()
    family_id = str(uuid.uuid4())

    await create_refresh_token_crud(
        db,
        user_id=user.id,
        token_hash=hash_token(refresh_token_str),
        family_id=family_id,
        expires_at=datetime.now(UTC) + timedelta(days=settings.jwt_refresh_token_expire_days),
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token_str)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: DbDep):
    """Rotate refresh token and issue new token pair."""
    token_hash = hash_token(body.refresh_token)
    existing = await get_active_by_hash(db, token_hash)

    if not existing:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        )

    # Revoke the old token (single-use)
    await revoke(db, existing)

    # Issue new pair in the same family
    access_token = create_access_token(user_id=existing.user_id)
    new_refresh_str = create_refresh_token()

    await create_refresh_token_crud(
        db,
        user_id=existing.user_id,
        token_hash=hash_token(new_refresh_str),
        family_id=existing.family_id,
        expires_at=datetime.now(UTC) + timedelta(days=settings.jwt_refresh_token_expire_days),
    )

    return TokenResponse(access_token=access_token, refresh_token=new_refresh_str)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, db: DbDep):
    """Revoke a refresh token (client should also discard access token)."""
    token_hash = hash_token(body.refresh_token)
    existing = await get_active_by_hash(db, token_hash)
    if existing:
        await revoke(db, existing)
