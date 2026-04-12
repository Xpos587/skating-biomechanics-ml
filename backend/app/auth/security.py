"""JWT creation, password hashing, and token utilities."""

from __future__ import annotations

import hashlib
import secrets

import jwt as pyjwt
from passlib.context import CryptContext

from backend.app.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def create_access_token(user_id: str, expires_delta_seconds: int | None = None) -> str:
    """Create a short-lived JWT access token."""
    settings = get_settings()
    if expires_delta_seconds is None:
        expires_delta_seconds = settings.jwt.access_token_expire_minutes * 60

    payload = {
        "sub": user_id,
        "type": "access",
        "exp": int(__import__("time").time() + expires_delta_seconds),
    }
    return pyjwt.encode(payload, settings.jwt.secret_key.get_secret_value(), algorithm="HS256")


def create_refresh_token() -> str:
    """Create a random opaque refresh token (hex-encoded 32 bytes)."""
    return secrets.token_hex(32)


def hash_token(token: str) -> str:
    """Hash a refresh token for DB storage (SHA-256)."""
    return hashlib.sha256(token.encode()).hexdigest()
