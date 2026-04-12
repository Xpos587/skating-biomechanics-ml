"""Tests for auth security utilities."""

import time

import pytest

from backend.app.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password():
    """Test password hashing and verification."""
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert len(hashed) > 20
    assert verify_password("secret123", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_hash_different_for_same_password():
    """Test that bcrypt produces different hashes for the same password (salt)."""
    h1 = hash_password("same")
    h2 = hash_password("same")
    assert h1 != h2


def test_verify_password_invalid_hash():
    """Test verification with garbage hash returns False."""
    assert verify_password("test", "not_a_real_hash") is False


def test_create_access_token():
    """Test JWT access token creation and decoding."""
    token = create_access_token(user_id="user-123")
    assert isinstance(token, str)
    assert len(token) > 20

    # Verify it's a valid JWT by checking structure (header.payload.signature)
    parts = token.split(".")
    assert len(parts) == 3


def test_create_access_token_custom_expiry():
    """Test access token with custom expiry."""
    token = create_access_token(user_id="user-123", expires_delta_seconds=1)
    # Token should still be valid immediately
    parts = token.split(".")
    assert len(parts) == 3

    # After 2 seconds it should be expired
    time.sleep(2)
    import jwt as pyjwt

    from backend.app.config import get_settings

    settings = get_settings()
    with pytest.raises(pyjwt.ExpiredSignatureError):
        pyjwt.decode(token, settings.jwt.secret_key.get_secret_value(), algorithms=["HS256"])


def test_create_refresh_token():
    """Test refresh token is a random opaque string."""
    token = create_refresh_token()
    assert isinstance(token, str)
    assert len(token) == 64  # hex-encoded 32 bytes


def test_create_refresh_token_unique():
    """Test that two refresh tokens are unique."""
    t1 = create_refresh_token()
    t2 = create_refresh_token()
    assert t1 != t2


def test_hash_token():
    """Test token hashing for DB storage."""
    from backend.app.auth.security import hash_token

    token = create_refresh_token()
    hashed = hash_token(token)
    assert isinstance(hashed, str)
    assert len(hashed) == 64
    # Same token produces same hash
    assert hash_token(token) == hashed
