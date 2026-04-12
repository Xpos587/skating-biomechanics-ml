"""RefreshToken CRUD operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from backend.app.models.refresh_token import RefreshToken

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create(
    db: AsyncSession,
    *,
    user_id: str,
    token_hash: str,
    family_id: str,
    expires_at: datetime,
) -> RefreshToken:
    """Create a new refresh token."""
    token = RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        family_id=family_id,
        expires_at=expires_at,
    )
    db.add(token)
    await db.flush()
    await db.refresh(token)
    return token


async def get_by_hash(db: AsyncSession, token_hash: str) -> RefreshToken | None:
    """Get a refresh token by its hash."""
    result = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    return result.scalar_one_or_none()


async def revoke(db: AsyncSession, token: RefreshToken) -> None:
    """Revoke a single refresh token."""
    token.is_revoked = True
    db.add(token)
    await db.flush()


async def revoke_family(db: AsyncSession, family_id: str) -> int:
    """Revoke all tokens in a family (token theft detection)."""
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.family_id == family_id,
            RefreshToken.is_revoked == False,  # noqa: E712
        )
    )
    tokens = result.scalars().all()
    count = 0
    for token in tokens:
        token.is_revoked = True
        db.add(token)
        count += 1
    await db.flush()
    return count


async def get_active_by_hash(db: AsyncSession, token_hash: str) -> RefreshToken | None:
    """Get a non-revoked, non-expired refresh token by hash."""
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.is_revoked == False,  # noqa: E712
            RefreshToken.expires_at > datetime.now(UTC),
        )
    )
    return result.scalar_one_or_none()
