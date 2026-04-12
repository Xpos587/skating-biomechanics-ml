"""User CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from backend.app.models.user import User

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_by_id(db: AsyncSession, user_id: str) -> User | None:
    """Get a user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_by_email(db: AsyncSession, email: str) -> User | None:
    """Get a user by email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create(db: AsyncSession, *, email: str, hashed_password: str, **kwargs) -> User:
    """Create a new user."""
    user = User(email=email, hashed_password=hashed_password, **kwargs)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def update(db: AsyncSession, user: User, **kwargs) -> User:
    """Update user fields."""
    for key, value in kwargs.items():
        if value is not None:
            setattr(user, key, value)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user
