"""Async database engine and session factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.config import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

settings = get_settings()

engine = create_async_engine(
    settings.database.url,
    pool_size=10,
    max_overflow=20,
    echo=settings.app.log_level == "DEBUG",
)

async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
