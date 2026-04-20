"""Shared test fixtures for backend tests."""

import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

# Force SKIP_AUTH=false for all tests so auth logic is exercised.
# Must happen before any import of app.config (which caches settings via @lru_cache).
os.environ["APP_SKIP_AUTH"] = "false"

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest_asyncio
from app.config import get_settings
from app.models import Base
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Clear cached settings so APP_SKIP_AUTH=false takes effect
get_settings.cache_clear()


@pytest_asyncio.fixture
async def db_engine():
    """Create a test database engine (SQLite in-memory for unit tests)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()
