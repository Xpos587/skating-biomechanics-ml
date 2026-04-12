"""Relationship CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from backend.app.models.relationship import Relationship

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create(
    db: AsyncSession,
    *,
    coach_id: str,
    skater_id: str,
    initiated_by: str,
) -> Relationship:
    rel = Relationship(coach_id=coach_id, skater_id=skater_id, initiated_by=initiated_by)
    db.add(rel)
    await db.flush()
    await db.refresh(rel)
    return rel


async def get_by_id(db: AsyncSession, rel_id: str) -> Relationship | None:
    result = await db.execute(select(Relationship).where(Relationship.id == rel_id))
    return result.scalar_one_or_none()


async def get_active(
    db: AsyncSession,
    coach_id: str,
    skater_id: str,
) -> Relationship | None:
    result = await db.execute(
        select(Relationship).where(
            Relationship.coach_id == coach_id,
            Relationship.skater_id == skater_id,
            Relationship.status != "ended",
        )
    )
    return result.scalar_one_or_none()


async def list_for_user(db: AsyncSession, user_id: str) -> list[Relationship]:
    """List all relationships where user is coach or skater."""
    result = await db.execute(
        select(Relationship)
        .where(
            (Relationship.coach_id == user_id) | (Relationship.skater_id == user_id),
        )
        .order_by(Relationship.created_at.desc())
    )
    return list(result.scalars().all())


async def list_pending_for_skater(db: AsyncSession, skater_id: str) -> list[Relationship]:
    result = await db.execute(
        select(Relationship)
        .where(
            Relationship.skater_id == skater_id,
            Relationship.status == "invited",
        )
        .order_by(Relationship.created_at.desc())
    )
    return list(result.scalars().all())


async def list_active_students(db: AsyncSession, coach_id: str) -> list[Relationship]:
    result = await db.execute(
        select(Relationship)
        .where(
            Relationship.coach_id == coach_id,
            Relationship.status == "active",
        )
        .order_by(Relationship.created_at.desc())
    )
    return list(result.scalars().all())


async def list_active_coaches(db: AsyncSession, skater_id: str) -> list[Relationship]:
    result = await db.execute(
        select(Relationship)
        .where(
            Relationship.skater_id == skater_id,
            Relationship.status == "active",
        )
        .order_by(Relationship.created_at.desc())
    )
    return list(result.scalars().all())


async def is_coach_for_student(db: AsyncSession, coach_id: str, skater_id: str) -> bool:
    result = await db.execute(
        select(Relationship).where(
            Relationship.coach_id == coach_id,
            Relationship.skater_id == skater_id,
            Relationship.status == "active",
        )
    )
    return result.scalar_one_or_none() is not None
