"""CRUD operations for choreography models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import desc, select

from backend.app.models.choreography import ChoreographyProgram, MusicAnalysis

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# --- Music Analysis ---


async def create_music_analysis(
    db: AsyncSession,
    *,
    user_id: str,
    filename: str,
    audio_url: str,
    duration_sec: float,
    **kwargs,
) -> MusicAnalysis:
    music = MusicAnalysis(
        user_id=user_id,
        filename=filename,
        audio_url=audio_url,
        duration_sec=duration_sec,
        **kwargs,
    )
    db.add(music)
    await db.flush()
    await db.refresh(music)
    return music


async def get_music_analysis_by_id(db: AsyncSession, music_id: str) -> MusicAnalysis | None:
    result = await db.execute(
        select(MusicAnalysis).where(MusicAnalysis.id == music_id),
    )
    return result.scalar_one_or_none()


# --- Choreography Program ---


async def create_program(
    db: AsyncSession,
    *,
    user_id: str,
    discipline: str,
    segment: str,
    **kwargs,
) -> ChoreographyProgram:
    program = ChoreographyProgram(
        user_id=user_id,
        discipline=discipline,
        segment=segment,
        **kwargs,
    )
    db.add(program)
    await db.flush()
    await db.refresh(program)
    return program


async def get_program_by_id(db: AsyncSession, program_id: str) -> ChoreographyProgram | None:
    result = await db.execute(
        select(ChoreographyProgram).where(ChoreographyProgram.id == program_id),
    )
    return result.scalar_one_or_none()


async def list_programs_by_user(
    db: AsyncSession,
    user_id: str,
    *,
    limit: int = 20,
    offset: int = 0,
) -> list[ChoreographyProgram]:
    query = (
        select(ChoreographyProgram)
        .where(ChoreographyProgram.user_id == user_id)
        .order_by(desc(ChoreographyProgram.created_at))
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    return list(result.scalars().all())


async def update_program(
    db: AsyncSession,
    program: ChoreographyProgram,
    **kwargs,
) -> ChoreographyProgram:
    for key, value in kwargs.items():
        if value is not None:
            setattr(program, key, value)
    db.add(program)
    await db.flush()
    await db.refresh(program)
    return program


async def delete_program(db: AsyncSession, program: ChoreographyProgram) -> None:
    await db.delete(program)
    await db.flush()
