"""Tests for choreography CRUD operations."""

from app.crud.choreography import (
    count_programs_by_user,
    create_music_analysis,
    create_program,
    delete_program,
    find_music_by_fingerprint,
    get_music_analysis_by_id,
    get_program_by_id,
    list_programs_by_user,
    update_music_analysis,
    update_program,
)
from app.models.choreography import MusicAnalysis


async def test_create_and_get_music(db_session):
    music = await create_music_analysis(
        db_session,
        user_id="user-123",
        filename="test.mp3",
        audio_url="music/test.mp3",
        duration_sec=180.0,
    )
    assert music.id is not None
    assert music.status == "pending"

    fetched = await get_music_analysis_by_id(db_session, music.id)
    assert fetched is not None
    assert fetched.filename == "test.mp3"


async def test_create_and_get_program(db_session):
    music = await create_music_analysis(
        db_session,
        user_id="user-123",
        filename="test.mp3",
        audio_url="music/test.mp3",
        duration_sec=180.0,
    )
    program = await create_program(
        db_session,
        user_id="user-123",
        music_analysis_id=music.id,
        discipline="mens_singles",
        segment="free_skate",
    )
    assert program.id is not None

    fetched = await get_program_by_id(db_session, program.id)
    assert fetched is not None
    assert fetched.discipline == "mens_singles"


async def test_list_programs_by_user(db_session):
    await create_program(
        db_session, user_id="user-123", discipline="mens_singles", segment="free_skate"
    )
    await create_program(
        db_session, user_id="user-123", discipline="mens_singles", segment="short_program"
    )
    await create_program(
        db_session, user_id="other-user", discipline="womens_singles", segment="free_skate"
    )

    programs = await list_programs_by_user(db_session, "user-123")
    assert len(programs) == 2


async def test_update_program(db_session):
    program = await create_program(
        db_session,
        user_id="user-123",
        discipline="mens_singles",
        segment="free_skate",
    )
    updated = await update_program(
        db_session,
        program,
        title="My Program",
        total_tes=45.5,
        is_valid=True,
    )
    assert updated.title == "My Program"
    assert updated.total_tes == 45.5


async def test_find_music_by_fingerprint(db_session):
    music = MusicAnalysis(
        user_id="user-1",
        filename="song.mp3",
        audio_url="music/user-1/song.mp3",
        duration_sec=180.0,
        bpm=120.0,
        status="completed",
        fingerprint="abcdef1234567890abcdef1234567890",
    )
    db_session.add(music)
    await db_session.flush()
    await db_session.refresh(music)

    found = await find_music_by_fingerprint(db_session, "abcdef1234567890abcdef1234567890")
    assert found is not None
    assert found.id == music.id

    not_found = await find_music_by_fingerprint(db_session, "00000000000000000000000000000000")
    assert not_found is None


async def test_update_music_analysis(db_session):
    music = await create_music_analysis(
        db_session,
        user_id="user-123",
        filename="test.mp3",
        audio_url="music/test.mp3",
        duration_sec=180.0,
    )
    updated = await update_music_analysis(
        db_session,
        music,
        bpm=120.0,
        meter="4/4",
        status="completed",
    )
    assert updated.bpm == 120.0
    assert updated.meter == "4/4"
    assert updated.status == "completed"


async def test_update_music_analysis_ignores_none(db_session):
    music = await create_music_analysis(
        db_session,
        user_id="user-123",
        filename="test.mp3",
        audio_url="music/test.mp3",
        duration_sec=180.0,
        bpm=100.0,
    )
    updated = await update_music_analysis(
        db_session,
        music,
        bpm=None,
        meter="3/4",
    )
    assert updated.bpm == 100.0  # unchanged
    assert updated.meter == "3/4"


async def test_count_programs_by_user(db_session):
    await create_program(
        db_session, user_id="user-123", discipline="mens_singles", segment="free_skate"
    )
    await create_program(
        db_session, user_id="user-123", discipline="mens_singles", segment="short_program"
    )
    await create_program(
        db_session, user_id="other-user", discipline="womens_singles", segment="free_skate"
    )

    count = await count_programs_by_user(db_session, "user-123")
    assert count == 2

    count_other = await count_programs_by_user(db_session, "other-user")
    assert count_other == 1


async def test_delete_program(db_session):
    program = await create_program(
        db_session,
        user_id="user-123",
        discipline="mens_singles",
        segment="free_skate",
    )
    program_id = program.id

    await delete_program(db_session, program)

    fetched = await get_program_by_id(db_session, program_id)
    assert fetched is None
