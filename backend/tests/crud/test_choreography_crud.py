"""Tests for choreography CRUD operations."""

from backend.app.crud.choreography import (
    create_music_analysis,
    create_program,
    get_music_analysis_by_id,
    get_program_by_id,
    list_programs_by_user,
    update_program,
)


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
