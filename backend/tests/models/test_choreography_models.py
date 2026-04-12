"""Tests for choreography ORM models."""

import pytest

from backend.app.models.choreography import ChoreographyProgram, MusicAnalysis


@pytest.fixture
def music_analysis_data():
    return {
        "user_id": "user-123",
        "filename": "test.mp3",
        "audio_url": "music/test.mp3",
        "duration_sec": 180.0,
        "bpm": 120.0,
        "meter": "4/4",
        "status": "completed",
        "structure": [{"type": "verse", "start": 0.0, "end": 30.0}],
        "energy_curve": {"timestamps": [0.0, 1.0], "values": [0.5, 0.8]},
        "downbeats": [0.0, 0.5, 1.0],
        "peaks": [5.0, 15.0, 30.0],
    }


@pytest.fixture
def program_data():
    return {
        "user_id": "user-123",
        "discipline": "mens_singles",
        "segment": "free_skate",
        "season": "2025_26",
        "layout": {"elements": [{"code": "3Lz", "goe": 2}]},
        "total_tes": 45.5,
        "estimated_goe": 5.0,
        "estimated_pcs": 35.0,
        "estimated_total": 85.5,
        "is_valid": True,
        "validation_errors": [],
        "validation_warnings": [],
    }


async def test_create_music_analysis(db_session, music_analysis_data):
    music = MusicAnalysis(**music_analysis_data)
    db_session.add(music)
    await db_session.flush()
    await db_session.refresh(music)

    assert music.id is not None
    assert len(music.id) == 36  # UUID
    assert music.filename == "test.mp3"
    assert music.bpm == 120.0
    assert music.status == "completed"
    assert music.structure == [{"type": "verse", "start": 0.0, "end": 30.0}]
    assert music.peaks == [5.0, 15.0, 30.0]


async def test_create_program(db_session, program_data):
    program = ChoreographyProgram(**program_data)
    db_session.add(program)
    await db_session.flush()
    await db_session.refresh(program)

    assert program.id is not None
    assert program.discipline == "mens_singles"
    assert program.segment == "free_skate"
    assert program.is_valid is True
    assert program.layout == {"elements": [{"code": "3Lz", "goe": 2}]}
