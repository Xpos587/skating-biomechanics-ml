"""Tests for POST /choreography/music/upload route."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

# Mock aiobotocore before importing
_mock_aiobotocore = MagicMock()
_mock_aiobotocore_session = MagicMock()
sys.modules["aiobotocore"] = _mock_aiobotocore
sys.modules["aiobotocore.session"] = _mock_aiobotocore_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user():
    u = MagicMock()
    u.id = "user_123"
    return u


@pytest.fixture
def mock_db():
    return AsyncMock()


def _make_mock_user(user_id: str) -> MagicMock:
    u = MagicMock()
    u.id = user_id
    return u


def _make_owned_entity(user_id: str = "user_123", **extra) -> MagicMock:
    entity = MagicMock()
    entity.user_id = user_id
    for k, v in extra.items():
        setattr(entity, k, v)
    return entity


# ---------------------------------------------------------------------------
# upload_music
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_file():
    f = MagicMock()
    f.filename = "test.mp3"
    f.read = AsyncMock(return_value=b"fake audio content")
    return f


@pytest.fixture
def mock_request():
    req = MagicMock()
    req.app.state.arq_pool = AsyncMock()
    return req


@pytest.fixture
def mock_tmp():
    tmp = MagicMock()
    tmp.name = "/tmp/test.mp3"
    tmp.write = MagicMock()
    tmp.__enter__ = MagicMock(return_value=tmp)
    tmp.__exit__ = MagicMock(return_value=False)
    return tmp


@pytest.mark.asyncio
async def test_upload_music_enqueues_job(mock_user, mock_db, mock_file, mock_request, mock_tmp):
    """Test that upload_music enqueues analyze_music_task and returns immediately."""
    from app.routes.choreography import upload_music
    from app.schemas import UploadMusicResponse

    mock_music = MagicMock()
    mock_music.id = "music_456"
    mock_music.filename = "test.mp3"

    with (
        patch("app.routes.choreography.create_music_analysis") as mock_create,
        patch("app.routes.choreography.upload_file"),
        patch("app.routes.choreography.tempfile.NamedTemporaryFile", return_value=mock_tmp),
    ):
        mock_create.return_value = mock_music

        response = await upload_music(mock_request, mock_user, mock_db, mock_file)

        assert isinstance(response, UploadMusicResponse)
        assert response.music_id == "music_456"
        assert response.filename == "test.mp3"

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["status"] == "pending"
        assert call_kwargs["user_id"] == "user_123"
        assert call_kwargs["filename"] == "test.mp3"

        mock_request.app.state.arq_pool.enqueue_job.assert_called_once_with(
            "analyze_music_task",
            music_id="music_456",
            r2_key="music/user_123/music_456.mp3",
            _queue_name="skating:queue:fast",
        )


@pytest.mark.asyncio
async def test_upload_music_handles_upload_failure(
    mock_user, mock_db, mock_file, mock_request, mock_tmp
):
    """Test that upload_music sets status to failed on upload error."""
    from app.routes.choreography import upload_music
    from fastapi import HTTPException

    mock_music = MagicMock()
    mock_music.id = "music_456"

    with (
        patch("app.routes.choreography.create_music_analysis", return_value=mock_music),
        patch("app.routes.choreography.upload_file", side_effect=OSError("Upload failed")),
        patch("app.routes.choreography.update_music_analysis") as mock_update,
        patch("app.routes.choreography.tempfile.NamedTemporaryFile", return_value=mock_tmp),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await upload_music(mock_request, mock_user, mock_db, mock_file)

        assert exc_info.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Upload failed" in exc_info.value.detail
        mock_update.assert_called_once_with(mock_db, mock_music, status="failed")


# ---------------------------------------------------------------------------
# _program_to_response
# ---------------------------------------------------------------------------


def test_program_to_response():
    from app.routes.choreography import _program_to_response
    from app.schemas import ChoreographyProgramResponse

    mock_program = MagicMock()
    mock_program.id = "prog_1"
    mock_program.user_id = "user_123"
    mock_program.music_analysis_id = None
    mock_program.title = "My Program"
    mock_program.discipline = "mens_singles"
    mock_program.segment = "free_skate"
    mock_program.season = "2025-2026"
    mock_program.layout = {"elements": []}
    mock_program.total_tes = 50.0
    mock_program.estimated_goe = 5.0
    mock_program.estimated_pcs = 80.0
    mock_program.estimated_total = 135.0
    mock_program.is_valid = True
    mock_program.validation_errors = []
    mock_program.validation_warnings = []

    result = _program_to_response(mock_program)
    assert isinstance(result, ChoreographyProgramResponse)
    assert result.id == "prog_1"
    assert result.title == "My Program"
    assert result.total_tes == 50.0


# ---------------------------------------------------------------------------
# get_music_analysis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_music_analysis_not_found(mock_user, mock_db):
    from app.routes.choreography import get_music_analysis
    from fastapi import HTTPException

    with patch("app.routes.choreography.get_music_analysis_by_id", return_value=None):
        with pytest.raises(HTTPException) as exc:
            await get_music_analysis("music_999", mock_user, mock_db)
        assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_music_analysis_unauthorized(mock_db):
    from app.routes.choreography import get_music_analysis
    from fastapi import HTTPException

    wrong_user = _make_mock_user("user_456")
    mock_music = _make_owned_entity("user_123")

    with patch("app.routes.choreography.get_music_analysis_by_id", return_value=mock_music):
        with pytest.raises(HTTPException) as exc:
            await get_music_analysis("music_1", wrong_user, mock_db)
        assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_get_music_analysis_success(mock_user, mock_db):
    from app.routes.choreography import get_music_analysis

    mock_music = MagicMock()
    mock_music.id = "music_1"
    mock_music.user_id = "user_123"
    mock_music.filename = "test.mp3"
    mock_music.audio_url = "https://r2.example.com/music.mp3"
    mock_music.duration_sec = 180.0
    mock_music.bpm = 120.0
    mock_music.meter = "4/4"
    mock_music.structure = [{"type": "A", "start": 0, "end": 30}]
    mock_music.energy_curve = {"timestamps": [0.0], "values": [0.5]}
    mock_music.downbeats = [0.0, 0.5]
    mock_music.peaks = [10.0, 20.0]
    mock_music.status = "completed"
    mock_music.created_at = MagicMock(isoformat=lambda: "2026-01-01T00:00:00")
    mock_music.updated_at = MagicMock(isoformat=lambda: "2026-01-01T00:01:00")

    with patch("app.routes.choreography.get_music_analysis_by_id", return_value=mock_music):
        result = await get_music_analysis("music_1", mock_user, mock_db)

    assert result["id"] == "music_1"
    assert result["status"] == "completed"
    assert result["duration_sec"] == 180.0


# ---------------------------------------------------------------------------
# generate_layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_id,entity,status_code",
    [
        ("user_123", None, 404),
        ("user_456", _make_owned_entity(), 403),
    ],
)
@pytest.mark.asyncio
async def test_generate_layout_errors(mock_db, user_id, entity, status_code):
    from app.routes.choreography import generate_layout
    from app.schemas import GenerateRequest
    from fastapi import HTTPException

    user = _make_mock_user(user_id)
    body = GenerateRequest(
        music_id="music_1", discipline="mens_singles", segment="free_skate", inventory={}
    )

    with patch("app.routes.choreography.get_music_analysis_by_id", return_value=entity):
        with pytest.raises(HTTPException) as exc:
            await generate_layout(body, user, mock_db)
        assert exc.value.status_code == status_code


@pytest.mark.asyncio
async def test_generate_layout_success(mock_user, mock_db):
    from app.routes.choreography import generate_layout
    from app.schemas import GenerateRequest, GenerateResponse

    mock_music = MagicMock()
    mock_music.id = "music_1"
    mock_music.user_id = "user_123"
    mock_music.duration_sec = 180.0
    mock_music.peaks = [10.0, 20.0]
    mock_music.structure = [{"type": "A", "start": 0, "end": 30}]

    body = GenerateRequest(
        music_id="music_1",
        discipline="mens_singles",
        segment="free_skate",
        inventory={"3A": 2, "4T": 1},
    )

    layout_dict = {
        "elements": [
            {"code": "3A", "timestamp": 10.0, "goe": 1, "jump_pass_index": 0},
            {"code": "4T", "timestamp": 50.0, "goe": 0, "jump_pass_index": 1},
        ],
        "total_tes": 15.5,
        "back_half_indices": [1],
    }

    with (
        patch("app.routes.choreography.get_music_analysis_by_id", return_value=mock_music),
        patch("app.routes.choreography.extract_features_for_csp", return_value={}),
        patch("app.routes.choreography.solve_layout", return_value=[layout_dict]),
    ):
        result = await generate_layout(body, mock_user, mock_db)

    assert isinstance(result, GenerateResponse)
    assert len(result.layouts) == 1
    assert result.layouts[0].total_tes == 15.5
    assert result.layouts[0].elements[0].code == "3A"
    assert result.layouts[0].elements[0].is_jump_pass is True
    assert result.layouts[0].elements[1].is_back_half is False


@pytest.mark.asyncio
async def test_generate_layout_empty_result(mock_user, mock_db):
    from app.routes.choreography import generate_layout
    from app.schemas import GenerateRequest, GenerateResponse

    mock_music = MagicMock()
    mock_music.id = "music_1"
    mock_music.user_id = "user_123"
    mock_music.duration_sec = 60.0
    mock_music.peaks = []
    mock_music.structure = []

    body = GenerateRequest(
        music_id="music_1",
        discipline="mens_singles",
        segment="short_program",
        inventory={},
    )

    with (
        patch("app.routes.choreography.get_music_analysis_by_id", return_value=mock_music),
        patch("app.routes.choreography.extract_features_for_csp", return_value={}),
        patch("app.routes.choreography.solve_layout", return_value=[]),
    ):
        result = await generate_layout(body, mock_user, mock_db)

    assert isinstance(result, GenerateResponse)
    assert len(result.layouts) == 0


# ---------------------------------------------------------------------------
# validate_choreography
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "is_valid,errors,warnings",
    [
        (True, [], []),
        (False, ["Too many triple axels"], ["Consider variety"]),
    ],
)
@pytest.mark.asyncio
async def test_validate_choreography(is_valid, errors, warnings):
    from app.routes.choreography import validate_choreography
    from app.schemas import ValidateRequest, ValidateResponse

    elements = [{"code": "3A"}] if is_valid else [{"code": "3A"}, {"code": "3A"}, {"code": "3A"}]
    segment = "free_skate" if is_valid else "short_program"

    body = ValidateRequest(discipline="mens_singles", segment=segment, elements=elements)

    mock_result = MagicMock()
    mock_result.is_valid = is_valid
    mock_result.errors = errors
    mock_result.warnings = warnings

    with patch("app.routes.choreography.validate_layout_engine", return_value=mock_result):
        result = await validate_choreography(body)

    assert isinstance(result, ValidateResponse)
    assert result.is_valid is is_valid
    assert result.errors == errors
    assert result.warnings == warnings


# ---------------------------------------------------------------------------
# render_rink_diagram
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_rink_diagram():
    from app.routes.choreography import render_rink_diagram
    from app.schemas import RenderRinkRequest

    body = RenderRinkRequest(elements=[{"code": "3A", "x": 10.0, "y": 5.0}])

    with patch("app.routes.choreography.render_rink", return_value="<svg>...</svg>"):
        result = await render_rink_diagram(body)

    assert result["svg"] == "<svg>...</svg>"


@pytest.mark.asyncio
async def test_render_rink_diagram_custom_size():
    from app.routes.choreography import render_rink_diagram
    from app.schemas import RenderRinkRequest

    body = RenderRinkRequest(
        elements=[],
        width=2000,
        height=1000,
        rink_width=56.0,
        rink_height=26.0,
    )

    with patch(
        "app.routes.choreography.render_rink", return_value="<svg>wide</svg>"
    ) as mock_render:
        result = await render_rink_diagram(body)

    mock_render.assert_called_once_with(
        [], width=2000, height=1000, rink_width=56.0, rink_height=26.0
    )
    assert result["svg"] == "<svg>wide</svg>"


# ---------------------------------------------------------------------------
# list_programs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_programs(mock_user, mock_db):
    from app.routes.choreography import list_programs
    from app.schemas import ProgramListResponse

    mock_program = MagicMock()
    mock_program.id = "prog_1"
    mock_program.user_id = "user_123"
    mock_program.music_analysis_id = None
    mock_program.title = "Test Program"
    mock_program.discipline = "mens_singles"
    mock_program.segment = "free_skate"
    mock_program.season = "2025-2026"
    mock_program.layout = None
    mock_program.total_tes = None
    mock_program.estimated_goe = None
    mock_program.estimated_pcs = None
    mock_program.estimated_total = None
    mock_program.is_valid = None
    mock_program.validation_errors = None
    mock_program.validation_warnings = None

    with (
        patch("app.routes.choreography.list_programs_by_user", return_value=[mock_program]),
        patch("app.routes.choreography.count_programs_by_user", return_value=1),
    ):
        result = await list_programs(mock_user, mock_db, limit=10, offset=0)

    assert isinstance(result, ProgramListResponse)
    assert result.total == 1
    assert len(result.programs) == 1
    assert result.programs[0].id == "prog_1"


@pytest.mark.asyncio
async def test_list_programs_empty(mock_db):
    from app.routes.choreography import list_programs
    from app.schemas import ProgramListResponse

    user = _make_mock_user("user_empty")

    with (
        patch("app.routes.choreography.list_programs_by_user", return_value=[]),
        patch("app.routes.choreography.count_programs_by_user", return_value=0),
    ):
        result = await list_programs(user, mock_db)

    assert isinstance(result, ProgramListResponse)
    assert result.total == 0
    assert result.programs == []


# ---------------------------------------------------------------------------
# create_new_program
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_new_program(mock_user, mock_db):
    from app.routes.choreography import create_new_program
    from app.schemas import SaveProgramRequest

    mock_program = MagicMock()
    mock_program.id = "prog_new"
    mock_program.user_id = "user_123"
    mock_program.music_analysis_id = None
    mock_program.title = "New Program"
    mock_program.discipline = "mens_singles"
    mock_program.segment = "free_skate"
    mock_program.season = "2025-2026"
    mock_program.layout = {"elements": []}
    mock_program.total_tes = 40.0
    mock_program.estimated_goe = 3.0
    mock_program.estimated_pcs = 75.0
    mock_program.estimated_total = 118.0
    mock_program.is_valid = True
    mock_program.validation_errors = []
    mock_program.validation_warnings = []

    body = SaveProgramRequest(
        title="New Program",
        layout={"elements": []},
        total_tes=40.0,
        estimated_goe=3.0,
        estimated_pcs=75.0,
        estimated_total=118.0,
        is_valid=True,
        validation_errors=[],
        validation_warnings=[],
    )

    with patch("app.routes.choreography.create_program", return_value=mock_program):
        result = await create_new_program(body, mock_user, mock_db)

    assert result.id == "prog_new"
    assert result.title == "New Program"


# ---------------------------------------------------------------------------
# get_program / update_existing_program / delete_existing_program / export_program
# Shared not-found + unauthorized pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route_func_name,crud_func_name,call_args",
    [
        ("get_program", "get_program_by_id", ("prog_999",)),
        ("delete_existing_program", "get_program_by_id", ("prog_999",)),
    ],
)
@pytest.mark.parametrize(
    "user_id,entity,status_code",
    [
        ("user_123", None, 404),
        ("user_456", _make_owned_entity(), 403),
    ],
)
@pytest.mark.asyncio
async def test_program_crud_errors(
    mock_db, route_func_name, crud_func_name, call_args, user_id, entity, status_code
):
    from app.routes.choreography import delete_existing_program, get_program
    from fastapi import HTTPException

    func_map = {"get_program": get_program, "delete_existing_program": delete_existing_program}
    func = func_map[route_func_name]
    user = _make_mock_user(user_id)

    with patch(f"app.routes.choreography.{crud_func_name}", return_value=entity):
        with pytest.raises(HTTPException) as exc:
            await func(*call_args, user, mock_db)
        assert exc.value.status_code == status_code


@pytest.mark.parametrize(
    "route_func_name,call_args",
    [
        ("update_existing_program", ("prog_999",)),
        ("export_program", ("prog_999",)),
    ],
)
@pytest.mark.parametrize(
    "user_id,entity,status_code",
    [
        ("user_123", None, 404),
        ("user_456", _make_owned_entity(), 403),
    ],
)
@pytest.mark.asyncio
async def test_program_body_crud_errors(
    mock_db, route_func_name, call_args, user_id, entity, status_code
):
    from app.routes.choreography import export_program, update_existing_program
    from app.schemas import ExportRequest, SaveProgramRequest
    from fastapi import HTTPException

    func_map = {
        "update_existing_program": update_existing_program,
        "export_program": export_program,
    }
    func = func_map[route_func_name]
    user = _make_mock_user(user_id)

    body_map = {
        "update_existing_program": SaveProgramRequest(title="Updated"),
        "export_program": ExportRequest(format="json"),
    }

    with patch("app.routes.choreography.get_program_by_id", return_value=entity):
        with pytest.raises(HTTPException) as exc:
            await func(*call_args, body_map[route_func_name], user, mock_db)
        assert exc.value.status_code == status_code


# ---------------------------------------------------------------------------
# get_program success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_program_success(mock_user, mock_db):
    from app.routes.choreography import get_program

    mock_program = MagicMock()
    mock_program.id = "prog_1"
    mock_program.user_id = "user_123"
    mock_program.music_analysis_id = None
    mock_program.title = "My Program"
    mock_program.discipline = "mens_singles"
    mock_program.segment = "free_skate"
    mock_program.season = "2025-2026"
    mock_program.layout = {"elements": []}
    mock_program.total_tes = 50.0
    mock_program.estimated_goe = 5.0
    mock_program.estimated_pcs = 80.0
    mock_program.estimated_total = 135.0
    mock_program.is_valid = True
    mock_program.validation_errors = []
    mock_program.validation_warnings = []

    with patch("app.routes.choreography.get_program_by_id", return_value=mock_program):
        result = await get_program("prog_1", mock_user, mock_db)

    assert result.id == "prog_1"
    assert result.title == "My Program"


# ---------------------------------------------------------------------------
# update_existing_program success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_existing_program_success(mock_user, mock_db):
    from app.routes.choreography import update_existing_program
    from app.schemas import SaveProgramRequest

    mock_program = MagicMock()
    mock_program.id = "prog_1"
    mock_program.user_id = "user_123"
    mock_program.music_analysis_id = None
    mock_program.title = "Old Title"
    mock_program.discipline = "mens_singles"
    mock_program.segment = "free_skate"
    mock_program.season = "2025-2026"
    mock_program.layout = None
    mock_program.total_tes = None
    mock_program.estimated_goe = None
    mock_program.estimated_pcs = None
    mock_program.estimated_total = None
    mock_program.is_valid = None
    mock_program.validation_errors = None
    mock_program.validation_warnings = None

    body = SaveProgramRequest(title="Updated Title")

    with (
        patch("app.routes.choreography.get_program_by_id", return_value=mock_program),
        patch("app.routes.choreography.update_program", return_value=mock_program) as mock_update,
    ):
        result = await update_existing_program("prog_1", body, mock_user, mock_db)

    mock_update.assert_called_once()
    assert result.id == "prog_1"


# ---------------------------------------------------------------------------
# delete_existing_program success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_existing_program_success(mock_user, mock_db):
    from app.routes.choreography import delete_existing_program

    mock_program = _make_owned_entity("user_123")

    with (
        patch("app.routes.choreography.get_program_by_id", return_value=mock_program),
        patch("app.routes.choreography.delete_program") as mock_delete,
    ):
        await delete_existing_program("prog_1", mock_user, mock_db)

    mock_delete.assert_called_once_with(mock_db, mock_program)


# ---------------------------------------------------------------------------
# export_program success (format variants)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["json", "svg", "pdf"])
@pytest.mark.asyncio
async def test_export_program_format(mock_user, mock_db, fmt):
    from app.routes.choreography import export_program
    from app.schemas import ExportRequest

    layout = (
        {"elements": [{"code": "3A", "x": 10.0, "y": 5.0}]}
        if fmt in ("svg", "pdf")
        else {"elements": [{"code": "3A"}]}
    )
    mock_program = _make_owned_entity(
        "user_123",
        id="prog_1",
        title="My Program",
        discipline="mens_singles",
        segment="free_skate",
        layout=layout,
        total_tes=50.0,
        estimated_total=135.0,
    )

    body = ExportRequest(format=fmt)

    svg_return = {"json": None, "svg": "<svg>rink</svg>", "pdf": "<svg>pdf-rink</svg>"}[fmt]
    render_patch = (
        patch("app.routes.choreography.render_rink", return_value=svg_return)
        if fmt != "json"
        else None
    )

    ctx = [patch("app.routes.choreography.get_program_by_id", return_value=mock_program)]
    if render_patch is not None:
        ctx.append(render_patch)

    if len(ctx) == 1:
        with ctx[0]:
            result = await export_program("prog_1", body, mock_user, mock_db)
    else:
        with ctx[0], ctx[1]:
            result = await export_program("prog_1", body, mock_user, mock_db)

    assert result["format"] == fmt
    if fmt == "json":
        assert result["data"]["id"] == "prog_1"
        assert result["data"]["title"] == "My Program"
    else:
        assert result["svg"] is not None
    if fmt == "pdf":
        assert "note" in result
        assert "headless browser" in result["note"]


@pytest.mark.asyncio
async def test_export_program_svg_no_layout(mock_user, mock_db):
    from app.routes.choreography import export_program
    from app.schemas import ExportRequest

    mock_program = _make_owned_entity(
        "user_123",
        id="prog_1",
        title="Empty Layout",
        discipline="mens_singles",
        segment="free_skate",
        layout=None,
        total_tes=0.0,
        estimated_total=0.0,
    )

    body = ExportRequest(format="svg")

    with (
        patch("app.routes.choreography.get_program_by_id", return_value=mock_program),
        patch(
            "app.routes.choreography.render_rink", return_value="<svg>empty</svg>"
        ) as mock_render,
    ):
        result = await export_program("prog_1", body, mock_user, mock_db)

    mock_render.assert_called_once_with([])
    assert result["format"] == "svg"
