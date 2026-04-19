# src/backend/routes/choreography.py
"""Choreography planner API routes."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from app.auth.deps import CurrentUser, DbDep  # noqa: TC001 — runtime FastAPI Depends
from app.crud.choreography import (
    create_music_analysis,
    create_program,
    delete_program,
    get_music_analysis_by_id,
    get_program_by_id,
    list_programs_by_user,
    update_music_analysis,
    update_program,
)
from app.schemas import (
    ChoreographyProgramResponse,
    ExportRequest,
    GenerateRequest,
    GenerateResponse,
    Layout,
    LayoutElement,
    ProgramListResponse,
    RenderRinkRequest,
    SaveProgramRequest,
    UploadMusicResponse,
    ValidateRequest,
    ValidateResponse,
)
from app.services.choreography.csp_solver import solve_layout
from app.services.choreography.music_analyzer import extract_features_for_csp
from app.services.choreography.rink_renderer import render_rink
from app.services.choreography.rules_engine import validate_layout as validate_layout_engine
from app.storage import upload_file

router = APIRouter(tags=["choreography"])


def _program_to_response(program) -> ChoreographyProgramResponse:
    """Convert ORM ChoreographyProgram to response schema."""
    return ChoreographyProgramResponse.model_validate(program)


# ---------------------------------------------------------------------------
# Music upload & analysis
# ---------------------------------------------------------------------------


def _get_duration(path: str, suffix: str) -> float:
    """Get audio duration from file header (no ML deps needed)."""
    import wave

    try:
        if suffix == ".wav":
            with wave.open(path, "rb") as wf:
                return round(wf.getnframes() / float(wf.getframerate()), 1)
    except Exception:
        pass
    return 180.0


@router.post(
    "/choreography/music/upload",
    response_model=UploadMusicResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_music(
    user: CurrentUser,
    db: DbDep,
    file: UploadFile,
):
    """Upload an audio file, analyze it, and store results."""
    import asyncio
    import logging

    logger = logging.getLogger(__name__)

    suffix = (
        f".{file.filename.rsplit('.', 1)[-1]}" if file.filename and "." in file.filename else ".mp3"
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Create record as "analyzing"
    music = await create_music_analysis(
        db,
        user_id=user.id,
        filename=file.filename or "unknown",
        audio_url="",
        duration_sec=0,
        status="analyzing",
    )

    try:
        # Run music analysis in thread pool (blocking librosa/madmom calls)
        # Graceful fallback: if ML deps aren't installed, use basic duration
        duration_sec = 0.0
        bpm = None
        energy_curve = None
        peaks = None
        structure = None

        r2_key = f"music/{user.id}/{music.id}{suffix}"

        async def _analyze():
            nonlocal duration_sec, bpm, energy_curve, peaks, structure
            try:
                from app.services.choreography.music_analyzer import analyze_music_sync

                logger.info("Running music analysis on %s", tmp_path)
                result = await asyncio.to_thread(analyze_music_sync, tmp_path)
                duration_sec = result["duration_sec"]
                bpm = result["bpm"]
                energy_curve = result["energy_curve"]
                peaks = result["peaks"]
                structure = result.get("structure") or []
                logger.info("Analysis complete: bpm=%.1f, duration=%.1f", bpm, duration_sec)
            except ImportError:
                logger.info("librosa not available, using basic duration estimation")
                duration_sec = await asyncio.to_thread(_get_duration, tmp_path, suffix)

        async def _upload():
            logger.info("Uploading to R2: %s", r2_key)
            await asyncio.to_thread(upload_file, tmp_path, r2_key)
            logger.info("R2 upload complete")

        await asyncio.gather(_analyze(), _upload())

        await update_music_analysis(
            db,
            music,
            audio_url=f"/files/{r2_key}",
            duration_sec=duration_sec,
            bpm=bpm,
            energy_curve=energy_curve,
            peaks=peaks,
            structure=structure,
            status="completed",
        )
    except Exception as e:
        logger.exception("Music analysis failed")
        await update_music_analysis(db, music, status="failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Music analysis failed: {type(e).__name__}: {e}",
        ) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadMusicResponse(music_id=music.id, filename=music.filename)


@router.get(
    "/choreography/music/{music_id}/analysis",
)
async def get_music_analysis(music_id: str, user: CurrentUser, db: DbDep):
    """Get music analysis result."""
    music = await get_music_analysis_by_id(db, music_id)
    if not music:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Music analysis not found"
        )
    if music.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return {
        "id": music.id,
        "user_id": music.user_id,
        "filename": music.filename,
        "audio_url": music.audio_url,
        "duration_sec": music.duration_sec,
        "bpm": music.bpm,
        "meter": music.meter,
        "structure": music.structure,
        "energy_curve": music.energy_curve,
        "downbeats": music.downbeats,
        "peaks": music.peaks,
        "status": music.status,
        "created_at": music.created_at.isoformat() if music.created_at else None,
        "updated_at": music.updated_at.isoformat() if music.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Layout generation & validation
# ---------------------------------------------------------------------------


@router.post(
    "/choreography/generate",
    response_model=GenerateResponse,
)
async def generate_layout(body: GenerateRequest, user: CurrentUser, db: DbDep):
    """Generate choreography layouts via CSP solver."""
    music = await get_music_analysis_by_id(db, body.music_id)
    if not music:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Music analysis not found"
        )
    if music.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    analysis = {
        "duration_sec": music.duration_sec,
        "peaks": music.peaks or [],
        "structure": music.structure or [],
    }
    music_features = extract_features_for_csp(analysis)
    layouts = await asyncio.to_thread(
        solve_layout,
        inventory=body.inventory,
        music_features=music_features,
        discipline=body.discipline,
        segment=body.segment,
    )

    response_layouts = []
    for layout in layouts:
        elements = [
            LayoutElement(
                code=e["code"],
                goe=e.get("goe", 0),
                timestamp=e.get("timestamp", 0.0),
                position=e.get("position"),
                is_back_half=False,
                is_jump_pass="jump_pass_index" in e,
                jump_pass_index=e.get("jump_pass_index"),
            )
            for e in layout["elements"]
        ]
        response_layouts.append(
            Layout(
                elements=elements,
                total_tes=layout["total_tes"],
                back_half_indices=layout["back_half_indices"],
            )
        )
    return GenerateResponse(layouts=response_layouts)


@router.post(
    "/choreography/validate",
    response_model=ValidateResponse,
)
async def validate_choreography(body: ValidateRequest):
    """Validate a layout against ISU rules."""
    layout = {
        "discipline": body.discipline,
        "segment": body.segment,
        "elements": body.elements,
    }
    result = validate_layout_engine(layout)
    return ValidateResponse(
        is_valid=result.is_valid,
        errors=result.errors,
        warnings=result.warnings,
    )


# ---------------------------------------------------------------------------
# Rink rendering
# ---------------------------------------------------------------------------


@router.post("/choreography/render-rink")
async def render_rink_diagram(body: RenderRinkRequest):
    """Render an SVG rink diagram with element markers."""
    svg = render_rink(
        body.elements,
        width=body.width,
        height=body.height,
    )
    return {"svg": svg}


# ---------------------------------------------------------------------------
# Program CRUD
# ---------------------------------------------------------------------------


@router.get("/choreography/programs", response_model=ProgramListResponse)
async def list_programs(
    user: CurrentUser,
    db: DbDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List user's choreography programs."""
    programs = await list_programs_by_user(db, user.id, limit=limit, offset=offset)
    return ProgramListResponse(
        programs=[_program_to_response(p) for p in programs],
        total=len(programs),
    )


@router.post(
    "/choreography/programs",
    response_model=ChoreographyProgramResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_new_program(body: SaveProgramRequest, user: CurrentUser, db: DbDep):
    """Create a new choreography program."""
    program = await create_program(
        db,
        user_id=user.id,
        discipline="mens_singles",
        segment="free_skate",
        title=body.title,
        layout=body.layout,
        total_tes=body.total_tes,
        estimated_goe=body.estimated_goe,
        estimated_pcs=body.estimated_pcs,
        estimated_total=body.estimated_total,
        is_valid=body.is_valid,
        validation_errors=body.validation_errors,
        validation_warnings=body.validation_warnings,
    )
    return _program_to_response(program)


@router.get("/choreography/programs/{program_id}", response_model=ChoreographyProgramResponse)
async def get_program(program_id: str, user: CurrentUser, db: DbDep):
    """Get a choreography program."""
    program = await get_program_by_id(db, program_id)
    if not program:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
    if program.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return _program_to_response(program)


@router.put("/choreography/programs/{program_id}", response_model=ChoreographyProgramResponse)
async def update_existing_program(
    program_id: str,
    body: SaveProgramRequest,
    user: CurrentUser,
    db: DbDep,
):
    """Update a choreography program."""
    program = await get_program_by_id(db, program_id)
    if not program:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
    if program.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    program = await update_program(
        db,
        program,
        title=body.title,
        layout=body.layout,
        total_tes=body.total_tes,
        estimated_goe=body.estimated_goe,
        estimated_pcs=body.estimated_pcs,
        estimated_total=body.estimated_total,
        is_valid=body.is_valid,
        validation_errors=body.validation_errors,
        validation_warnings=body.validation_warnings,
    )
    return _program_to_response(program)


@router.delete(
    "/choreography/programs/{program_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_existing_program(program_id: str, user: CurrentUser, db: DbDep):
    """Delete a choreography program."""
    program = await get_program_by_id(db, program_id)
    if not program:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
    if program.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    await delete_program(db, program)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@router.post("/choreography/programs/{program_id}/export")
async def export_program(
    program_id: str,
    body: ExportRequest,
    user: CurrentUser,
    db: DbDep,
):
    """Export a program as SVG, PDF, or JSON."""
    program = await get_program_by_id(db, program_id)
    if not program:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
    if program.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    if body.format == "json":
        return {
            "format": "json",
            "data": {
                "id": program.id,
                "title": program.title,
                "discipline": program.discipline,
                "segment": program.segment,
                "layout": program.layout,
                "total_tes": program.total_tes,
                "estimated_total": program.estimated_total,
            },
        }

    elements = program.layout.get("elements", []) if program.layout else []
    svg = render_rink(elements)

    if body.format == "svg":
        return {"format": "svg", "svg": svg}

    # PDF: return SVG with a note (full PDF generation requires additional deps)
    return {
        "format": "pdf",
        "note": "SVG source included; server-side PDF rendering requires headless browser",
        "svg": svg,
    }
