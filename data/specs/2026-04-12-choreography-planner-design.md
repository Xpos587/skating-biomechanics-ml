# Choreography Planner — Design Spec

**Date:** 2026-04-12
**Status:** DRAFT
**Scope:** MVP — Interactive 2D editor for Singles (Men/Women) programs

---

## 1. Overview

AI-powered choreography planner for figure skating. Upload music → AI generates element layout (CSP solver) → edit in interactive timeline + rink diagram → export as SVG/PDF.

**Target users:** Coaches and competitive skaters.
**Key differentiator vs Skate Plan (iOS):** AI generation of optimal layouts, ISU rule validation.

### MVP Scope

- Singles only (Men + Women)
- Short Program + Free Skate
- ISU 2025/26 season rules
- Server-side music analysis (madmom + MSAF)
- CSP solver for element placement (OR-Tools)
- Interactive timeline editor with drag-and-drop
- SVG rink diagram with skating path
- IJS score estimation (TES + GOE + PCS)
- Export: SVG rink image, element list + score, PDF protocol
- No 3D, no LLM, no diffusion models, no AI dance generation

### What's NOT in MVP

- Pairs / Ice Dance
- 3D visualization (SMPL-X)
- AI transition generation (LLM)
- Music generation (inverse problem)
- Video analysis integration ("plan vs fact")
- Per-element GOE estimation from video (ML)
- Multi-program season planning

---

## 2. Architecture

### Data Flow

```
User uploads music → POST /api/v1/choreography/upload-music
        ↓
madmom + MSAF analysis (async arq job) → cache in DB
        ↓
User selects program type + sets jump inventory
        ↓
POST /api/v1/choreography/generate → CSP solver (OR-Tools)
        ↓  input: inventory, music features, ISU rules
        ↓  output: 3 JSON layouts (sorted by TES)
        ↓
User picks layout, edits in timeline (drag-and-drop)
        ↓
POST /api/v1/choreography/validate → rules engine → errors/warnings
        ↓
Export: SVG rink diagram, PDF protocol, element list with score
```

### Module Placement

| Module | Location | Why |
|--------|----------|-----|
| Routes | `backend/app/routes/choreography.py` | REST endpoints |
| DB Models | `backend/app/models/choreography.py` | SQLAlchemy ORM |
| Pydantic Schemas | `backend/app/schemas/choreography.py` | Request/response types |
| Rules Engine | `backend/app/services/choreography/rules_engine.py` | ISU rules (Zayak, well-balanced, BV) |
| CSP Solver | `backend/app/services/choreography/csp_solver.py` | OR-Tools constraint satisfaction |
| Music Analyzer | `backend/app/services/choreography/music_analyzer.py` | madmom + MSAF wrapper |
| Rink Renderer | `backend/app/services/choreography/rink_renderer.py` | SVG generation |
| Frontend Pages | `frontend/src/app/(app)/choreography/` | Next.js pages |
| Frontend Components | `frontend/src/components/choreography/` | Timeline, Rink, Score, Inventory |
| Frontend API | `frontend/src/lib/api/choreography.ts` | API client |
| Frontend Types | `frontend/src/types/choreography.ts` | TypeScript types |

**Key constraint:** Backend does NOT import ML. Rules engine and CSP are pure Python business logic (no GPU, no onnxruntime).

### Cross-Pipeline Integration Notes

ML pipeline work is happening in parallel (improved 2D/3D poses, TAS element segmentation, GOE from video). Potential future intersections:

1. **Element codes** must be unified — single source of truth shared between ML `element_defs.py` and choreography rules engine
2. **GOE from video** (ML, future) vs **estimated GOE** (planner, MVP) — different metrics but comparable results
3. **Future feature:** Upload training video → TAS segmentation → compare against choreography plan → "you hit 5/7 planned jumps"

---

## 3. API Endpoints

All under `/api/v1/choreography/`.

```
POST   /upload-music              Upload mp3/wav → {music_id, filename}
GET    /music/:id/analysis        Music analysis result (BPM, structure, energy)
POST   /generate                  Generate layouts → {layouts: [...]} (3 options)
POST   /validate                  Validate layout → {errors[], warnings[], score}
POST   /render-rink               Layout → SVG string
GET    /programs                  List user's saved programs
GET    /programs/:id              Get full program
PUT    /programs/:id              Save edited program
DELETE /programs/:id              Delete program
POST   /programs/:id/export       Export → file (svg/pdf/json)
GET    /elements/registry        All ISU elements with BV
```

### New Backend Dependencies

```toml
# backend/pyproject.toml — additions
"madmom>=0.16.1",     # RNN-based beat/downbeat tracking
"msaf>=0.1.0",        # Music Structure Analysis Framework
"librosa>=0.10.0",    # STFT, MFCC, onset detection
"ortools>=9.0",       # Google OR-Tools for CSP
```

### New Frontend Dependencies

```json
{
  "wavesurfer.js": "^7.0",
  "@dnd-kit/core": "^6.0",
  "@dnd-kit/sortable": "^8.0"
}
```

---

## 4. DB Models

### MusicAnalysis

```python
class MusicAnalysis(TimestampMixin, Base):
    id: str (PK, UUID)
    user_id: str (FK to users)
    filename: str
    audio_url: str (R2 key)
    duration_sec: float
    bpm: float | None
    meter: str | None          # "4/4", "3/4"
    structure: dict | None    # JSON: [{type, start, end}, ...]
    energy_curve: dict | None # JSON: {timestamps: [...], values: [...]}
    downbeats: list[float] | None
    peaks: list[float] | None  # energy peaks (timestamps for jump placement)
    status: str              # pending / analyzing / completed / failed
```

### ChoreographyProgram

```python
class ChoreographyProgram(TimestampMixin, Base):
    id: str (PK, UUID)
    user_id: str (FK to users)
    music_analysis_id: str (FK to music_analysis, nullable)
    title: str | None
    discipline: str          # "mens_singles", "womens_singles"
    segment: str             # "short_program", "free_skate"
    season: str              # "2025_26"
    layout: dict | None      # JSON: full program layout (elements + transitions)
    total_tes: float | None
    estimated_goe: float | None
    estimated_pcs: float | None
    estimated_total: float | None
    is_valid: bool | None    # rules engine validation result
    validation_errors: list[str] | None
    validation_warnings: list[str] | None
```

**Note:** Alembic migration needed for both tables.

---

## 5. ISU Rules Engine

### Element Database

Table of all ISU elements with properties needed for layout generation:

| Field | Type | Description |
|-------|------|-------------|
| `code` | str | ISU code: 3Lz, 2A, CSSp4, StSq4, ChSq1 |
| `type` | enum | jump / spin / step_sequence / choreo_sequence |
| `base_value` | float | ISU BV (Communication 2707) |
| `rotations` | float | 0.5, 1, 2, 3, 4 (jumps only) |
| `has_toe_pick` | bool | lutz, flip, toe_loop |
| `entry_edge` | str | LFO, RBI, etc. (for jump edge compatibility) |
| `exit_edge` | str | Edge after landing |
| `combo_eligible` | bool | Can be in combination/sequence |
| `short_program_eligible` | bool | Allowed in SP |

Static data (JSON or Python dict), loaded at startup. Not in DB.

### Well-Balanced Program Rules

**Free Skate (Senior Singles 2025/26):**
- Maximum 7 jumping passes (at least 1 Axel-type)
- Maximum 3 jump combinations/sequences (only one 3-jump combo, Euler max once)
- Maximum 3 spins (combination, flying, single-position — all different codes)
- 1 step sequence (level 1-4)
- 1 choreographic sequence

**Short Program (Senior Singles 2025/26):**
- Maximum 7 jumping passes
- Maximum 3 jump combinations/sequences
- Maximum 3 spins (combination, flying, single-position — all different codes)
- 1 step sequence (specific level required)
- 1 choreographic sequence

### Zayak Rule

- Jumps with 3+ rotations: max 2 attempts
- If attempted twice, at least one must be in a combination/sequence
- Violation → element invalidated (0 points)

### Back-Half Bonus

- Free Skate: last 3 of 7 jumping passes get +10% BV
- Short Program: last jumping pass gets +10% BV

### Edge Compatibility

- Jump exit edge determines what transition is physically possible next
- Example: Lutz exits on RBO → cannot immediately do forward inside three-turn
- Simplified for MVP: adjacency matrix of valid element→transition→element sequences

### Score Calculation

```
TES = Σ(element.base_value × goe_factor) + back_half_bonus
GOE range: -5 to +5 per element
goe_factor: scales BV (e.g., GOE +3 on 5.90 BV = 5.90 × 1.4 = 8.26)
back_half_bonus = element.base_value × 0.1 for each qualifying element
```

### MVP Scope for Rules

- Singles only (Men + Women)
- Short Program + Free Skate only
- 1 season (2025/26)
- Rules as static Python module, updated manually per season

---

## 6. Music Analysis

### Pipeline

```
Upload mp3/wav → save to R2 → enqueue arq job
        ↓
arq worker:
  1. madmom.features.onset_strength() → onset envelope
  2. madmom.beat.BeatTracker → BPM + meter
  3. madmom.beat.DBNLSDownBeatTracker → downbeats
  4. librosa.feature.rms() → energy curve (per 0.5s window)
  5. scipy.signal.find_peaks() → energy peaks
  6. MSAF.process() → structure boundaries (verse/chorus/bridge)
        ↓
Save result to MusicAnalysis table
```

### What CSP Solver Uses

| Music Feature | How CSP Uses It |
|---------------|----------------|
| `peaks` | Jumps placed near energy peaks (musical gravity) |
| `structure` | Spins on verse, choreo on bridge, jumps on chorus |
| `downbeats` | Elements aligned to musical beats |
| `duration` | Time violation check (≤2:40 FS / ≤2:50 FS men) |
| `bpm` | Display only (informational) |
| `meter` | Display only (informational) |

### Caching

- Music analysis is cached per file (MusicAnalysis table)
- Re-uploading same file → returns cached result
- Cache invalidated if audio file changes

---

## 7. CSP Solver

### Problem Formulation

**Variables:** Sequence of elements E₁, E₂, ..., Eₙ from skater's inventory
**Domains:** Timestamps [0, duration] and spatial positions on 60m×30m rink
**Constraints:**

| Constraint | Rule |
|-----------|------|
| C_capacity | Jumping passes ≤ 7 |
| C_zayak | Count(J_i) ≤ 2 for jumps with 3+ rotations; if == 2 then one must be in combo |
| C_combo_count | Jump combinations/sequences ≤ 3 |
| C_combo_triple | Max one 3-jump combination |
| C_euler | Euler max once between two listed jumps |
| C_spins | Spins ≤ 3, all different codes |
| C_spin_types | Must include: combination, flying, single-position |
| C_step_seq | Exactly 1 step sequence |
| C_choreo_seq | Exactly 1 choreographic sequence |
| C_axel | At least 1 Axel-type jump |
| C_back_half | Last 3 jumping passes (FS) for +10% BV bonus |
| C_energy | Jumps placed near music energy peaks |
| C_structure | Spins on verses, choreo on bridge |
| C_duration | Total ≤ 2:40 (FS women) / ≤ 2:50 (FS men) |
| C_spatial | Elements connected by feasible transitions |

### Solver: OR-Tools CP-SAT

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
# Add variables, constraints
# Objective: maximize total TES (with back-half bonus)
# Solver: backtracking with forward checking
# Output: top 3 layouts by score
```

### User Input (Jump Inventory)

Coach defines which elements the skater can reliably do:
```json
{
  "jumps": ["3Lz", "3F", "3Lo", "3S", "2A", "2Lz", "2T", "1Eu"],
  "spins": ["CSp4", "LSp4", "CCSp4"],
  "combinations": ["3Lz+2T", "3Lz+3T", "3S+2T", "3Lz+1Eu+2S"]
}
```

---

## 8. Frontend UI Design

### Layout

```
┌──────────────────────────────────────────────────────────┐
│ Top Bar: Logo | Music info (BPM, duration) | Program    │
│          select | [Inventory] [Generate] [Export]       │
├─────────────────────────────────┬────────────────────────┤
│ TIMELINE (60%)                │ RINK DIAGRAM (40%)      │
│ ┌─────────────────────────────┐ │ ┌──────────────────┐   │
│ │ Verse/Chorus/Bridge bars   │ │ │                  │   │
│ ├─────────────────────────────┤ │ │   Ice Rink       │   │
│ │ Audio waveform + playhead   │ │ │   (SVG)          │   │
│ ├─────────────────────────────┤ │ │   + Skating path  │   │
│ │ Elements lane (drag-drop)   │ │ │   + Element dots  │   │
│ │  [3Lz][+2T]  [CSp4]       │ │ │                  │   │
│ ├─────────────────────────────┤ │ └──────────────────┘   │
│ │ Transitions lane             │ │ Legend + controls      │
│ ├─────────────────────────────┤ │                          │
│ │ Energy curve lane           │ │                          │
│ └─────────────────────────────┘ │                          │
├─────────────────────────────────┴────────────────────────┤
│ Score Bar: TES | GOE | PCS | Total | Duration | 7/7 | 3/3│
└──────────────────────────────────────────────────────────┘
```

### Element Block Colors

- Jumps: orange gradient
- Spins: purple gradient
- Step sequences: green gradient
- Choreographic sequences: blue gradient
- Back-half bonus: yellow border
- Transitions: gray dashed

### ISU Element Codes

All element blocks use standard ISU notation (English): 3Lz, 2A, CSp4, StSq4, ChSq1, +2T, +1Eu, etc.

### Key Interactions

1. **Generate flow:** Upload music → select program type → set inventory → click Generate → see 3 layout options → pick one
2. **Edit flow:** Drag element blocks horizontally on timeline → rules engine validates in real-time → score updates instantly
3. **Export flow:** Click Export → choose format (SVG/PDF/JSON) → download file
4. **Playback:** Playhead moves across timeline, dot moves along rink path

### Rink Diagram

- SVG, top-down orthographic, 60m × 30m
- Realistic skating paths (arcs for edges, dashed lines for flight, circles for spins, dots for turns)
- Element labels with jump codes
- Back-half elements highlighted
- Zoom/pan controls
- Responsive (fills available space)

---

## 9. Export Formats

### SVG Rink Image
- Full rink with traced path and element markers
- Include title, element legend, score summary

### PDF Protocol
- ISU-style score sheet layout
- Element table with BV, GOE, GOE factor, total
- Program component breakdown
- Total TES + estimated PCS + estimated Total

### JSON
- Full program layout (machine-readable)
- Compatible with potential future import

---

## 10. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| ISU rule error → skater gets deduction | CRITICAL | "Not official" disclaimer, deterministic rules engine, test suite |
| Rules change annually | MEDIUM | Modular rules module, static config per season |
| Element codes mismatch with ML pipeline | MEDIUM | Unified element code registry |
| Music analysis fails (EDM, ambient) | LOW | Fallback to RMS energy only |
| OR-Tools CSP too slow | LOW | Typical program has <20 elements, solver finishes in <1s |
| Frontend drag-and-drop complexity | MEDIUM | Use @dnd-kit (proven library) |
| Rink SVG rendering performance | LOW | Server-side SVG generation, not real-time |
