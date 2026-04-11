# docs/CLAUDE.md — Documentation

## Structure

```
docs/
├── DATASETS.md                        # Dataset registry and relationships
├── research/                          # Research findings and paper summaries
│   ├── RESEARCH.md                    # Research memory bank (index)
│   ├── RESEARCH_SUMMARY_2026-03-28.md # Exa + Gemini findings (41 papers)
│   ├── RESEARCH_POSE_TOOLS_2026-03-31.md  # Pose estimation tool comparison
│   ├── ATHLETEPOSE3D_INTEGRATION.md   # AthletePose3D dataset integration
│   └── ...                            # IMU, segmentation, Re-ID, spatial reference
├── specs/                             # Technical specifications (design docs)
│   ├── 2026-04-11-i18n-design.md
│   ├── 2026-04-11-s3-only-storage-design.md
│   ├── 2026-04-11-saas-auth-db-profiles-design.md
│   ├── 2026-04-11-strava-fs-design.md
│   └── ...
└── plans/                             # Implementation plans (from writing-plans skill)
    ├── 2026-04-02-rtmpose-finetune-dataset.md
    └── ...
```

## Subdirectories

| Directory | Purpose | Naming Convention |
|-----------|---------|-------------------|
| `research/` | Paper summaries, Exa/Gemini findings, integration notes | `RESEARCH_*.md`, `*_RESEARCH.md` |
| `specs/` | Design documents from brainstorming skill | `YYYY-MM-DD-<topic>-design.md` |
| `plans/` | Implementation plans with bite-sized tasks | `YYYY-MM-DD-<feature>.md` |

## Key References

- `research/RESEARCH_SUMMARY_2026-03-28.md` — comprehensive summary of 41 papers across 5 themes
- `research/RESEARCH_POSE_TOOLS_2026-03-31.md` — YOLO, RTMPose, Pose3DM comparison
- `DATASETS.md` — dataset registry with download links and relationships
