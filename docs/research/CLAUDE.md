# docs/research/CLAUDE.md — Research Documentation

## Structure

`RESEARCH.md` is the **single index** — all research files must be listed there with date, topic, and outcome.

## Naming Convention

```
RESEARCH_<TOPIC>_<YYYY-MM-DD>.md     # Deep research on a specific topic
RESEARCH_SUMMARY_<YYYY-MM-DD>.md      # Comprehensive multi-source summary
RESEARCH_PROMPT_<TOPIC>_<YYYY-MM-DD>.md  # Research prompt + results (Exa/Gemini)
<PROJECT>_<TOPIC>.md                   # Integration/evaluation notes
```

Do NOT create new files with legacy names: `*_RESEARCH.md`, `*_PROMPT.md`, `*_SUMMARY.md`.

## Categories

When adding a new file, classify it in `RESEARCH.md` under one of these sections:

| Section | What goes here |
|---------|---------------|
| **Active References** | Currently relevant findings that inform implementation |
| **Integration Plans** | Specific tool/library integration (mark DONE when complete) |
| **Hardware & Sensors** | Physical sensors, IMU, data collection hardware |
| **GitHub Repositories** | Open-source project evaluations |
| **Action Segmentation** | Temporal segmentation, element boundary detection |
| **Evaluated & Rejected** | Alternatives considered and why they were dropped |
| **Original Research** | Historical first-pass research |

## Template

Every research file must have a header:

```markdown
# <Topic> — Research
**Date:** YYYY-MM-DD
**Sources:** <where the data came from>
**Status:** ACTIVE | INTEGRATED | REJECTED | HISTORICAL

---

## Summary
<2-3 sentence overview>

## Key Findings
1. ...
2. ...

## Implications for Project
- <how this affects our pipeline>
```

## Rules

1. **Update `RESEARCH.md`** when adding or removing any file
2. **No Python scripts** in research/ — code goes to `experiments/` or `ml/`
3. **Dead-end research stays** — rejected alternatives are kept for context (don't delete)
4. **Dates are real** — use the date the research was done, not the date you write the file
