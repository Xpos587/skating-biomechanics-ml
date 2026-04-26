# FineFS v2 Quality Audit Report

**Date:** 2026-04-22
**Dataset:** FineFS (video frames + 3D skeletons → YOLO pose format)
**Status:** IN PROGRESS

## Conversion Progress

| Metric | Value |
|--------|-------|
| Train videos | 933/933 (100%) |
| Val videos | 234/234 (100%) |
| Train labels | ~149K (estimated) |
| Val labels | ~37K (estimated) |
| Duration | ~2 hours total |
| Speed | ~7.5s per video |

## Audit Checklist

### 1. Label Format ✅/❌
- **Expected:** 56 fields per line (class_id + 4 bbox + 17×3 keypoints)
- **Check:** All labels have exactly 56 space-separated values
- **Class ID:** Must be 0 (single class: person)
- **Coords:** All values in [0, 1] range

### 2. Bbox Quality ✅/❌
- **Critical fix:** Global normalization per video (NOT per-frame)
- **Expected mean:** ~0.3-0.5 (width and height)
- **Anti-pattern:** NOT ~1.0 (indicates v1 per-frame bug)
- **Range:** Should vary significantly across frames (min < 0.2, max > 0.6)

### 3. Keypoints ✅/❌
- **Values:** No NaN, no Inf
- **Range:** All coords in [0, 1]
- **Visibility:** All 17 keypoints have v=1.0 (all treated as visible in FineFS)
- **Spread:** x_range and y_range should be > 0.05 (not degenerate)

### 4. Image-Label Correspondence ✅/❌
- **Count:** Number of .jpg == Number of .txt
- **Stems:** Filenames match (videoID_fXXXXXX)
- **Valid images:** All .jpg files readable by cv2.imread

### 5. Split Integrity ✅/❌
- **No leakage:** video_id not duplicated between train/val
- **Video-level split:** 80/20 at video level, not frame level
- **Unique IDs:** train_ids ∩ val_ids = ∅

### 6. Spot Check (Visual) ✅/❌
- **Sample:** 5-10 random images
- **Metrics:** bbox size, keypoint ranges
- **Sanity:** Bbox should reasonably frame the person

## Audit Command

```bash
ssh vastai "cd /root/skating-biomechanics-ml && python3 experiments/yolo26-pose-kd/scripts/audit_finefs.py"
```

## Expected Results

If all checks PASS:
- Overall: PASS
- Ready for next step: Teacher heatmap generation
- Command: `python3 experiments/yolo26-pose-kd/scripts/extract_teacher_heatmaps.py`

If any check FAIL:
- Report specific failure with metrics
- Determine if block is critical or can be worked around
- May need to re-run conversion with fix

## Key Differences from v1

| Aspect | v1 (FAILED) | v2 (EXPECTED) |
|--------|-------------|---------------|
| Normalization | Per-frame (bug) | Global per-video (fixed) |
| Bbox mean | ~1.0 (constant) | ~0.3-0.5 (varies) |
| Source | Dummy black images | Real video frames |
| Coverage | All frames | Element timing only |

---

**Next Step After Audit:**
Generate teacher heatmaps using trained yolo26m-pose model
- Input: FineFS v2 YOLO dataset
- Output: Heatmap tensors for student training
- Script: `extract_teacher_heatmaps.py`

**Status:** Awaiting conversion completion
