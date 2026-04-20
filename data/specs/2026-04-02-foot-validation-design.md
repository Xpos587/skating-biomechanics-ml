# Foot Keypoint Projection Validation

**Date:** 2026-04-02
**Status:** Approved

## Problem

Projecting AthletePose3D 142kp foot markers (indices 49, 26, 10, 112, 93, 77) to 2D via perspective camera projection produces 21.9% bad annotations. When the foot points toward/away from the camera, the depth difference between heel and ankle causes the heel to project above the ankle in the image — physically correct in 3D but wrong for 2D annotation.

**Statistics (37337 frames, 500 sequences):**

| Point | Bad% | Median dist | P90 dist |
|-------|------|-------------|----------|
| L_heel (17) | 34.6% | 19px | 139px |
| L_bigtoe (18) | 1.3% | 22px | 55px |
| L_smtoe (19) | 21.7% | 40px | 211px |
| R_heel (20) | 36.2% | 19px | 204px |
| R_bigtoe (21) | 35.2% | 45px | 223px |
| R_smtoe (22) | 2.7% | 31px | 55px |

## Decision

**Validate each projected foot point independently against its reference ankle.** Invalid points get `vis=0` (not labeled). Valid points keep `vis=2`. No fallback, no imputation.

**Rationale:** For RTMPose fine-tuning, annotation quality is more important than coverage. Training on incorrectly placed keypoints teaches the model to predict wrong positions.

## Validation Rules

Applied after `project_foot_frame()` returns `(6, 2)` array, before `merge_coco_foot_keypoints()`.

```
For each foot point i (0-5):
  reference_ankle = coco_2d[15]  (L_ankle) if i < 3 else coco_2d[16]  (R_ankle)
  dist = ||foot_2d[i] - reference_ankle||

  if i is heel (0 or 3):
    valid = dist <= 60 AND foot_2d[i, 1] >= reference_ankle[1] - 30
  else:  # toe or small toe
    valid = dist <= 80

  if not valid:
    foot_2d[i] = [NaN, NaN]  # merge_coco_foot_keypoints will set vis=0
```

**Threshold rationale:**
- 60px heel: catches 34-36% bad heels, P90 of good data is ~139px but median is 19px — 60px separates modes well
- 80px toe: catches 1-35% bad toes while preserving 99%+ of good big toes (P90=55px)
- `heel_y >= ankle_y - 30`: prevents heels from appearing above torso (perspective artifact)

**Expected outcome:** ~78% of foot points retained as valid. Frames with fewer valid foot points still contribute COCO 17kp annotations.

## Implementation

Add `validate_foot_projection(foot_2d, coco_2d)` function to `src/datasets/projector.py`. Call it in `process_sequence()` in `scripts/prepare_athletepose3d.py` before `merge_coco_foot_keypoints()`.

Update `scripts/validate_projection.py` and `scripts/batch_validate_labels.py` to show validation results (green = valid, red = rejected).

Update tests in `tests/test_projection.py` to cover validation scenarios.

## Files Changed

- `src/datasets/projector.py` — add `validate_foot_projection()`
- `scripts/prepare_athletepose3d.py` — call validation before merge
- `scripts/validate_projection.py` — show valid/rejected status
- `scripts/batch_validate_labels.py` — color-code valid vs rejected foot points
- `tests/test_projection.py` — add validation tests
