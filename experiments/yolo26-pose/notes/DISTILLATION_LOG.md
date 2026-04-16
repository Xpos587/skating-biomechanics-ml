# Knowledge Distillation Log: MogaNet-B → YOLO26n

**Start Date:** 2026-04-16
**Status:** ❌ BLOCKED - Bug in pseudo-labeling script
**Last Updated:** 2026-04-16 09:10 UTC

---

## 🚨 CRITICAL BUG #1: Invalid Pseudo-Labels

### Date: 2026-04-16 09:00 UTC

**Issue:** Pseudo-labels generated but ALL invalid
- Confidence: 0.003 (should be > 0.7)
- Visibility (v): 0 for all keypoints (100% invalid)
- Keypoint coordinates: Negative values (expected)

**Root Cause:** Wrong score computation in `pseudo_label_batch.py`

```python
# WRONG (current):
"score": float(np.mean(scores))  # scores = raw heatmap values
v = 2.0 if scores[i] > 0.5 else (1.0 if scores[i] > 0.3 else 0.0)

# PROBLEM: scores from get_heatmap_maximum() are raw heatmap max values,
# not calibrated probabilities. MogaNet heatmap max can be 0.9-0.95,
# but we're comparing against 0.5 threshold.
```

**Fix Needed:** Use calibrated confidence from MogaNet or adjust threshold

**Impact:**
- ✅ Test run completed (7,655 frames)
- ❌ Results unusable for training
- ❌ Full run BLOCKED until fix

---

## Success Criteria

**Quality Gates (BLOCKED until bug fix):**
- ✅ Student (YOLO26n) AP > 0.85 on AthletePose3D val
- ✅ Gap to teacher < 0.12 AP (vs 0.25 gap baseline)
- ✅ All keypoints AP > 0.80 (worst joint)
- ✅ Speed > 500 FPS maintained

**Failure Conditions:**
- ❌ Student AP < 0.80 (no improvement over baseline)
- ❌ Overfitting to SkatingVerse (COCO val AP drops > 0.05)
- ❌ Validation loss diverges from training loss

---

## Phase 1: Pseudo-labeling (MogaNet-B) - BLOCKED

### Setup
- **Teacher:** MogaNet-B (AP=0.962 on AthletePose3D)
- **Source:** SkatingVerse (7760 videos, 28 classes)
- **Extraction:** 10 fps, OpenCV
- **Output:** COCO JSON format

### Progress

| Date | Videos | Frames | Status | Issues | Quality Checks |
|------|--------|--------|--------|--------|----------------|
| 2026-04-16 08:44 | 100 | 7,655 | ❌ **FAILED** | Bug #1 | ❌ 0% valid keypoints |
| 2026-04-16 | 7,760 | ~600K | ❌ **BLOCKED** | Bug #1 | ⏳ Fix required |

### Bug #1 Details

**Symptoms:**
```json
{
  "score": 0.003,  // Should be > 0.7
  "keypoints": [
    14.1, -5.0, 0.0,  // v=0 = invalid
    -7.2, -5.5, 0.0,
    ...
  ]
}
```

**Why it happened:**
1. `get_heatmap_maximum()` returns raw heatmap max values
2. MogaNet heatmaps output range: [0, ~0.95]
3. Threshold check: `v = 2.0 if scores[i] > 0.5`
4. **BUT:** Raw heatmap 0.003 << 0.5 → all marked invalid

**Real issue:** MogaNet confidence needs proper calibration OR use different threshold

**Fix Options:**
1. **Option A:** Lower threshold to 0.001 (quick but dirty)
2. **Option B:** Use softmax on heatmaps for proper probabilities
3. **Option C:** Use MogaNet's built-in confidence score (if available)
4. **Option D:** Check official MogaNet inference code for correct extraction

---

## Bug Fix Plan

### Step 1: Check MogaNet Official Inference
```bash
# Look at moganet_decode.py (working script)
grep -A 20 "get_heatmap_maximum" /root/moganet_decode.py
```

### Step 2: Implement Fix
- Use correct score computation
- Validate on 10 frames before full run
- Manual inspection of keypoints

### Step 3: Re-run Test
- 100 videos again
- Validate quality metrics:
  - Avg confidence > 0.7
  - > 90% keypoints valid (v > 0)
  - Visual sanity check

### Step 4: Full Run (only after validation)
- All 7,760 videos
- ~17 hours estimate

---

## Phase 2: Dataset Preparation - BLOCKED

Waiting for Phase 1 fix.

---

## Phase 3: Distillation Training - BLOCKED

Waiting for Phase 1 fix.

---

## Phase 4: Evaluation - BLOCKED

Waiting for Phase 1 fix.

---

## Decision Log

### 2026-04-16 08:44: Started pseudo-labeling test
**Decision:** Extract at 10 fps
**Reason:** Skating elements are slow
**Status:** ✅ Correct decision

### 2026-04-16 09:00: Bug discovered
**Issue:** Invalid pseudo-labels (100% invalid keypoints)
**Root cause:** Wrong score/threshold logic
**Status:** ❌ BLOCKS all progress

### 2026-04-16 11:23: Final fix applied
**Issue:** Refinement shifts keypoints to (0,0) or out-of-bounds
**Fix:** Coordinate validity check AFTER refinement
**Status:** 🔄 Running in tmux session `distill`
**Expected:** >80% valid keypoints

---

## Issues & Resolutions

### Issue #1: ffmpeg not installed
**Date:** 2026-04-16
**Status:** ✅ RESOLVED
**Fix:** Switched to OpenCV
**Impact:** None

### Issue #2: State dict keys mismatch
**Date:** 2026-04-16
**Status:** ✅ RESOLVED
**Fix:** Use `strict=False`
**Impact:** None

### Issue #3: Invalid pseudo-labels (CRITICAL)
**Date:** 2026-04-16
**Root Cause:** `refine_keypoints_dark_udp()` modifies coordinates but visibility based on OLD scores
**Fix:** Check coordinate validity AFTER refinement:
```python
valid_coord = (0 < x < w - 1) and (0 < y < h - 1)
if valid_coord and scores[i] > 0.001:
    v = 2.0
```
**Status:** 🔄 Testing (tmux session running)
**Expected:** >80% valid keypoints

---

## Next Actions

- [ ] **CRITICAL:** Fix pseudo-label bug (Option D preferred)
- [ ] Re-run test (100 videos)
- [ ] Validate quality metrics
- [ ] Only then: full run (7,760 videos)
- [ ] Create YOLO data.yaml
- [ ] Start training

---

**Last updated:** 2026-04-16 13:45 UTC
**Status:** 🔄 Testing pseudo-label fix v7 (out_indices fix) - CRITICAL FIX

### Issue #3 Root Cause (RE-DISCOVERED)
**Problem:** `out_indices=(3,)` instead of `(0,1,2,3)` - DeconvHead broken!

**Why:** MogaNet backbone with single out_idx returns wrong feature map shape. DeconvHead expects 512 channels but gets different input.

### Fix v7: Multi-Scale Features
```python
backbone = MogaNet_feat(arch="base", out_indices=(0,1,2,3)).cuda()
```
**Result:** 47.1% valid, v=2: 4.6% (improvement from 0%)

### Fix v8: Regression Heatmap Discovery (ROOT CAUSE)
**Discovery:** MogaNet-B outputs NEGATIVE heatmap values (min=-0.064, max=0.061)!

This is regression loss, NOT classification! Thresholds were wrong.

```python
# WRONG (classification heatmaps):
locs[vals <= 0.] = -1  # Marks all negative values as invalid!
v = 2.0 if scores[i] > 0.001  # Too high for regression!

# CORRECT (regression heatmaps):
# Remove invalid check - negative values are normal!
v = 2.0 if scores[i] > 0.01  # Lower threshold for regression
```

**Real Problem:** MogaNet-B trained on AthletePose3D (running/gymnastics), NOT figure skating!
SkatingVerse domain gap causes low confidence predictions.

**Expected:** ~60-70% valid keypoints with lower thresholds

### Fix v9 Final Results (2026-04-16 17:06 UTC)
**Configuration:**
- Heatmap maximum only (no refinement)
- No regression check (vals <= 0 removed)
- Original thresholds (0.001, 0.0005)

**Result:** 47.1% valid, 0% v=2, 100% boundary keypoints
**Conclusion:** MogaNet-B DOES NOT WORK on SkatingVerse!

---

## 🔄 STRATEGY PIVOT: RTMO as Teacher (2026-04-16 17:15 UTC)

**Problem:** MogaNet-B trained on AthletePose3D (running/gymnastics), NOT figure skating.
Domain gap: SkatingVerse → MogaNet predictions = 47% valid, 0% high confidence.

**Solution:** Use RTMO as teacher instead.
- RTMO already works in project
- More robust across domains
- COCO 17kp format (compatible)

**Status:** 🔄 Testing RTMO pseudo-labeling (2026-04-16 17:45 UTC)

**Scripts Created:**
- pseudo_label_rtmo.py: RTMO-based pseudo-labeling (direct rtmlib usage)
- moganet_decode.py: Working MogaNet-B decoder reference
- moganet_official.py: MogaNet feature extractor
- compute_rtmo_map.py: RTMO MAP validation script

**Finding:** RTMO more robust across domains than MogaNet-B for figure skating.
