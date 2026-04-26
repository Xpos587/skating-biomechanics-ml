# Teacher Heatmaps Status Report

**Date:** 2026-04-22
**File:** `/root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher_heatmaps.h5`
**Status:** VERIFICATION PENDING

---

## Background

The teacher heatmaps file is a critical component for the Knowledge Distillation (KD) training pipeline. It contains pre-computed heatmaps from MogaNet-B (teacher model) that will be used to train YOLO26-Pose (student model).

**Why this matters:**
- The entire KD strategy depends on these heatmaps being correct
- If corrupted or malformed, the student model will learn incorrect targets
- According to the coordinator report, this is **Area 2: Critical** for validation

---

## Expected Specifications

From `generate_teacher_heatmaps.py` and plan documentation:

### File Structure
- **Format:** HDF5 (.h5)
- **Dataset:** `/heatmaps`
- **Shape:** `(264874, 17, 72, 96)`
  - 264874: Total training images (estimated from FineFS v2 + AP3D + COCO)
  - 17: H3.6M keypoints
  - 72×96: Heatmap resolution (1/4 of 288×384 input)
- **Dtype:** `float16` (for storage efficiency)
- **Compression:** None (chunks are used instead)
- **Chunking:** `(32, 17, 72, 96)` or similar (batch-oriented)

### Data Characteristics
- **Encoding:** MSRA Gaussian (peak=1.0 at keypoint centers)
- **Range:** [0, 1] (clamped during generation)
- **No per-channel normalization** (unlike some other approaches)
- **Formula:** `exp(-((x-μx)² + (y-μy)²) / (2*sigma²))`

### Expected File Size
- Per heatmap: 17 × 72 × 96 × 2 bytes (float16) = 233,472 bytes ≈ 228 KB
- Total: 264874 × 228 KB ≈ **60 GB** (estimated)
- With HDF5 overhead: **~58 GB** (from plan estimates)

---

## Current Status

### What We Know

1. **Generation Script Exists**
   - Location: `experiments/yolo26-pose-kd/scripts/generate_teacher_heatmaps.py`
   - Last reviewed: 2026-04-22
   - Uses MogaNet-B model with proper MSRA Gaussian encoding
   - Applies `torch.clamp(heatmaps, 0.0, 1.0)` before saving

2. **Pre-flight Checks Completed**
   - ✅ FineFS quality: Good (94.1% keypoint visibility, no NaN/Inf)
   - ✅ HDF5 performance: 4179+ heatmaps/sec (43× faster than needed)
   - ✅ Sigma head POC: YOLO26-Pose has built-in sigma head
   - ❌ FSAnno: YouTube videos unavailable (61.5% broken links)

3. **File Location**
   - Server: Vast.ai GPU instance (IP: 167.172.27.229)
   - Path: `/root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher_heatmaps.h5`
   - **Current accessibility:** SSH connection failing (server may be down or SSH key not configured)

### What We Don't Know

1. **File Existence**
   - ❓ Has the generation script been run?
   - ❓ Did it complete successfully?
   - ❓ Were there any errors during generation?

2. **Data Integrity**
   - ❓ Is the file corrupted (incomplete write, disk full, etc.)?
   - ❓ Are all 264874 heatmaps present?
   - ❓ Is the shape correct?

3. **Data Quality**
   - ❓ Are there any NaN or Inf values?
   - ❓ Are values in the [0, 1] range?
   - ❓ Do Gaussian peaks exist (max values close to 1.0)?
   - ❓ Is the spatial structure correct (peaks at keypoint locations)?

4. **Generation Parameters**
   - ❓ What batch size was used?
   - ❓ How long did generation take?
   - ❓ Were there any skipped images (label parsing failures, etc.)?

---

## Verification Plan

### Immediate Action Required

**Run verification script on Vast.ai server:**

```bash
# Option 1: SSH and run directly
ssh root@167.172.27.229 "cd /root/skating-biomechanics-ml && python3 experiments/yolo26-pose-kd/scripts/verify_teacher_heatmaps.py"

# Option 2: If SSH is down, restart server first
# (Check Vast.ai console for instance status)

# Option 3: Copy file locally and verify (if server access is problematic)
scp root@167.172.27.229:/root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher_heatmaps.h5 /tmp/
python3 experiments/yolo26-pose-kd/scripts/verify_teacher_heatmaps.py --path /tmp/teacher_heatmaps.h5
```

### Verification Checklist

The script will check:

1. **File Integrity** ✅/❌
   - File opens without errors
   - Dataset `/heatmaps` exists
   - File size is reasonable (~58 GB)

2. **Data Structure** ✅/❌
   - Shape matches expected: `(264874, 17, 72, 96)`
   - Dtype is `float16`
   - Chunking is reasonable for batch loading

3. **Data Quality** ✅/❌
   - No NaN values
   - No Inf values
   - Values in [0, 1] range
   - Gaussian peaks present (max > 0.95)

4. **Spatial Structure** ✅/❌
   - Peaks centered at keypoint locations
   - No artifacts (striping, blocking, noise)
   - Reasonable spread for Gaussian encoding

5. **Generation Code Review** ✅/❌
   - MSRA Gaussian used (not per-channel norm)
   - Clamping applied
   - Batch processing optimized

---

## Potential Issues

### Issue 1: File Does Not Exist

**Symptoms:** SSH connection fails or file not found at expected path

**Possible causes:**
- Vast.ai instance destroyed or not running
- Generation script never executed
- File saved to different location

**Resolution:**
1. Check Vast.ai console for instance status
2. If instance destroyed: need to regenerate (requires GPU rental)
3. If wrong path: search for file with `find /root -name "teacher_heatmaps.h5"`

### Issue 2: File Corrupted

**Symptoms:** HDF5 open error, shape mismatch, NaN/Inf values

**Possible causes:**
- Incomplete write (disk full, process killed)
- Disk I/O errors
- Incorrect dtype/shape during generation

**Resolution:**
- Regenerate from scratch (requires re-running generation script)
- Check disk space: `df -h` (need ~100 GB free)
- Check generation logs for errors

### Issue 3: Data Quality Issues

**Symptoms:** Values outside [0, 1], no Gaussian peaks, wrong spatial structure

**Possible causes:**
- Bug in generation script (wrong normalization)
- MogaNet-B model output different than expected
- Coordinate system mismatch

**Resolution:**
- Review generation script code
- Test MogaNet-B inference on few images manually
- Check if model weights are correct (moganet_b_ap2d_384x288.pth)

### Issue 4: Shape Mismatch

**Symptoms:** Shape is different from expected

**Possible causes:**
- Different number of training images than estimated
- Different heatmap resolution (model output size changed)
- Transposed dimensions

**Resolution:**
- If just count difference: update expected shape (not critical)
- If resolution different: need to regenerate or add resize layer
- If transposed: fix generation script

---

## Dependencies

### Required for Verification

1. **Python libraries:**
   ```bash
   pip install h5py numpy pillow tqdm
   ```

2. **File access:**
   - SSH access to Vast.ai server (IP: 167.172.27.229)
   - Or local copy of file

3. **Generation script** (for code review):
   - `experiments/yolo26-pose-kd/scripts/generate_teacher_heatmaps.py`

### Required for Regeneration (if needed)

1. **MogaNet-B model:**
   - Path: `/root/data/datasets/raw/athletepose3d/model_params/moganet_b_ap2d_384x288.pth`
   - Size: ~?? MB (AthletePose3D weights)

2. **Training data:**
   - FineFS v2: YOLO format (images + labels)
   - AP3D: YOLO format (if used)
   - COCO: YOLO format (10% mix)

3. **GPU:**
   - Recommended: RTX 4090 or RTX 5090
   - Estimated time: 1.5-3 hours (from plan)
   - Estimated cost: ~$0.89 (on Vast.ai)

---

## Next Steps

### Step 1: Verify File Exists (TODAY)

```bash
# Check if Vast.ai instance is running
# (Check Vast.ai console)

# Try SSH connection
ssh -o ConnectTimeout=10 root@167.172.27.229 "echo 'Connection OK'"

# Check if file exists
ssh root@167.172.27.229 "ls -lh /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data/teacher_heatmaps.h5"
```

**Outcome:**
- If file exists: Proceed to Step 2
- If file missing: Investigate why (check logs, check generation script output)
- If SSH fails: Check Vast.ai console, restart instance if needed

### Step 2: Run Verification Script (TODAY)

```bash
ssh root@167.172.27.229 "cd /root/skating-biomechanics-ml && python3 experiments/yolo26-pose-kd/scripts/verify_teacher_heatmaps.py"
```

**Outcome:**
- If all checks pass: ✅ SAFE TO USE — proceed with KD training
- If checks fail: Review specific failures, decide regeneration vs. workarounds

### Step 3: Address Issues (IF NEEDED)

Based on verification results:

- **File missing:** Regenerate (requires GPU rental)
- **File corrupted:** Regenerate (requires GPU rental)
- **Quality issues:** Investigate root cause, may need script fixes
- **Shape mismatch:** May be acceptable if just count difference, otherwise regenerate

### Step 4: Document Results (ALWAYS)

After verification:

1. Save verification report JSON
2. Update this document with actual findings
3. Create summary for KD plan review

---

## Success Criteria

The teacher heatmaps file is considered **SAFE TO USE** when:

1. ✅ File exists at expected path
2. ✅ File opens without HDF5 errors
3. ✅ Shape is approximately correct (may vary in count)
4. ✅ Dtype is float16
5. ✅ No NaN or Inf values in 100+ random samples
6. ✅ All values in [0, 1] range
7. ✅ Gaussian peaks present (max > 0.95)
8. ✅ Spatial structure looks correct (visual inspection)

**If any criterion fails:** Investigate and resolve before using for KD training.

---

## References

- **Generation script:** `experiments/yolo26-pose-kd/scripts/generate_teacher_heatmaps.py`
- **Verification script:** `experiments/yolo26-pose-kd/scripts/verify_teacher_heatmaps.py`
- **KD Plan:** `data/plans/2026-04-18-kd-moganet-yolo26-plan.md`
- **Coordinator Report:** `data/plans/2026-04-18-kd-moganet-yolo26-coordinator-report.md`
- **Pre-flight Results:** `data/plans/2026-04-21-preflight-results.md`

---

**Last updated:** 2026-04-22
**Next action:** Run verification script on Vast.ai server
**Owner:** User (Michael)
**Blocker for:** KD training (Task 13 in plan)
