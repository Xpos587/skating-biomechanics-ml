# Research Prompt: Best Approach for Foot Keypoint Annotations in Pose Estimation Fine-Tuning

## Context

I am building an AI coach for figure skating that uses RTMPose (via rtmlib) for 2D pose estimation. RTMPose uses the HALPE26 keypoint format (26 keypoints: 17 COCO body + 6 foot + 3 face duplicates). I want to fine-tune RTMPose on figure skating data to improve accuracy on this domain.

## What I Have

**AthletePose3D dataset** (Nagoya University):
- 1.3M frames across 12 sports (including figure skating)
- 142 3D keypoints per frame (marker-based motion capture, sub-millimeter accuracy)
- 5154 videos (1920x1088, 60fps), 24 calibrated cameras with intrinsic/extrinsic parameters
- **Pre-computed 2D projections exist**: `_coco.npy` files with 17 COCO keypoints (17kp) projected from 3D mocap. These are accurate and verified.
- **What's missing**: The 6 foot keypoints (indices 17-22 in HALPE26) are NOT in the pre-computed files.

## What I Tried and Why It Failed

**Approach: Project foot markers from 3D mocap to 2D using camera parameters**

The AthletePose3D 142kp format includes foot markers:
- L_heel (index 49), L_big_toe (26), L_5th_MTP (10)
- R_heel (112), R_big_toe (93), R_5th_MTP (77)

I wrote a perspective camera projection pipeline:
1. Load 3D foot marker positions from `.npy` files
2. Project to 2D using camera intrinsic (K) and extrinsic (R, t) matrices
3. Validate against reference ankle positions from `_coco.npy`
4. Merge: 17 COCO kp + 6 projected foot kp + 3 face dupes = 26 HALPE26 kp

**Critical formula** (from official AthletePose3D repo): `rot_mat[1:, :] *= -1` before projection.

**Results after validation:**
- Projection is mathematically correct (verified against `_coco.npy` body keypoints: 0-15px error)
- But foot keypoints have **21.9% bad projections** due to perspective distortion
- When the foot is pointed toward/away from the camera, the depth difference between heel and ankle causes the heel to project ABOVE the ankle in the image (physically correct in 3D, wrong in 2D image)
- Heels are worst: 35-36% bad. Toes vary: L_bigtoe 1.3% bad, R_bigtoe 35.2% bad, L_smtoe 21.7%, R_smtoe 2.7%
- Even after strict validation (reject if dist > 60px from ankle OR point above ankle), only 57% of frames keep all 6 foot points
- Adding "above ankle" check for all points (not just heels) catches more but still leaves many partially-annotated frames

**The fundamental problem**: Perspective projection of 3D foot markers produces 2D positions that don't match what's visually apparent in the image. A heel that's 200mm behind the ankle in 3D space projects to a completely different location in 2D than where the heel actually appears in the image. For pose estimation training, annotations must match the image — not the 3D ground truth.

## Key Statistics

| Point | Bad Rate | Median Dist to Ankle | P90 Dist |
|-------|----------|----------------------|----------|
| L_heel | 34.6% | 19px | 139px |
| L_bigtoe | 1.3% | 22px | 55px |
| L_smtoe | 21.7% | 40px | 211px |
| R_heel | 36.2% | 19px | 204px |
| R_bigtoe | 35.2% | 45px | 223px |
| R_smtoe | 2.7% | 31px | 55px |

Frame-level: 57.3% frames have all 6 foot points valid, 42.7% have at least one bad foot point.

## What I Need

I need to decide on the best strategy for producing foot keypoint annotations for RTMPose fine-tuning. The options I see:

1. **Drop foot keypoints entirely** — Train on COCO 17kp only. Reliable but loses foot information that's critical for figure skating (blade edge detection, jump takeoff/landing analysis).

2. **Pseudo-labels from pre-trained RTMPose** — Run the existing pre-trained RTMPose-HALPE26 model on all frames, use its predictions as training labels. Risks: self-reinforcing errors, domain gap (model trained on general images, not skating).

3. **Filter to reliable frames only** — Use only the 57% of frames where all 6 foot projections pass validation. Loses significant training data.

4. **Two-stage training** — First fine-tune body 17kp on all data, then fine-tune foot 6kp only on reliable frames.

5. **Something else?** — Is there a better approach I'm not considering?

## Constraints

- Target hardware: RTX 3050 Ti (4GB VRAM)
- Fine-tuning framework: mmpose (RTMPose)
- Annotation format: COCO JSON with HALPE26 26kp
- Training data: ~200k frames from AthletePose3D figure skating subset
- Images are 1920x1088, high quality, well-lit indoor ice rink
- Skaters wear various clothing (dark pants, sometimes similar colors)

## Questions for Research

1. What is the state-of-the-art approach for generating foot keypoint annotations for pose estimation fine-tuning when only 3D mocap data is available?

2. Are there published methods for handling the perspective projection problem when projecting 3D foot markers to 2D? Specifically for sports where the foot is frequently pointed at the camera (figure skating, dance, gymnastics)?

3. What is the recommended training strategy when only a subset of keypoints are reliably annotated? Should I use COCO 17kp for all frames and HALPE26 foot kp only for a subset?

4. Is pseudo-labeling from a pre-trained model a viable approach for foot keypoints specifically? What are the risks of error amplification?

5. Are there alternative foot keypoint datasets that could be used for pre-training or transfer learning? (FreiHand, LIP, MPI-INF-3DHP, etc.)

6. Has anyone published on fine-tuning RTMPose/mmpose with partial keypoint annotations? How does mmpose handle keypoints with vis=0 (not annotated)?

7. Would a weak-perspective or orthographic projection be better than full perspective for foot keypoints specifically? The depth variation within the foot is small (~200mm) compared to the camera distance (~10m).

8. What is the practical impact on downstream accuracy of training with only COCO 17kp vs HALPE26 26kp? Is the foot information worth the annotation complexity?
