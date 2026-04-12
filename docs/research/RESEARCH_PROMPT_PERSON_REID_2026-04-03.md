# Research Prompt: Person Re-Identification Across Full Occlusion in Monocular Figure Skating Video

**Target:** Gemini Deep Research
**Date:** 2026-04-03
**Language:** Russian (prompt) → English (research expected)

---

## Промпт для Gemini Deep Research

```
I need a comprehensive research report on **person re-identification (Re-ID) and person-specific identity learning** for a monocular video analysis system in figure skating. The system must track a specific target person across the entire video, including through **full occlusions** where the person completely disappears from frame for extended periods (5-30 seconds).

## SYSTEM CONTEXT

I'm building an AI figure skating coach that analyzes monocular video recordings of ice skating sessions. The system:

1. Detects multiple people in each frame using RTMPose (26-keypoint body model with foot keypoints)
2. Converts detections to H3.6M 17-keypoint format (standard biomechanics format)
3. Tracks each person across frames and assigns stable track IDs
4. The USER selects a specific target person (e.g., "the skater in the center")
5. The system follows ONLY that person for the entire video, extracting their pose for biomechanical analysis (joint angles, jump height, edge detection, etc.)

The system runs on a single camera (monocular), records ice skating sessions where:
- Multiple skaters may be on the ice simultaneously (2-8 people)
- People wear similar clothing (black/dark athletic wear — color-based Re-ID is unreliable)
- The ice rink has uniform white background with minimal texture
- Camera is typically static (tripod/phone mount) or slowly panning
- Skaters move fast and perform rotations (spins, jumps)
- People frequently occlude each other — a skater passing between the camera and the target can completely block the target for many seconds

## THE CORE PROBLEM

When the target person is **fully occluded** (completely hidden behind another person), both tracking algorithms I've tried fail:

### What I've Implemented (and why it fails):

**1. Sports2D Hungarian Algorithm (Pose2Sim approach)**
- Computes pairwise keypoint distance matrix between consecutive frames
- Uses scipy.optimize.linear_sum_assignment for optimal one-to-one matching
- Auto-computes max_dist threshold from bounding box diagonals
- Stores "lost" track keypoints for re-association after reappearance
- **FAILURE MODE:** During occlusion, the occluder moves to where the target was. The Hungarian algorithm assigns the target's track ID to the occluder (minimum distance match). After occlusion, the target gets a new ID. The system follows the wrong person.

**2. DeepSORT (deep-sort-realtime)**
- Appearance-based ReID via CNN embedding (MobileNetV3-based)
- Kalman filter predicts position during missed frames
- cosine distance for appearance matching + IoU for spatial matching
- **FAILURE MODE:** The CNN embedder operates on cropped bounding boxes. On ice, all skaters look similar (dark clothing, similar body proportions). Appearance features are not discriminative enough. The Kalman filter predicts the target should be where the occluder is (because the occluder physically occupies that space), so DeepSORT also assigns the target's track to the occluder.

**3. Anti-steal protection (my custom addition)**
- Detects track ID "theft" when centroid jumps > 0.15 normalized distance between frames
- When theft detected, clears the impostor's data and searches for the real target among remaining detections
- Uses combined position proximity + biometric body proportions for re-matching
- **FAILURE MODE:** If the target is FULLY occluded (zero detections of the target person), there's nothing to match against. The system can only leave a gap (NaN poses) and hope the target reappears later — but when they do, there's no mechanism to recognize them.

### The fundamental issue:

All current approaches are **frame-to-frame reactive** — they try to match detections in frame N+1 to tracks from frame N. None of them build a **person-specific identity model** that can answer "is this person the same as the one I was tracking before the occlusion?" after a 20-second gap.

## WHAT I NEED TO RESEARCH

I need to find approaches that can **learn and remember a specific person's identity** from the video itself, so that after a full occlusion, the system can recognize the target when they reappear. Specifically:

### 1. One-shot / Few-shot Person Re-ID from video
- Can we build a person-specific Re-ID model from just the first few seconds of video where the target is visible?
- What architectures work for learning discriminative features from very few examples?
- How to handle the "domain gap" — the person looks different from different angles, distances, lighting conditions, and while performing different skating movements (standing vs. spinning vs. jumping)?

### 2. Temporal identity models
- Instead of frame-by-frame matching, can we build a **trajectory-based identity model** that captures how the person moves (gait, speed patterns, turning habits)?
- Can we use the skeleton/keypoint data (already extracted) as the identity signal instead of appearance (RGB pixels)?
- Skaters have distinctive movement patterns — push-off rhythm, spin entry angles, arm positions during jumps. Can these be quantified as identity features?

### 3. Online / incremental learning for person identity
- Is there a way to incrementally build a person model as the video plays, without pre-training?
- Can we maintain a "gallery" of the target person's features that grows over time?
- How to balance: the model should be stable (not forget the identity) but also adaptable (handle appearance changes due to sweat, wet clothing, lighting)?

### 4. Practical approaches for limited compute
- My hardware: NVIDIA RTX 3050 Ti with 4GB VRAM
- Must run in near-real-time (<100ms per frame at 30fps)
- The person detection and pose estimation already use most of the GPU budget
- Can we do Re-ID on CPU while pose estimation runs on GPU?
- What lightweight Re-ID models exist (<50MB, <10ms per inference)?

### 5. Skeletal/biometric identity (not appearance-based)
- Since we already have 17-keypoint H3.6M poses for every detected person, can we use skeletal structure as identity?
- Anthropometric ratios: shoulder width / torso length, femur / tibia ratio, arm span / height — these are unique per person
- But: the poses are 2D (normalized pixel coordinates), so ratios change with perspective. Can we compensate?
- Bone lengths should be constant — can we estimate them from 2D poses and use them as identity?
- Gait features: step frequency, stride length, push-off timing — can these distinguish skaters?

### 6. Hybrid approaches
- What if we combine: (a) skeletal identity for short gaps + (b) trajectory prediction for medium gaps + (c) skeletal re-matching for long gaps?
- Can we use the Kalman filter state (position, velocity) as an additional matching signal?
- What about multi-hypothesis tracking — maintain multiple candidate identities and resolve later?

### 7. Handling the specific failure mode
- The key scenario: Person A (target) is at position X. Person B (occluder) passes in front, completely blocking A for 5-30 seconds. During this time, the detector might still detect A's keypoints on B (because the pose estimator doesn't know about occlusion). After B moves away, A is visible again at a slightly different position.
- How do commercial systems (e.g., sports broadcast tracking, Hawk-Eye, player tracking in NBA/NHL) handle this?
- Are there specific algorithms for "track recovery after occlusion" that I should know about?

## CONSTRAINTS

1. **Single camera, monocular** — no multi-view triangulation possible
2. **No pre-registration** — the system doesn't know the skaters beforehand, must learn identity from the video itself
3. **Real-time or near-real-time** — the analysis runs after recording, but should complete in minutes, not hours, for a 2-minute video
4. **Hardware: RTX 3050 Ti (4GB VRAM)** — limited GPU memory
5. **Black/dark clothing on white ice** — appearance-based methods are inherently limited
6. **Fast motion + rotations** — figure skaters spin at 2-6 rev/s, move at 5-10 m/s
7. **Variable person count** — 1-8 people in frame, people enter/exit the rink
8. **Already have 17-keypoint poses per frame** — can leverage existing pose data, not just raw pixels

## WHAT I WANT IN THE REPORT

1. **Ranked list of approaches** — from most promising to least, with justification for figure skating specifically
2. **Concrete algorithms/models** — specific paper titles, GitHub repos, model names, not just vague concepts
3. **Feasibility assessment** — which approaches can run on RTX 3050 Ti 4GB within the compute budget?
4. **Implementation complexity** — rough estimate of implementation effort for each approach
5. **Expected accuracy** — what Re-ID accuracy can I expect for each approach in the skating domain?
6. **Skeletal identity deep dive** — dedicated section on using skeleton keypoints as identity features, since I already have this data
7. **Existing sports tracking solutions** — how do professional sports analytics systems (STATSports, Catapult, Hawk-Eye, NHL player tracking) handle occlusion recovery?
8. **Code examples or pseudocode** — for the top 2-3 most promising approaches
9. **Training data requirements** — do any approaches require training data? Can I use the video itself as training data (self-supervised)?
10. **Integration strategy** — how to integrate the chosen approach into an existing pipeline that already has frame-by-frame pose detection + tracking

## FORMATTING

Please provide the report in **English** with:
- All paper citations (arXiv IDs where available)
- GitHub repository URLs for open-source implementations
- Specific model checkpoints/datasets to download
- Comparison tables where possible
- Pseudocode for key algorithms
```

---

## Заметки для себя

- Сохранить в `research/RESEARCH_PERSON_REID_2026-04-03.md` после получения ответа
- Обновить `research/RESEARCH.md` после анализа
- Связанные файлы: `src/tracking/sports2d.py`, `src/tracking/deepsort_tracker.py`, `src/pose_estimation/rtmlib_extractor.py`
