# Deep Research Prompt: Physics-Based Figure Skating Detection System

**Research Date:** 2026-03-27
**Target:** Gemini Deep Research (external, no project context)
**Goal:** Identify state-of-the-art approaches for physics-enhanced figure skating analysis

---

## Context

You are researching for an ML-based figure skating coaching system. The current pipeline uses:
- YOLOv11n for person detection
- MediaPipe BlazePose (33 keypoints, 2D) for pose estimation
- Normalization: root-centering + scale normalization
- Rule-based biomechanics analysis

**Current Problems:**
1. **Occlusion artifacts:** When limbs overlap (e.g., knee passing behind knee), pose estimation flips keypoints
2. **Person switching:** Skeleton jumps to background skaters when they pass behind main subject
3. **No blade state:** Cannot detect inside edge, outside edge, flat, or toe pick
4. **Missing physics:** No knowledge of height, weight, or physical constraints
5. **Limited element classification:** Need hierarchical approach (basic → complex elements)

---

## Research Questions

### Theme 1: 2D vs 3D Pose Estimation for Occlusion

**Core Problem:** When body parts overlap in 2D projection, pose models confuse left/right or flip keypoints.

**Research Questions:**
1. Does 3D pose estimation (e.g., Pose3DM-L, VPoser, TCMR) significantly reduce occlusion artifacts compared to 2D?
2. Which 3D pose models work best for:
   - Single-view monocular video (our constraint)
   - Sports with rapid rotations (figure skating, gymnastics)
   - Real-time or near-real-time processing (<500ms per frame)
3. Can temporal coherence (using previous frames to predict current pose) reduce flickering?
4. What are the computational trade-offs? (we need <2GB VRAM total)

**Success Criteria:** Identify specific model(s) that handle occlusion better than BlazePose with acceptable compute.

---

### Theme 2: Blade Edge and Skate State Detection

**Core Problem:** Figure skating technique depends critically on blade state:
- **Inside edge** (внутреннее ребро)
- **Outside edge** (наружное ребро)
- **Flat** (плоскость лезвия)
- **Toe pick** (зубец) vs **whole blade**

Current pose models cannot detect this.

**Research Questions:**
1. Has anyone published on skate blade state detection from video?
2. Can we infer blade state from:
   - Foot angle relative to motion direction?
   - Lower leg kinematics (ankle, knee, hip angles)?
   - Center of pressure trajectory?
3. Are there visual cues (spark patterns, ice marks, blade shine) that ML can learn?
4. Would specialized camera angles (side view, low angle) help?

**Success Criteria:** Determine if blade detection is feasible from monocular video, or if we need specialized sensors.

---

### Theme 3: Physical Parameter Estimation from Video

**Core Problem:** Biomechanics analysis depends on:
- Height (affects joint angles, scale)
- Weight (affects jump height, rotation speed)
- Leg length, arm length (affects moment of inertia)

But we don't know these for random YouTube videos.

**Research Questions:**
1. Can we estimate height/weight from:
   - Skeleton proportions (femur length vs tibia vs torso)?
   - Movement patterns (jump height vs force)?
   - Comparison to reference database of athletes?
2. What is the error range for anthropometric estimation from video?
3. Are there pose models that jointly estimate:
   - 3D mesh (SMPL/SMPL-X) + shape parameters (beta)?
   - Body measurements directly?
4. How sensitive are biomechanics metrics to ±10cm height or ±10kg weight errors?

**Success Criteria:** Determine if we can estimate physical parameters with <20% error from monocular video.

---

### Theme 4: Multi-Person Tracking and Re-identification

**Core Problem:** When background skaters pass behind main subject, skeleton jumps to them and never returns.

**Current Approach:** YOLOv11n detects all people, but we have no tracking.

**Research Questions:**
1. What are state-of-the-art multi-object tracking (MOT) approaches for:
   - Sports with occlusions (basketball, soccer, skating)?
   - Real-time performance?
2. Should we use:
   - Detection + tracking (YOLO + DeepSORT/OCSORT)?
   - Pose-based tracking (track unique skeleton signatures)?
   - Appearance re-identification (clothing color, body shape)?
3. Can we use motion prediction (Kalman filter, optical flow) to maintain tracking during brief occlusions?
4. How do skating rink environments (white background, similar clothing) affect tracking?

**Success Criteria:** Identify tracking approach that maintains person identity through 5+ second occlusions.

---

### Theme 5: Hierarchical Element Classification

**Core Problem:** Need to classify figure skating elements into:
- **Basic elements:** forward stroke, backward stroke, crossover, three-turn, bracket, rocker, counter, mohawk, choctaw, etc.
- **Complex elements:** Combinations of basics (e.g., salchow = back crossover → three-turn → jump → landing)

**Proposed Approach:**
1. Collect YouTube videos of basic elements
2. Manually label ~5 examples per basic element
3. Train initial classifier
4. Human-in-the-loop correction to improve
5. Classify complex elements as sequences of basic elements

**Research Questions:**
1. What is the state-of-the-art in:
   - Action recognition from sports video?
   - Few-shot learning for new actions?
   - Temporal action segmentation (finding element boundaries)?
2. Which architectures work best:
   - 3D CNNs (I3D, X3D, SlowFast)?
   - Video transformers (VideoMAE, ViViT)?
   - Pose-based action recognition (skeleton-only)?
3. Are there existing figure skating datasets or benchmarks?
4. How to handle variable-duration elements?

**Success Criteria:** Identify architecture that can classify basic elements with >80% accuracy from <50 labeled examples per class.

---

## Additional Considerations

### Computational Constraints
- Target hardware: Consumer GPU (RTX 3050 Ti, 4GB VRAM)
- Real-time preference: <100ms per frame for full pipeline
- Mobile deployment potential: Future smartphone app

### Data Availability
- We can scrape YouTube for training data
- No access to professional skating training facilities
- Manual labeling is time-consuming but possible

### Physics Integration
- We want to use physical laws (conservation of angular momentum, center of mass trajectory) to:
  - Validate pose estimates (detect when pose is physically impossible)
  - Predict hidden joint positions (occluded limb inference)
  - Improve element classification (physics-based features)

---

## Desired Output Format

For each theme (1-5), provide:

1. **State-of-the-art summary** (2-3 paragraphs)
   - Key papers and approaches from 2023-2026
   - Performance metrics (accuracy, speed, compute)

2. **Recommended approach for our use case**
   - Specific models/algorithms to try
   - Expected improvement over current baseline
   - Implementation complexity (low/medium/high)

3. **Open research questions**
   - What hasn't been solved yet?
   - Where we might need to innovate

4. **Key references**
   - Papers with links (arXiv, CVPR, ICCV)
   - Open-source implementations (GitHub)
   - Datasets we can use

---

## Search Strategy

Use these search terms (and variations):

**Theme 1:**
- "3D pose estimation monocular video occlusion"
- "Pose3DM-L figure skating gymnastics"
- "temporal coherence pose estimation sports"
- "BlazePose occlusion problems solutions"

**Theme 2:**
- "figure skating blade detection computer vision"
- "ice skate edge detection video"
- "toe pick detection figure skating"
- "figure skating biomechanics blade state"

**Theme 3:**
- "anthropometric estimation from video height weight"
- "SMPL body shape parameters estimation monocular"
- "height estimation from skeleton proportions"
- "biomechanics sensitivity body measurements error"

**Theme 4:**
- "multi-object tracking sports occlusion"
- "person re-identification figure skating"
- "DeepSORT YOLOv11 real-time tracking"
- "pose-based tracking video sports"

**Theme 5:**
- "figure skating action recognition dataset"
- "few-shot action recognition sports video"
- "temporal action segmentation figure skating"
- "hierarchical action classification complex movements"

---

## Priority Ranking

Please prioritize research in this order:
1. **Theme 2 (Blade Detection)** - Most unique to skating, highest value
2. **Theme 1 (3D vs 2D)** - Fundamental to occlusion problem
3. **Theme 4 (Tracking)** - Blocking current workflow
4. **Theme 3 (Physical Estimation)** - Important enhancement
5. **Theme 5 (Classification)** - Can iterate on current approach

---

## Notes for Researcher

- This is for a Python-based ML project (UV package manager)
- We prefer lightweight models over SOTA accuracy
- Open-source and permissive licensing required
- Russian language output is required for user-facing features
- Target users: figure skaters and coaches (not researchers)

---

**End of Research Prompt**
