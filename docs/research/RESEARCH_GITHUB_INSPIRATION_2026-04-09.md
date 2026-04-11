# GitHub Inspiration Research
**Date:** 2026-04-09
**Sources:** 5 GitHub repositories

---

## 1. mitfsc-show-tools (span314)
**URL:** https://github.com/span314/mitfsc-show-tools
**Stars:** 2 | **Tech:** Python stdlib (280 lines, zero deps)

Logistics tool for MIT Figure Skating Club exhibitions. Generates schedule, program booklet, and announcer blurbs from Google Forms CSV.

**For us: ~0.** Only idea — CSV schema for program metadata (title, skaters, music, choreographer, duration).

---

## 2. Synergie / AIOnIce (Mart1t1)
**URL:** https://github.com/Mart1t1/Synergie
**Stars:** 2 | **Tech:** Keras/TensorFlow, Transformer encoder, Xsens IMU

Jump classification from IMU sensor data (accelerometer + gyroscope). 7 classes (6 jumps + fall). Input: fixed `(400, 6)` tensor — 400 timesteps of 6 IMU channels, takeoff at frame 200.

### Ideas to borrow

| Idea | Application |
|------|-------------|
| 200-frame approach context before takeoff | Add pre-takeoff context to DTW alignment |
| 2nd derivative of gyroscope for segmentation | Analog: jerk (d²CoM/dt²) for phase detection instead of adaptive sigma |
| Fixed-size window (400 frames, takeoff = frame 200) | Normalize jump duration before DTW comparison |
| Rotation counting via Euler angle integration | Add to `phase_detector.py` via scipy Rotation API on 3D poses |
| Heuristic segmentation + ML classification hybrid | Same pattern as our rule-based phase detection + rule-based recommender |

---

## 3. skate_predict (khanhnguyendata)
**URL:** https://github.com/khanhnguyendata/skate_predict
**Stars:** 5 | **Tech:** numpy, pandas (hand-rolled gradient descent, no ML frameworks)

Predicts skater rankings at ISU World Championship via factorization: `score = f(skater_ability) × g(event_difficulty) + baseline`. Kendall's tau for evaluation.

### Ideas to borrow

| Idea | Application |
|------|-------------|
| Factorization: separate skill from conditions | Factor out camera angle, ice rink, video quality from biomechanical metrics |
| Z-normalization per-event | If adding scoring — normalize per-competition |
| Kendall's tau instead of RMSE | Correct metric for ranking jump quality |
| ISU scores are publicly available (isuresults.com) | Validate our biomechanical metrics against real judge scores |
| Pairwise logistic regression for ranking | Compare two skaters' biomechanical quality rather than absolute scores |
| Sequential residual factorization | Decompose complex movements into orthogonal factors (rotation vs height vs landing) |

---

## 4. awesome.skating.ai (na018)
**URL:** https://github.com/na018/awesome.skating.ai
**Stars:** 33 | **Tech:** TensorFlow 2, Python 3.6

**NOT a curated list.** Master thesis (Nadin-Katrin Apel, Stuttgart, 2020). 3-stage CNN for skating pose estimation: BGNet (background) → HPNet (body parts) → KPNet (keypoints). BODY_25 (25kp).

### Key finding
- Confirms: OpenPose fails on Biellmann and fast rotations → validates our RTMPose choice
- Class Imbalance Loss (anatomical distance weighting) — interesting for training
- Thesis PDF may contain literature, but our 41 papers (2025-2026) are more current

---

## 5. Figure-Skating-Choreography- (cathzvccc) — MOST INTERESTING
**URL:** https://github.com/cathzvccc/Figure-Skating-Choreography-
**Stars:** 331 | **Tech:** librosa, matplotlib, PyQt5 (single file, ~15KB, no ML)

Maps music BPM segments to skating moves via static lookup table. Algorithm: split audio into 5s windows → compute BPM → match to hardcoded dictionary (e.g., "BPM 130-150 → twizzles, hydroblading").

### What it actually does
NOT real choreography composition. It's a glorified BPM-to-move spreadsheet. No ISU rules, no sequencing, no spatial planning, no ML.

### But the choreography idea IS feasible — with the right approach

| Approach | Feasibility | Value |
|----------|-------------|-------|
| BPM-to-move lookup (this repo) | Trivial | Very low |
| Rule-based ISU program planner | Medium (2-3 months) | Medium |
| LLM-assisted choreography + music analysis | High (leverage existing models) | High |
| ML-generated choreography from scratch | Low (no training data, no eval metric) | Uncertain |
| **Choreography evaluation** (our strength) | **Very high** | **Very high** |

### What a real system would need
1. **Musical understanding:** phrase structure (AABA), energy dynamics, accent/downbeat alignment, key changes, time signatures
2. **ISU program composition rules:** required elements per level, Zayak rule, transitions, program length (2:50 SP, 4:00 FS), ice coverage
3. **Biomechanical constraints:** recovery time after jumps, speed buildup, fatigue model, stamina curves
4. **Spatial planning:** 60m × 30m rink coverage, entry/exit paths, pattern variety
5. **Aesthetic scoring:** interpretation, choreographic composition, transitions, performance

### Most promising direction for our project
**Choreography evaluation** (not generation). We already have poses, phase detection, metrics. Add `librosa` for music analysis and compare music energy peaks with skater movement peaks. ISU component scores (interpretation, choreography) are public — can train an evaluator.

---

## Actionable Next Steps

1. **Quick wins from Synergie:**
   - Add rotation counting via scipy Rotation on 3D poses
   - Try jerk-based (2nd derivative) phase detection as alternative to adaptive sigma

2. **From skate_predict:**
   - Scrape ISU scores to correlate with our biomechanical metrics
   - Add Kendall's tau if we implement any ranking

3. **From choreography repo:**
   - Prototype: `librosa` music analysis + compare with our CoM trajectory
   - Validate against ISU PCS (Program Component Scores)
   - This is a unique differentiator — no one else has both pose estimation AND music analysis
