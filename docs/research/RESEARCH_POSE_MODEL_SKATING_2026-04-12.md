# Pose Model Research & SkatingVerse Analysis (2026-04-12)

## DeepGlint 1st Place — SkatingVerse 95.73% (28 classes)

**Paper:** arXiv:2404.14032 (Apr 2024), DeepGlint team

### Architecture: 3 models + ensemble

| Model | Type | Accuracy | Details |
|--------|-----|----------|---------|
| UniformerV2-L | Video (ViT) | 95.02% | CLIP-pretrained, 30 epochs fine-tune |
| Unmasked Teacher | Video (ViT) | 94.55% | Kinetics-pretrained, 50+10 epochs |
| InfoGCN | Skeleton (GCN) | 92.03% | From scratch, 300 epochs |
| **Ensemble** | Weighted sum | **95.73%** | Softmax([94.5, 95.0, 92.0]) |

### Key techniques
1. **DINO detector** (IDEA Research, NOT DINOv2/v3) — DETR-variant for person bbox extraction, then FFmpeg crop
2. **ViTPose** for skeleton extraction — 320 frames per video
3. **Test-Time Augmentation** — 5 temporal + 3 spatial = 15 predictions fused
4. **Hardware:** 8x A100 (80GB) for video models, 1x A100 for InfoGCN

### SkatingVerse 28 classes (really 26 useful)
- 23 single jumps (1-4 rotation: Axel, Flip, Lutz, Loop, Salchow, Toeloop)
- 3 spins (Camel, Sit, Upright)
- 2 junk: No Basic (12), Sequence (27)
- **No multi-jump combinations, no step sequences** — much easier than FSC 64

---

## SkatingVerse vs AthletePose3D

| | SkatingVerse | AthletePose3D |
|--|-------------|---------------|
| Source | TV broadcast competitions | Controlled lab/ice rink |
| Videos | 28K clips, ~184 hours | 5K short clips |
| Resolution | 720p-1080p (broadcast) | 1920x1080 |
| FPS | ~25-30 | 60 (skating) |
| Camera | 1 moving, zooming | 12 synchronized fixed |
| Pre-extracted poses | **No** — run yourself | **Yes** — GT 3D + COCO 2D + H3.6M |
| Classes | 28 (all elements) | 6 jumps only |
| Environment | Ice, audience, graphics, glare | Clean, single athlete |

---

## Pose Model Comparison (2025-2026)

| Model | Keypoints | Foot kp | Speed | Quality | For us |
|--------|-----------|---------|-------|----------|--------|
| **RTMPose** | 17 (HALPE26) | Via HALPE26 | 30 FPS | Baseline | Current |
| **CIGPose** (CVPR 2026) | 133 (COCO-WholeBody) | **6 direct** | ~25 FPS | +1.7 AP | Worth trying |
| **ViTPose** | 17-133 | Yes | 1.5 FPS | Best | Too slow |
| **DWPose** | 17-133 | Yes | ~5 FPS | ≈ ViTPose | Too slow |
| **RTMO** | 17 | No | 60 FPS | ≈ RTMPose | Faster alternative |
| **YOLO11-Pose** | 17 | No | 100+ FPS | Medium | Fastest, one-pass |
| **DETRPose** (Jun 2025) | - | - | Slow | SOTA COCO | New, untested |
| **CIGPose** (CVPR 2026) | - | - | - | Sports-specific | See above |

### CIGPose Deep Dive (CVPR 2026)

**Paper:** arXiv:2603.09418
**GitHub:** https://github.com/53mins/CIGPose (Apache 2.0)
**ONNX:** https://github.com/namas191297/cigpose-onnx (pip install cigpose-onnx)

- **Architecture:** RTMPose backbone (CSPNeXt-P5) + Causal Intervention Module (CIM) + Hierarchical GNN
- **CIM:** Identifies confounded keypoints via predictive uncertainty, replaces with learned "canonical embeddings" — specifically targets occlusion (critical for skating: ice reflections, body rotation)
- **Keypoints:** 133 (COCO-WholeBody): 0-16 body, 17-22 foot (heel/index/pinky per side), 23-90 face, 91-132 hands
- **Performance vs RTMPose-x (same backbone, 384x288):** Body +2.1 AP, Foot +3.1 AP, Whole +1.7 AP
- **NOT trained on sports data** — general COCO-WholeBody + UBody
- **ONNX ready:** 14 pre-exported models, 54-230MB, GPU via onnxruntime-gpu
- **Not in rtmlib** — separate ONNX wrapper

### Keypoint Formats

| Format | Keypoints | Source | Notes |
|--------|-----------|--------|-------|
| COCO 17 | nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles | COCO dataset | Standard for most models |
| H3.6M 17 | pelvis, R/L hip/knee/foot, spine, thorax, neck, head, R/L shoulder/elbow/wrist | Human3.6M | Different index order than COCO |
| HALPE26 | COCO 17 + 6 foot + 3 face | MMPose/rtmlib | First 17 = COCO body |
| COCO-WholeBody 133 | COCO 17 + 6 foot + 68 face + 21 L hand + 21 R hand | COCO | Used by CIGPose, ViTPose |

**COCO 17 → H3.6M 17 mapping** (our pipeline):
```
COCO: 0=nose, 1-2=eyes, 3-4=ears, 5-6=shoulders, 7-8=elbows, 9-10=wrists, 11-12=hips, 13-14=knees, 15-16=ankles
H3.6M: 0=pelvis, 1-3=R hip/knee/foot, 4-6=L hip/knee/foot, 7=spine, 8=thorax, 9=neck, 10=head, 11-13=L shoulder/elbow/wrist, 14-16=R shoulder/elbow/wrist
```

---

## DINO Detection Clarification

**DINO (detector)** — IDEA Research, ICLR 2023. DETR-based end-to-end detector.
**DINOv2/v3** — Meta. Self-supervised feature extractor. Does NOT output bboxes natively.

| | DINO (detector) | YOLOv11n | DINOv3 (feature extractor) |
|--|-----------------|---------|--------------------------|
| mAP | 63.3 (Swin-L) | ~39 | — (needs detection head) |
| FPS | ~23 | ~125 | — |
| Person detection | Good | Excellent | Not native |

**Recommendation:** Stick with YOLO for person detection. DINOv3 as backbone for future custom detectors only. Grounding DINO (IDEA Research) adds text-guided detection but weights are API-only.

---

## Path to 90%+ on FSC 64 Classes

### Why 67.9% now
- 64 classes (combos, spins, steps) — much harder than SV's 26 single elements
- BiGRU (simple RNN) vs InfoGCN (attention graph)
- No person cropping
- RTMPose (not sports-optimized)

### Recommended steps
1. **Benchmark CIGPose vs RTMPose on 100 skating videos** before full 28K extraction
2. **Try InfoGCN** (already have `exp_infogcn.py`) on FSC 64 — expect improvement over BiGRU
3. **Add person cropping** (YOLO detect → crop → feed to classifier) — big boost
4. **Multi-modal ensemble** (skeleton + video features) — what DeepGlint did
5. **Extract SkatingVerse skeletons** on vast.ai (~40h) for additional training data

### Quick win: InfoGCN on FSC 64
- Our exp_infogcn.py is ready (pending)
- InfoGCN alone: ~70-75% expected (vs BiGRU 67.9%)
- InfoGCN + balanced sampler + multi-modal: possibly 80%+
