# Spatial Reference Research Summary (Exa)
**Date:** 2026-03-28
**Source:** Exa Web Search (20 results)

---

## Executive Summary

**Критические находки:**
1. **Ice hockey rink detection существует!** - Несколько papers конкретно по хоккею на льду
2. **Sports field calibration** - зрелая область с готовыми решениями для soccer/basketball
3. **VIO (Visual-Inertial Odometry)** - работает на смартфонах, даёт 6DOF pose
4. **Vanishing points** - классический метод для camera pose estimation

**Рекомендация:** Hybrid подход = Sports field lines + IMU fallback

---

## Категория 1: Horizon Line Detection

### Papers Found

1. **Horizon line detection using supervised learning** (Ahmad et al., 2020)
   - Edge-based vs edge-less (classification)
   - Fusion strategy for robustness
   - Dataset: extensive sky/horizon images

2. **A fast horizon detector** (Zardoua et al., 2024, arXiv:2110.13694)
   - Maritime video (аналогично - граница лёд/стены)
   - Line fitting on filtered edges
   - Real-time capable

3. **Real-time line detection** (Fernandes & Oliveira, 2008)
   - Improved Hough transform voting
   - Software real-time on large images
   - GPU implementation available (Shapira & Hassner, 2018)

**OpenCV Implementation:**
```python
# Basic approach
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
# Extract horizon angle from detected lines
```

---

## Категория 2: Camera Pose Estimation

### Vanishing Points (SOTA для structured scenes)

1. **Camera pose estimation in soccer scenes** (Kashany & Pourreza, 2010)
   - Two vanishing points → camera rotation matrix
   - Extract pan/tilt directly
   - Mathematical proof provided

2. **A Fast and Simple Method** (Guo et al., 2022, arXiv:2208.08295)
   - **Single vanishing point** sufficient for absolute orientation
   - Closed-form solution
   - Real-time: 25.6 FPS on embedded (Gumstix)

3. **Monocular-based pose estimation** (IEEE 2014)
   - Vanishing geometry for indoor images
   - RANSAC + Levenberg-Marquardt not needed
   - Efficient for real-time

**Key Insight:** Для катка достаточно одного vanishing point (горизонт) чтобы получить roll/pitch!

---

## Категория 3: Sports Field Registration

### Hockey/Ice Rink Specific 🏒

1. **Boundary-aware semantic segmentation** (Wang et al., 2026, CVIU)
   - **ICE HOCKEY** rink registration
   - Rink boundary as new segmentation class
   - Boundary-aware loss + dynamic class weighting
   - Datasets: NHL, SHL
   - **Code available?** Paper mentions dataset

2. **BenderNet and RingerNet** (Walters et al., Waterloo)
   - **Efficient DNN for ice rink line segmentation**
   - Lightweight: suitable for real-time
   - Dilated depthwise separable convolutions
   - Homography from segmented lines

3. **Object Detection for Ice Surface Localisation** (IRCOBI 2024)
   - **YOLOv5** for landmark detection on ice
   - F1 = 0.99, mAP = 98.5%
   - Homography from 4+ landmarks
   - IoU = 0.96 for field localisation
   - **GitHub available?**

### General Sports (Soccer/Basketball)

1. **PnLCalib** (Gutiérrez-Pérez & Agudo, 2026, CVIU)
   - Points + Lines optimization
   - 3D soccer field model
   - Outperforms SoccerNet-Calibration
   - **GitHub: https://github.com/mguti97/PnLCalib** ⭐

2. **TVCalib** (Theiner & Ewerth, 2023, WACV)
   - Soccer field registration
   - Segment-based reprojection loss
   - Camera pose + focal length estimation

3. **Real-Time Camera Pose Estimation** (Citraro et al.)
   - Fully-convolutional deep architecture
   - Field lines + player locations
   - Soccer, basketball, volleyball benchmarks
   - **Real-time** ⭐

4. **Enhancing Soccer Camera Calibration** (Falaleev, 2024, arXiv:2410.07401)
   - Keypoint exploitation (line-line, line-conic intersections)
   - Winner: SoccerNet Camera Calibration Challenge 2023
   - **GitHub: https://github.com/NikolasEnt/soccernet-calibration-sportlight** ⭐

---

## Категория 4: Visual-Inertial Odometry (VIO)

### Smartphone VIO Libraries

1. **VINS-Mobile** (HKUST Aerial Robotics)
   - **GitHub: https://github.com/HKUST-Aerial-Robotics/VINS-Mobile** ⭐
   - Monocular VIO on iOS devices
   - Real-time SLAM with loop closure
   - AR demo + drone control
   - **Linux version: VINS-Mono**

2. **python-droneposelib** (marcusvaltonen, 2021)
   - **GitHub: https://github.com/marcusvaltonen/python-droneposelib** ⭐
   - Python VIO library
   - MIT license
   - For drone navigation

3. **LRPL-VIO** (Zheng et al., 2024, Sensors)
   - Lightweight, robust
   - Point + line features
   - Low computational cost

4. **Android VIO Tester** (AaltoML)
   - **GitHub: https://github.com/AaltoML/android-viotester** ⭐
   - Benchmark app for Android
   - Data collection + camera calibration
   - Recordings can be processed offline

**Key Insight:** VIO даёт полный 6DOF pose (x,y,z, roll, pitch, yaw) в реальном времени!

---

## Категория 5: Figure Skating Specific

1. **Multi-view 3D reconstruction** (Tian et al., 2020)
   - Multi-technology correction framework
   - Temporal information + multi-perspective + trajectory smoothness
   - **Multi-camera system** (не подходит для нашего случая)

2. **Multi-view 3D** (Zeng et al., 2023)
   - Voxel-based recovery
   - Trajectory separable error rectification
   - Also multi-camera

3. **Automatic Edge Error Judgment** (IEEE 2025)
   - **3D pose from inertial sensors** ⭐
   - IMU-based edge detection
   - Relevant for sensor fusion approach!

**Gap:** Нет monocular camera calibration для figure skating - все используют multi-camera.

---

## Implementation Recommendations

### MVP Approach (Simplest Working Thing)

```
1. Horizon Line Detection (Hough Transform)
   ├── OpenCV cv2.HoughLines
   ├── Filter by angle (horizontal lines only)
   └── Extract average angle = camera roll

2. Compensate Poses
   ├── Rotate poses by -horizon_angle
   └── Now "vertical" = true vertical!

3. Visualize XYZ Axes
   ├── Draw axes in corner with compensation
   ├── Z = perpendicular to horizon
   └── X = parallel to horizon
```

### Production Approach (If MVP works)

```
1. Sports Field Registration (borrow from hockey/soccer)
   ├── Detect rink markings (lines, circles, crease)
   ├── Compute homography to template rink
   └── Extract full camera pose

2. VIO Integration (optional, for moving camera)
   ├── VINS-Mobile on smartphone
   ├── Fuse visual + IMU data
   └── Get 6DOF pose in real-time

3. Fallback Hierarchy
   ├── Primary: Field markings (most accurate)
   ├── Secondary: Horizon line (always available)
   └── Tertiary: Gravity prior (assume level)
```

---

## GitHub Repositories to Explore

| Repo | Stars | Focus | Python? |
|------|-------|-------|--------|
| mguti97/PnLCalib | - | Sports field calibration | Likely |
| NikolasEnt/soccernet-calibration-sportlight | - | Soccer camera calib | Yes |
| HKUST-Aerial-Robotics/VINS-Mobile | 1353 | Monocular VIO iOS | C++ |
| marcusvaltonen/python-droneposelib | 10 | Python VIO | **Yes** ⭐ |
| AaltoML/android-viotester | 51 | Android VIO benchmark | Java |
| mokumus/OpenCV-Homography | 3 | Soccer homography | **Yes** ⭐ |

---

## Key Papers to Read

1. **Guo et al. (2022)** "A Fast and Simple Method for Absolute Orientation"
   - arXiv:2208.08295
   - Single vanishing point = sufficient!

2. **Citraro et al.** "Real-Time Camera Pose Estimation for Sports Fields"
   - Fully-convolutional architecture
   - Real-time capable

3. **Wang et al. (2026)** "Boundary-aware semantic segmentation for ice hockey rink"
   - CVIU 2026
   - **ICE HOCKEY specific!**

4. **Falaleev (2024)** "Enhancing Soccer Camera Calibration"
   - arXiv:2410.07401
   - SoccerNet 2023 winner

5. **IRCOBI (2024)** "Object Detection for Ice Surface Localisation"
   - YOLOv5 for ice landmarks
   - 98.5% mAP!

---

## Next Steps for Gemini Deep Research

**Что Exa НЕ нашёл:**
- Существующие библиотеки именно для figure skating camera calibration
- Python библиотеки для horizon detection (только papers)
- OpenCV tutorials specifically for ice rink detection
- Comparison of methods for moving camera in sports

**Вопросы для Gemini:**
1. Какие Python библиотеки существуют для camera pose estimation?
2. Как интегрировать IMU данные с видео (sensor fusion)?
3. Существуют ли готовые решения для ice rink detection?
4. Какой метод лучше: vanishing points vs homography vs deep learning?
5. Можно ли использовать ARKit/CoreMotion данные с iPhone?

---

## Citation Summary

**Total Papers Found:** 20
**With Code:** 6 GitHub repos
**Ice Hockey Specific:** 3 papers ⭐
**Real-time Capable:** 8 papers
**Python Implementations:** 4 repos

**Most Promising for MVP:**
1. **Horizon line + vanishing point** (classical, fast)
2. **YOLOv5 for landmarks** (98.5% accuracy on ice)
3. **python-droneposelib** (pure Python VIO)
