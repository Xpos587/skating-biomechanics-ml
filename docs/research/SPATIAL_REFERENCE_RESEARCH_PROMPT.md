# Deep Research Prompt: Spatial Reference & Camera Calibration for Skating Analysis

## Context

Разрабатываем ML-систему анализа биомеханики фигурного катания. У нас есть BlazePose 2D keypoints, но есть **критическая проблема**:

**Проблема:** Все измерения углов (наклон торса, колени, и т.д.) производятся относительно кадра камеры, а не относительно истинной вертикали/горизонтали. Если камера наклонена или движется - все данные неверны.

**Пример:** Камера наклонена на 10° вправо → фигурист кажется наклонённым на 10° влево, хотя он стоит прямо.

## Goal

Найти state-of-the-art методы для:
1. **Детекции плоскости льда** относительно камеры
2. **Компенсации движения/наклона камеры** в реальном времени
3. **Восстановления пространственной привязки** (XYZ осей) независимо от камеры

## Constraints

- **Hardware:** RTX 3050 Ti 4GB VRAM, CPU-only предпочтительно
- **Real-time:** <100ms per frame бюджет
- **Input:** Однокамерная видео съёмка (моно)
- **Environment:** Фигурный каток с ледяной площадкой, бортиками, трибунами

## Research Questions

### 1. Horizon Line Detection

**Что искать:**
- Как детектировать линию горизонта на видео катка?
- Методы: Hough transform, edge detection, semantic segmentation?
- Как отличить лёд от стен/трибун?
- Что делать если горизонт не виден (зум на фигуриста)?

**Ключевые слова:**
- "horizon line detection computer vision"
- "ice rink segmentation hockey skating"
- "hough line horizon detection sports"
- "camera calibration indoor sports"

### 2. Camera Pose Estimation

**Что искать:**
- Как оценить положение камеры (pose estimation) из одного кадра?
- Vanishing points, parallel lines, grid detection?
- Методы для динамических сцен (движущаяся камера)

**Ключевые слова:**
- "single image camera pose estimation"
- "vanishing point detection indoor"
- "camera calibration from natural scenes"
- "geometric camera calibration sports analysis"

### 3. IMU/Sensor Fusion

**Что искать:**
- Можно ли использовать данные сенсоров телефона (accelerometer, gyroscope)?
- Есть ли библиотеки для sensor + visual fusion?
- Как синхронизировать видео с IMU data?

**Ключевые слова:**
- "visual inertial odometry VIO"
- "IMU camera calibration smartphone"
- "sensor fusion accelerometer gyroscope pose"
- "mobile device camera orientation estimation"

### 4. Ice Surface Detection

**Что искать:**
- Как детектировать плоскость льда?
- Feature detection: circles, lines, markings on ice
- 3D reconstruction of ice rink from video

**Ключевые слова:**
- "ice hockey rink detection computer vision"
- "sports court segmentation soccer basketball"
- "planar surface detection video"
- "homography estimation sports fields"

### 5. Figure Skating Specific

**Что искать:**
- Существующие системы анализа фигурного катания - как они решают проблему?
- DVS (Dartfish, etc.) - как они делают spatial reference?
- Научные论文 по skating biomechanics - как они калибруют камеры?

**Ключевые слова:**
- "figure skating analysis camera calibration"
- "skating biomechanics 3D reconstruction"
- "ice skating motion capture calibration"
- "sports performance analysis spatial reference"

### 6. Real-time Libraries

**Что искать:**
- OpenCV функции для camera calibration
- Библиотеки: PyTorch3D, Open3D, COLMAP?
- Lightweight методы для real-time pose estimation

**Ключевые слова:**
- "opencv camera calibration real-time"
- "pytorch3d camera estimation"
- "lightweight slam algorithm"
- "monocular visual odometry python"

## Expected Output

### Структура отчёта

1. **Executive Summary** (3-5 bullets)
   - Критические находки
   - Рекомендуемый подход

2. **Horizon Detection** (если применимо)
   - Лучшие методы + accuracy
   - OpenCV code examples
   - Плюсы/минусы

3. **Camera Pose Estimation**
   - State-of-the-art методы
   - Real-time feasibility
   - Библиотеки/фреймворки

4. **Специфика фигурного катания**
   - Существующие решения
   - Уникальные проблемы катка

5. **Implementation Plan**
   - Рекомендуемый стек для MVP
   - Код примеры (если есть)
   - Оценка производительности

### Цитирование

- Арxiv ссылки
- GitHub репозитории
- API документация
- Сравнительные таблицы

## Success Criteria

**Хороший research найдёт:**
1. Как детектировать горизонт/плоскость льда robustly
2. Как компенсировать наклон камеры в real-time
3. Конкретные библиотеки/методы с код примерами
4. Trade-offs между точностью и скоростью

## Additional Context

**Текущий стек:**
- Python 3.11+
- OpenCV (cv2)
- MediaPipe BlazePose (33 keypoints)
- NumPy, SciPy

**Желаемый результат:**
```python
# Usage example
from skating_biomechanics_ml.utils import SpatialReferenceDetector

detector = SpatialReferenceDetector()
camera_pose = detector.estimate_pose(frame)  # Roll, pitch, yaw
compensated_poses = detector.compensate(poses, camera_pose)
draw_spatial_axes(frame, camera_pose)  # XYZ visualization
```

---

**Дата:** 2026-03-28
**Продолжительность:** 20-30 минут
**Глубина:** Comprehensive (все вышеперечисленные темы)
