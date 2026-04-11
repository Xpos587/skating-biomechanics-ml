# Deep Research Prompt: Enhanced Visualization for Figure Skating Analysis

## Context

Разрабатываем AI-тренер для фигурного катания. Уже есть:
- **BlazePose** (33 ключевых точек) для детекции позы
- **Автоматическая сегментация** видео на элементы
- **Анализ биомеханики** (углы, скорость, ребро)
- **Субтитры VTT** с комментариями тренера на русском

## Проблема

Текущая визуализация не показывает скелет и не использует субтитры. Нужно создать comprehensive debug HUD.

## Research Questions

### 1. Skeleton Visualization Best Practices

**Что искать:**
- Как рисовать скелет человека поверх видео?
- Лучшие практики для overlay pose estimation на видео
- Цветовые схемы для скелета (чтобы было читаемо)
- Как визуализировать уверенность (confidence) каждой точки?
- OpenCV best practices для рисования линий/точек

**Ключевые слова:**
- "pose estimation skeleton visualization opencv"
- "draw skeleton overlay video mediapipe blazepose"
- "pose visualization color schemes"
- "human pose drawing best practices"

### 2. Kinematics Visualization

**Что искать:**
- Как визуализировать вектора скорости и ускорения?
- Как показать направление движения суставов?
- Цветовое кодирование для скорости (медленно→быстро)
- Как рисовать траектории (trails) движения?

**Ключевые слова:**
- "velocity vector visualization video"
- "motion visualization sports analysis"
- "acceleration vector arrow opencv"
- "trajectory drawing video computer vision"
- "sports biomechanics visualization"

### 3. Edge/Blade Detection for Skating

**Что искать:**
- Как детектировать какое ребро конька используется?
- Физика скольжения конька по льду
- Как определить: inside edge (внутрь), outside edge (наружу), flat (на плоскости)?
- Визуализация следа конька на льду (blade tracking)
- Особенности figure skating biomechanics

**Ключевые слова:**
- "figure skating edge detection biomechanics"
- "ice skate blade inside outside edge detection"
- "skating edge indicator foot pressure"
- "figure skating biomechanics edge change"
- "blade tracking ice skating computer vision"

### 4. Video HUD Design for Debugging

**Что искать:**
- Лучшие практики для HUD дизайна в компьютерном зрении
- Как организовать информацию на экране для удобства?
- Примеры sports analysis software (Dartfish, etc.)
- Как показывать много данных без загромождения экрана?

**Ключевые слова:**
- "sports analysis software HUD design"
- "video annotation overlay debug information"
- "computer vision visualization best practices"
- "opencv draw text overlay video"
- "sports telemetry display design"

### 5. Subtitle Integration for Video Analysis

**Что искать:**
- Как синхронизировать видео с субтитрами по timestamp?
- Лучшие практики отображения субтитров поверх видео
- Semi-transparent backgrounds для читаемости
- Многоязычные субтитры (русский + английский)

**Ключевые слова:**
- "opencv subtitle overlay video timestamp"
- "draw text video semi-transparent background"
- "vtt subtitle synchronization video"
- "video annotation text overlay best practices"

### 6. Figure Skating Specific

**Что искать:**
- Figure skating biomechanics research papers
- Как измеряют углы сгибания коленей/локтей в фигурном катании?
- Трёхкоротный прыжок (axel) биомеханика
- Как определяют качество исполнения элементов?
- Датасеты по фигурному катанию с аннотациями

**Ключевые слова:**
- "figure skating biomechanics research papers"
- "skating jump biomechanics analysis"
- "figure skating element classification dataset"
- "ice skating motion capture biomechanics"
- "figure skating edge change detection"

## Existing Codebase Context

Уже есть:
- **BLAZEPOSE_SKELETON_EDGES** - соединения между 33 точками
- **BKey enum** - индексы всех точек (LEFT_SHOULDER=11, LEFT_HIP=23, etc.)
- **SubtitleParser** - парсинг VTT файлов
- **compute_edge_indicator()** - детекция ребра
- **kinematics metrics** - airtime, jump height, knee angles

## Expected Output

1. **Best practices** для визуализации скелета на видео
2. **Примеры кода** для рисования векторов скорости
3. **Методы** детекции ребра конька
4. **HUD layout** примеры из sports analysis software
5. **Research papers** по фигурному катанию с биомеханикой
6. **Code snippets** которые можно адаптировать

## Goal

Создать comprehensive визуализацию с:
- Скелетом (33 точки)
- Векторами скорости (color-coded)
- Индикатором ребра (inside/outside/flat)
- Субтитрами (комментарии тренера)
- HUD с биомеханикой

Для отладки работы ElementSegmenter и улучшения качества детекции элементов.
