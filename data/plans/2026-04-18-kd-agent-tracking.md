# ТРЕКИНГ ОТЧЁТОВ АГЕНТОВ

**План:** KD MogaNet-B → YOLO26-Pose
**Координатор:** Review Coordinator
**Последнее обновление:** 2026-04-21

---

## СТАТУС АГЕНТОВ

### Technical Auditor
**Зоны ответственности:**
- ✅ Area 1: Ultralytics API совместимость
- ✅ Area 2: Офлайн теплокарты корректность
- ✅ Area 6: DistilPose loss реализация
- ✅ Gap 1: Метрики качества (добавлено)

**Ожидаемые deliverables:**
- [ ] Проверка версии Ultralytics и YOLO26-Pose совместимости
- [ ] Verify форма выхода MogaNet-B (запуск на тестовом image)
- [ ] Полные формулы DistilPose loss из source repo
- [ ] Анализ HDF5 I/O производительности
- [ ] Unit test план для heatmap generation
- [ ] Критерии успеха (AP >= 0.85 - обоснование)

**Статус:** ⏳ Ожидание отчёта

---

### Data Strategist
**Зоны ответственности:**
- ✅ Area 3: Качество датасетов (FineFS, FSAnno, AP3D, COCO)
- ✅ Area 7 (частично): Прогрессивная sizing стратегия
- ✅ Gap 2: Ablation fallback стратегии
- ✅ Gap 4: Keypoint format mappings

**Ожидаемые deliverables:**
- [ ] Проверка FineFS формата (структура NPZ, mapping 17kp)
- [ ] **CRITICAL:** Проверка доступности YouTube видео из FSAnno `video_sources.json`
- [ ] Верификация AP3D формата (H3.6M 17kp)
- [ ] Анализ COCO mix стратегии (категории, кол-во изображений)
- [ ] Рекомендации по аугментациям для figure skating
- [ ] Fallback grid search если KD застрянет
- [ ] Verify всех keypoint mappings (FineFS→COCO, SMPL→COCO, H3.6M→COCO)

**Статус:** ⏳ Ожидание отчёта

---

### Budget Analyst
**Зоны ответственности:**
- ✅ Area 4: Калибровка времени и стоимости
- ✅ Area 7 (частично): Прогрессивная sizing — бюджетные implications
- ✅ Gap 3: Inference FPS benchmark

**Ожидаемые deliverables:**
- [ ] Актуальные цены Vast.ai на RTX 4090/5090 (апрель 2026)
- [ ] Анализ calibration стратегии (5 epochs на реальном GPU)
- [ ] Пересчёт всех оценок времени после calibration
- [ ] Contingency budget рекомендация (на основе риска)
- [ ] Анализ стоимости pre-compute heatmaps
- [ ] Inference FPS benchmark план (RTX 3050 Ti target)
- [ ] Обновлённая таблица бюджета с real numbers

**Статус:** ⏳ Ожидание отчёта

---

### Risk Assessor
**Зоны ответственности:**
- ✅ Area 5: Точки отказа (silent failures)
- ✅ Все области (cross-cutting): Edge cases, missing configs

**Ожидаемые deliverables:**
- [ ] Failure mode analysis для всех 16 tasks
- [ ] Silent failure detection checklist
- [ ] Vast.ai unverified instance risk mitigation
- [ ] HDF5 concurrency bottleneck анализ
- [ ] MogaNet-B top-down inference edge cases
- [ ] Улучшенная smoke test стратегия (15 min → ?)
- [ ] Checkpoint sync to R2 рекомендация
- [ ] Recovery strategy если instance dies midway

**Статус:** ⏳ Ожидание отчёта

---

## ПЕРЕСЕКАЮЩИЕСЯ ПРОБЛЕМЫ (TRACKING)

### Issue 1: DistilPose формулы неполные
**Области:** 2 (heatmaps) + 6 (loss)
**Owner:** Technical Auditor
**Зависимости:** Data Strategist (heatmap shape verification from MogaNet-B)
**Статус:** ⏳ Ожидание

---

### Issue 2: FSAnno BLOCKER
**Области:** 3 (данные) + 4 (бюджет) + 5 (риски)
**Owner:** Data Strategist (первичная проверка)
**Зависимости:**
- Budget Analyst → пересчитать бюджет если FSAnno excluded
- Risk Assessor → оценить fallout от потери 6% train set
**Статус:** ⏳ Ожидание

---

### Issue 3: Vast.ai непредсказуемость
**Области:** 4 (бюджет) + 5 (риски)
**Owner:** Budget Analyst (contingency) + Risk Assessor (recovery)
**Зависимости:** Нет (независимая оценка)
**Статус:** ⏳ Ожидание

---

## GAPs TRACKING

### Gap 1: Метрики качества (Technical Auditor)
**Проблема:** Откуда AP >= 0.85? Нужен ли cross-domain test?
**Статус:** ⏳ Назначено Technical Auditor
**Ожидание:** Обоснование threshold'ов + plan for cross-domain evaluation

---

### Gap 2: Ablation fallback (Data Strategist)
**Проблема:** Если KD застрянет на AP=0.80 — что дальше?
**Статус:** ⏳ Назначено Data Strategist
**Ожидание:** Grid search план (freeze depths, lr, epochs)

---

### Gap 3: Inference FPS (Budget Analyst)
**Проблема:** Distilled model может быть медленнее
**Статус:** ⏳ Назначено Budget Analyst
**Ожидание:** Benchmark методология + target FPS for RTX 3050 Ti

---

### Gap 4: Keypoint mapping (Data Strategist)
**Проблема:** FineFS/AP3D/FSAnno разные 17kp formats
**Статус:** ⏳ Назначено Data Strategist
**Ожидание:** Verify все mappings + unit tests для conversion

---

## СИНТЕЗ ФИНАЛЬНОГО ОТЧЁТА (CHECKLIST)

Когда все агенты представят отчёты:

### Фаза 1: Сбор
- [ ] Получить отчёт от Technical Auditor
- [ ] Получить отчёт от Data Strategist
- [ ] Получить отчёт от Budget Analyst
- [ ] Получить отчёт от Risk Assessor

### Фаза 2: Картирование
- [ ] Map находки к 7 критическим областям
- [ ] Map находки к 4 gaps
- [ ] Идентифицировать uncovered areas

### Фаза 3: Приоритизация
- [ ] Классифицировать находки: MUST/SHOULD/NICE
- [ ] Identify блокеры для Task 0 (калибровка)
- [ ] Identify блокеры для Task 13 (KD training)

### Фаза 4: Ресурсная оценка
- [ ] Обновить время (с calibration data)
- [ ] Обновить стоимость (с contingency)
- [ ] Оценить вероятность успеха (%)

### Фаза 5: Рекомендации
- [ ] Что сделать ПЕРЕД началом обучения
- [ ] Что мониторить во время выполнения
- [ ] Plan B для критических failures

### Фаза 6: Финальный отчёт
- [ ] Написать синтезированный отчёт на русском
- [ ] Создать updated plan с изменениями
- [ ] Подготовить exec summary для пользователя

---

## КОММУНИКАЦИЯ

**Формат отчётов:** Markdown с чёткой структурой:
1. Критические находки (что сломано/рискованно)
2. Рекомендации (как исправить)
3. Open questions (что требует дополнительного исследования)

**Тон:** Объективный, основанный на фактах, без speculation

**Дедлайн:** Все отчёты должны быть представлены до начала Task 0

---

## СЛЕДУЮЩИЕ ШАГИ

1. ⏳ Ожидание входящих отчётов от всех 4 агентов
2. 📊 Map находки к критическим областям
3. 🔍 Identify gaps и cross-cutting concerns
4. 📝 Синтезировать финальный отчёт
5. ✅ Подготовить updated plan с изменениями

---

**Текущий статус:** 🔵 Ожидание входящих отчётов (0/4 агентов представили)
