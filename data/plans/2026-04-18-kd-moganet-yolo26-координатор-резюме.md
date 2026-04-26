# РЕЗЮМЕ КООРДИНАТОРА (краткая версия)

**План:** KD MogaNet-B → YOLO26-Pose
**Дата:** 2026-04-21
**Статус:** Готов к приёму отчётов от агентов

---

## 7 КРИТИЧЕСКИХ ОБЛАСТЕЙ

| # | Область | Почему критично | Агент |
|---|---------|-----------------|-------|
| 1 | **Ultralytics API** | Кастомный `DistilPoseTrainer` может не сработать если API изменилось | Technical Auditor |
| 2 | **Офлайн теплокарты** | Неверная форма/формат → student учится на мусоре | Technical Auditor |
| 3 | **Качество данных** | FineFS/FSAnno/AP3D/COCO — если метки неверны, KD не спасёт | Data Strategist |
| 4 | **Бюджет и время** | Все оценки основаны на ненадёжном коэффициенте (72× ошибка) | Budget Analyst |
| 5 | **Точки отказа** | YouTube видео, HDF5 I/O, Vast.ai стабильность — silent killers | Risk Assessor |
| 6 | **DistilPose loss** | MSRA encoding — реализовать неправильно = некорректные градиенты | Technical Auditor |
| 7 | **Прогрессивный sizing** | n→s→m переходы — какие критерии для early stop? | Data Strategist + Budget Analyst |

---

## ПЕРЕСЕКАЮЩИЕСЯ ПРОБЛЕМЫ

### 1. Формулы DistilPose не полные
- **Области:** 2 (heatmaps), 6 (loss)
- **Проблема:** План не даёт полных формул для MSRA unbiased encoding
- **Действие:** Technical Auditor → Verify из DistilPose repo

### 2. FSAnno BLOCKER
- **Области:** 3 (данные), 4 (бюджет), 5 (риски)
- **Проблема:** YouTube видео могут быть недоступны → потеря 6% train set
- **Действие:** Data Strategist → проверить доступность URL в `video_sources.json`

### 3. Vast.ai непредсказуемость
- **Области:** 4 (бюджет), 5 (риски)
- **Проблема:** Unverified instance может умереть midway
- **Действие:** Budget Analyst → добавить 20% contingency, Risk Assessor → sync to R2

---

## 4 ПРОПУЩЕННЫХ ОБЛАСТИ

| Gap | Проблема | Почему не покрыто | Кто должен |
|-----|----------|-------------------|------------|
| **1** | **Метрики успеха** | Ни один агент не специализируется на evaluation | Technical Auditor |
| **2** | **Ablation fallback** | План пропускает Stage 2, нет plan B если KD застрянет | Data Strategist |
| **3** | **Inference FPS** | План не измеряет скорость после蒸馏 | Budget Analyst |
| **4** | **Keypoint mapping** | FineFS/AP3D/FSAnno используют разные 17kp formats | Data Strategist |

---

## СТРУКТУРА ФИНАЛЬНОГО СИНТЕЗА

Когда все агенты представят:

```
1. КРИТИЧЕСКИЕ НАХОДКИ
   ├─ Технические блокеры
   ├─ Проблемы с данными
   ├─ Бюджетные риски
   └─ Точки отказа

2. ПРИОРИТЕТИЗИРОВАННЫЕ ИЗМЕНЕНИЯ
   ├─ MUST FIX (критические ошибки)
   ├─ SHOULD FIX (существенные улучшения)
   └─ NICE TO HAVE (оптимизации)

3. ОБНОВЛЁННАЯ ОЦЕНКА РЕСУРСОВ
   ├─ Время (с calibration)
   ├─ Стоимость (с contingency)
   └─ Вероятность успеха (%)

4. РЕКОМЕНДАЦИИ ПО СЛЕДУЮЩИМ ШАГАМ
   ├─ До начала обучения
   ├─ Во время выполнения
   └─ Plan B при проблемах
```

---

## СЛЕДУЮЩИЕ ДЕЙСТВИЯ

- [ ] Technical Auditor → Проверить Areas 1, 2, 6
- [ ] Data Strategist → Проверить Areas 3, 7
- [ ] Budget Analyst → Проверить Areas 4, 7
- [ ] Risk Assessor → Проверить Area 5 + cross-cutting

**Дедлайн:** До Task 0 (калибровка на rented GPU)

---

**Статус:** 🔵 Ожидание входящих отчётов
