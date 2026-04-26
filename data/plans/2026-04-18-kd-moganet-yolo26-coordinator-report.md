# КООРДИНАТОРСКИЙ ОТЧЁТ: Проверка плана KD MogaNet-B → YOLO26-Pose

**Дата:** 2026-04-21
**План:** `data/plans/2026-04-18-kd-moganet-yolo26-plan.md`
**Статус:** Ожидание отчётов от агентов

---

## КРИТИЧЕСКИЕ ОБЛАСТИ ДЛЯ Проверки

### 1. Техническая осуществимость интеграции с Ultralytics
**Почему критично:** План полагается на кастомный `DistilPoseTrainer` который наследуется от `PoseTrainer` в Ultralytics. Если API изменилось или нет хуков для переопределения `compute_loss()` — вся архитектура KD рискует провалиться.

**Что нужно проверить:**
- Совместимость версии Ultralytics с YOLO26-Pose
- Наличие метода `compute_loss()` в `PoseTrainer` и его сигнатура
- Возможность добавления дополнительных каналов (sigma_x, sigma_y) к pose head во время обучения
- Потенциальные конфликты с внутренними механизмами оптимизации Ultralytics

**Владелец:** Technical Auditor

---

### 2. Корректность офлайн теплокарт (heatmaps)
**Почему критично:** Вся стратегия KD строится на **предварительном вычислении** теплокарт MogaNet-B (Task 13 step 0). Если форма тензора неверна, координатные системы не совпадают, или top-down cropping работает неправильно — student будет учиться на неверных целях.

**Что нужно проверить:**
- Фактическая форма выхода MogaNet-B (план предполагает (17, 48, 64), но это нужно Verify)
- Корректность GT bbox → crop → resize pipeline
- Совпадение координатных пространств (нормализованные [0,1] vs пиксельные)
- Потеря точности при float16 сжатии в HDF5
- Производительность I/O для HDF5 при батчевой загрузке

**Владелец:** Technical Auditor

---

### 3. Качество и репрезентативность данных обучения
**Почему критично:** План комбинирует 4 датасета (FineFS, FSAnno, AthletePose3D, COCO). Если данные низкого качества, имеют неправильные метки, или несбалансированы — даже идеальная KD не спасёт модель.

**Что нужно проверить:**
- **FineFS:** Реальное количество кадров после фильтрации (>= 5 keypoints), корректность 3D→2D проекции
- **FSAnno:** BLOCKER — доступность YouTube видео. Если недоступны, теряются ~21K кадров (6% тренировочного сета)
- **AthletePose3D:** Соответствие формата 17kp H3.6M, качество мульти-спорт обобщения
- **COCO mix:** Риск катастрофического забывания (план упоминает -38% AP при domain-only fine-tuning)
- **Аугментации:** Какие использует Ultralytics по умолчанию? Подходят ли они для figure skating?

**Владелец:** Data Strategist

---

### 4. Калибровка времени обучения и бюджет
**Почему критично:** Все оценки времени/стоимости основаны на **внутреннем коэффициенте** Ultralytics ("2.8s compute per 1000 images on RTX 4090"), который план признаёт НЕ надёжным (72× расхождение с реальностью). Без реальной калибровки есть риск:
- Превышения бюджета ($150)
- Недооценки времени → прерывание обучения на midway
- Слишком оптимистичный план → невозможность завершить все gated stages

**Что нужно проверить:**
- Реальная производительность YOLO26 на RTX 4090/5090 (epoch time)
- Стоимость часа на Vast.ai (актуальные цены)
- Время предвычисления теплокарт (план: 1.5-3h)
- Резерв для gated stages (Stage 2.5, 3.5) — могут ли они поместиться в бюджет?

**Владелец:** Budget Analyst

---

### 5. Потенциальные точки отказа (silent failures)
**Почему критично:** План имеет несколько мест, где ошибки могут быть **тихими** (неявными):
- MogaNet-B top-down inference может падать на экстремальных позах (inverted spins)
- FSAnno YouTube видео могут быть недоступны (нет проверки в плане)
- HDF5 может быть медленным при concurrency
- Ultralytics может игнорировать кастомный loss если неправильно переопределён
- Vast.ai unverified instance может умереть midway (нет стратегии recovery кроме checkpointing)

**Что нужно проверить:**
- Добавить ли в план явные проверки на каждом этапе?
- Нужен ли более агрессивный smoke test для Vast.ai?
- Что если calibration покажет 2× медленнее чем ожидалось?

**Владелец:** Risk Assessor

---

### 6. Корректность DistilPose loss реализации
**Почему критито:** План использует **MSRA unbiased encoding** для симуляции теплокарт от student. Это нестандартный подход — если реализовать неправильно, градиенты будут некорректными.

**Что нужно проверить:**
- Правильность формулы 2D Gaussian (MSRA unbiased encoding)
- Совпадение формы выхода с teacher heatmap shape
- Корректность Softplus для sigma (всегда положительный?)
- Тестируемость разных функциональных компонентов (unit tests)
- Производительность vectorized grid computation (может быть bottleneck)

**Владелец:** Technical Auditor

---

### 7. прогрессивная стратегия sizing (n → s → m)
**Почему критично:** План предполагает обучение только YOLO26n, затем переход к s/m только если n не достигнет AP >= 0.85. Это разумно для бюджета, но есть риск:
- Если YOLO26n застрянет на AP=0.84, мы потратим 100 epochs впустую
- Нет ясных критериев для "early stop" на n — надо ли ждать всех 100 epochs?

**Что нужно проверить:**
- Добавить ли промежуточные checkpoints для ранней оценки?
- Что если baseline YOLO26n уже имеет AP=0.82 — стоит ли вообще KD?
- Нужен ли ablation между n → s → m или сразу jump на m если n проваливается?

**Владелец:** Data Strategist + Budget Analyst (joint)

---

## МАППИНГ ОТЧЁТОВ АГЕНТОВ

### Technical Auditor
**Ожидаемые покрытия:**
- ✅ Area 1: Ultralytics API совместимость
- ✅ Area 2: Offline heatmaps корректность
- ✅ Area 6: DistilPose loss реализация

**Гaps если не покрыто:**
- Неизвестно, можно ли вообще интегрировать кастомный loss в Ultralytics
- Неизвестно, будет ли HDF5 I/O bottleneck при training time

---

### Data Strategist
**Ожидаемые покрытия:**
- ✅ Area 3: Качество датасетов (FineFS, FSAnno, AP3D, COCO)
- ✅ Area 7 (частично): Прогрессивная sizing стратегия

**Гaps если не покрыто:**
- BLOCKER на FSAnno (YouTube видео) — может ли план работать без них?
- Недостаточно аугментаций для figure skating специфических поз (inverted spins, fast rotations)

---

### Budget Analyst
**Ожидаемые покрытия:**
- ✅ Area 4: Калибровка времени и стоимости
- ✅ Area 7 (частично): Прогрессивная sizing — бюджетные implications

**Гaps если не покрыто:**
- Нет резервного плана если calibration покажет 2× медленнее
- Неизвестно сколько времени займёт pre-compute heatmaps на реальном GPU

---

### Risk Assessor
**Ожидаемые покрытия:**
- ✅ Area 5: Точки отказа (silent failures)
- ✅ Все области (cross-cutting): Edge cases, missing configs

**Гaps если не покрыто:**
- Не все точки отказа могут быть выявлены без глубокого анализа кода
- Missing fallback стратегии для critical failures

---

## ПЕРЕСЕКАЮЩИЕСЯ ПРОБЛЕМЫ

### Issue 1: Неопределённость в формулах DistilPose loss
**Области:** 2 (heatmaps), 6 (loss implementation)
**Проблема:** План использует MSRA unbiased encoding, но не даёт полных формул. Technical Auditor должен Verify:
- Совпадает ли `eps` в знаменателе с оригинальным DistilPose?
- Используется ли `Softplus(sigma)` или `sigma = Softplus(raw)`?
- Как обрабатываются visibility flags в heatmap generation?

**Рекомендация:** Technical Auditor должен привести полную формулу из источника (DistilPose repo) и сравнить с планом.

---

### Issue 2: FSAnno BLOCKER может сломать весь data pipeline
**Области:** 3 (данные), 4 (бюджет), 5 (риски)
**Проблема:** Если YouTube видео недоступны, план теряет ~21K кадров (6% train set). Data Strategist должен проверить доступность, Budget Analyst — пересчитать бюджет, Risk Assessor — оценить fallout.

**Рекомендация:** Data Strategist → проверить `video_sources.json` на доступность URL. Если >50% недоступны → исключить FSAnno из плана entirely.

---

### Issue 3: Vast.ai unverified instances могут добавить непредсказуемость
**Области:** 4 (бюджет), 5 (риски)
**Проблема:** Smoke test 15 min не гарантирует стабильность при 100+ часах обучения. Budget Analyst должен добавить contingency budget, Risk Assessor — более агрессивный recovery strategy.

**Рекомендация:** Budget Analyst → добавить 20% contingency на top of calibration-based estimate. Risk Assessor → предложить checkpoint sync to R2 для extra safety.

---

## ПРОПУЩЕННЫЕ ОБЛАСТИ (НЕОБХОДИМО ДОБАВИТЬ)

### Gap 1: Метрики качества для student model
**Почему не покрыто:** Ни один агент не специализируется на evaluation metrics.
**Проблема:** План использует AP, AP50, per-joint AP, но не уточняет:
- Какой порог для "success" на skating val? (план: AP >= 0.85 — откуда это число?)
- Нужен ли cross-domain test (AP3D)?
- Как измерить "gap" между teacher и student для gated Stage 3.5?

**Suggested agent:** Technical Auditor (добавить к checklist)

---

### Gap 2: Ablation стратегии для hyperparameters
**Почему не покрыто:** План пропускает Stage 2 (fine-tune ablation) но не даёт альтернатив если KD застрянет.
**Проблема:** Если Stage 3 KD converges на AP=0.80 — что дальше? Попробовать разные freeze depths? Другие lr? Больше epochs?

**Suggested agent:** Data Strategist (предложить fallback grid search)

---

### Gap 3: Производительность inference для production
**Почему не покрыто:** План фокусируется на training, но не измеряет inference speed после KD.
**Проблема:** Distilled model может быть медленнее из-за дополнительных вычислений (sigma). Нужен benchmark FPS на RTX 3050 Ti (production target).

**Suggested agent:** Budget Analyst (добавить inference benchmark в final evaluation)

---

### Gap 4: Совместимость формата ключевых точек (keypoint format)
**Почему не покрыто:** FineFS, FSAnno, AP3D используют разные 17kp formats (H3.6M vs COCO vs SMPL).
**Проблема:** План упоминает "FineFS 17kp → COCO 17kp mapping" но не детализирует. Ошибка в mapping → мусор в GT labels.

**Suggested agent:** Data Strategist (Verify все mappings перед conversion)

---

## СТРУКТУРА ФИНАЛЬНОГО СИНТЕЗА

Когда все агенты представят отчёты, я создам:

```
СИНТЕЗИРОВАННЫЙ ОТЧЁТ
=====================

1. КРИТИЧЕСКИЕ НАХОДКИ
   - [Технические] Блокеры для реализации
   - [Данные] Проблемы с качеством/доступностью
   - [Бюджет] Риски превышения стоимости
   - [Риски] Точки отказа

2. ПРИОРИТЕТИЗИРОВАННЫЕ ИЗМЕНЕНИЯ В ПЛАН
   - MUST FIX (критические ошибки)
   - SHOULD FIX (существенные улучшения)
   - NICE TO HAVE (оптимизации)

3. ОБНОВЛЁННАЯ ОЦЕНКА РЕСУРСОВ
   - Время (с calibration)
   - Стоимость (с contingency)
   - Вероятность успеха (%)

4. РЕКОМЕНДАЦИИ ПО СЛЕДУЮЩИМ ШАГАМ
   - Что делать до начала обучения
   - Что мониторить во время выполнения
   - План B если что-то пойдёт не так
```

---

## СЛЕДУЮЩИЕ ДЕЙСТВИЯ

1. **Technical Auditor:** Проверить Areas 1, 2, 6 → отчёт через code review + speculative execution
2. **Data Strategist:** Проверить Areas 3, 7 → отчёт через dataset sampling + format verification
3. **Budget Analyst:** Проверить Areas 4, 7 → отчёт через market research + calibration simulation
4. **Risk Assessor:** Проверить Area 5 + cross-cutting → отчёт через failure mode analysis

**Дедлайн:** Все отчёты должны быть представлены до начала Task 0 (калибровка на rented GPU).

---

**Статус координации:** Ожидание входящих отчётов от агентов.
