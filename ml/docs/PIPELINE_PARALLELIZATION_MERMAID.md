# Pipeline Parallelization - Visual Diagrams

## Current Pipeline (Sequential)

```mermaid
gantt
    title Current Pipeline (Sequential) - Total: 12.0s
    dateFormat X
    axisFormat %L

    section Extraction
    RTMO Inference       :crit, 5.6s, rtm
    Gap Filling          :0.8s, after rtm, gap
    Normalization        :0.3s, after gap, norm
    Smoothing            :0.5s, after norm, smooth

    section Analysis
    3D Lifting           :1.5s, after smooth, lift
    Phase Detection      :0.8s, after smooth, phase
    Metrics              :1.2s, after phase, metrics
    Reference Load       :after metrics, ref
    DTW Alignment        :0.9s, after ref, dtw
    Recommendations      :0.4s, after dtw, rec
```

## Proposed Pipeline (Parallel)

```mermaid
gantt
    title Proposed Pipeline (Parallel) - Total: 10.0s
    dateFormat X
    axisFormat %L

    section Extraction
    RTMO Inference       :crit, 5.6s, rtm
    Gap Filling          :0.8s, after rtm, gap

    section Parallel 1
    Normalization        :0.3s, after gap, norm
    Smoothing            :0.5s, after gap, smooth

    section Parallel 2
    3D Lifting           :1.5s, after norm lift, lift
    Phase Detection      :0.8s, after smooth phase, phase

    section Parallel 3
    Metrics              :1.2s, after phase metrics, metrics
    Reference Load       :after phase ref, ref
    Physics              :0.9s, after lift physics, physics

    section Parallel 4
    DTW Alignment        :0.9s, after metrics dtw, dtw
    Recommendations      :0.4s, after metrics rec, rec
```

## Critical Path Analysis

```mermaid
graph TD
    A[Video Input] -->|5.6s| B[RTMO Inference]
    B -->|0.8s| C[Gap Filling]
    C -->|0.3s| D[Normalization]
    C -->|0.5s| E[Smoothing]
    D --> F{Parallel Branch}
    E --> F
    F -->|1.5s| G[3D Lifting]
    F -->|0.8s| H[Phase Detection]
    G -->|0.9s| I[Physics]
    H -->|1.2s| J[Metrics]
    J -->|0.9s| K[DTW Alignment]
    K -->|0.4s| L[Recommendations]
    I --> M[Report]
    L --> M

    style A fill:#FFE6E6
    style B fill:#FFE6E6
    style F fill:#FFFFE6
    style G fill:#FFFFE6
    style H fill:#FFFFE6
    style M fill:#FFE6E6
```

## Parallelization Opportunities

```mermaid
graph LR
    subgraph "Sequential (Cannot Parallelize)"
        A1[RTMO Inference<br/>5.6s, 47%<br/>GPU-bound]
        A2[Gap Filling<br/>0.8s, 7%<br/>Data dependency]
    end

    subgraph "Independent Stages (Can Parallelize)"
        B1[Normalization<br/>0.3s, 3%]
        B2[Smoothing<br/>0.5s, 4%]
        B3[3D Lifting<br/>1.5s, 13%]
        B4[Phase Detection<br/>0.8s, 7%]
        B5[Metrics<br/>1.2s, 10%]
        B6[Reference Load<br/>I/O-bound]
        B7[Physics<br/>0.9s, 8%]
        B8[DTW Alignment<br/>0.9s, 8%]
        B9[Recommendations<br/>0.4s, 3%]
    end

    A1 --> B1
    A2 --> B2

    style A1 fill:#FFE6E6
    style A2 fill:#FFE6E6
    style B1 fill:#E6FFE6
    style B2 fill:#E6FFE6
    style B3 fill:#E6FFE6
    style B4 fill:#E6FFE6
    style B5 fill:#E6FFE6
    style B6 fill:#E6FFE6
    style B7 fill:#E6FFE6
    style B8 fill:#E6FFE6
    style B9 fill:#E6FFE6
```

## Performance Comparison

```mermaid
graph BAR
    title Single Video Performance (364 frames)
    x-axis ["Current", "Priority 1", "Priority 1+3", "All"]
    y-axis "Time (seconds)" 0 --> 12
    bar [12.0, 10.0, 5.0, 4.2]
```

## Batch Processing Scaling

```mermaid
graph LINE
    title Batch Processing Speedup
    x-axis "Workers" 1 --> 8
    y-axis "Speedup (x)" 0 --> 8
    line [1.0, 2.0, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0]
```

## Implementation Priority Matrix

```mermaid
quadrantChart
    title Implementation Priority Matrix
    x-axis "Low Impact" --> "High Impact"
    y-axis "High Effort" --> "Low Effort"

    Priority_1: [0.8, 0.2]
    Priority_2: [0.9, 0.6]
    Priority_3: [0.7, 0.5]
    Priority_4: [0.4, 0.3]
    Priority_5: [0.2, 0.2]
    Priority_6: [0.6, 0.9]
```

## Risk vs. Impact Matrix

```mermaid
quadrantChart
    title Risk vs. Impact Matrix
    x-axis "Low Impact" --> "High Impact"
    y-axis "High Risk" --> "Low Risk"

    Enhanced_Async: [0.7, 0.2]
    Batch_Processing: [0.9, 0.5]
    Multi_GPU: [0.8, 0.5]
    CPU_Optimization: [0.4, 0.3]
    IO_Prefetching: [0.2, 0.2]
    Batch_RTMO: [0.6, 0.9]
```

## Multi-GPU Architecture

```mermaid
graph TB
    subgraph "Video Input"
        A[Video File<br/>364 frames]
    end

    subgraph "Chunk Splitter"
        B[Split into 2 chunks<br/>Chunk 1: 0-181<br/>Chunk 2: 182-363]
    end

    subgraph "GPU 0"
        C1[Process Chunk 1<br/>2.8s]
    end

    subgraph "GPU 1"
        C2[Process Chunk 2<br/>2.8s]
    end

    subgraph "Merge"
        D[Merge Results<br/>0.1s]
    end

    subgraph "Rest of Pipeline"
        E[Analysis Stages<br/>4.2s]
    end

    A --> B
    B --> C1
    B --> C2
    C1 --> D
    C2 --> D
    D --> E

    style A fill:#FFE6E6
    style B fill:#FFFFE6
    style C1 fill:#E6F2FF
    style C2 fill:#E6F2FF
    style D fill:#FFFFE6
    style E fill:#FFE6E6
```

## Batch Processing Architecture

```mermaid
graph TB
    subgraph "Input"
        A[10 Videos<br/>120s sequential]
    end

    subgraph "Process Pool (4 workers)"
        B1[Worker 1<br/>Video 1, 2, 3]
        B2[Worker 2<br/>Video 4, 5, 6]
        B3[Worker 3<br/>Video 7, 8]
        B4[Worker 4<br/>Video 9, 10]
    end

    subgraph "Output"
        C[10 Reports<br/>30s parallel]
    end

    A --> B1
    A --> B2
    A --> B3
    A --> B4
    B1 --> C
    B2 --> C
    B3 --> C
    B4 --> C

    style A fill:#FFE6E6
    style B1 fill:#E6F2FF
    style B2 fill:#E6F2FF
    style B3 fill:#E6F2FF
    style B4 fill:#E6F2FF
    style C fill:#E6FFE6
```

## Timeline

```mermaid
gantt
    title Implementation Timeline
    dateFormat  YYYY-MM-DD
    section Week 1
    Priority 1 Enhanced Async    :a1, 2026-04-18, 2d
    Testing & Validation         :a2, after a1, 1d

    section Week 2-3
    Priority 2 Batch Processing  :b1, 2026-04-22, 4d
    Testing & Validation         :b2, after b1, 1d

    section Week 4
    Priority 3 Multi-GPU         :c1, 2026-04-29, 3d
    Testing & Validation         :c2, after c1, 1d

    section Optional
    Priority 4 CPU Optimization  :d1, 2026-05-06, 3d
    Priority 5 I/O Prefetching   :e1, 2026-05-09, 2d
```

---

**End of Visual Diagrams**
