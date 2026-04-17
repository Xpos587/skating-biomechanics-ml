# Parallelization & Performance Optimization - Implementation Plans

**Status:** Ready for Execution
**Total Duration:** 8-12 weeks (all phases)
**Target:** 80-100x overall speedup

---

## Overview

This document contains detailed implementation plans for optimizing the skating biomechanics ML pipeline across 4 phases. Each phase is independently executable and can be deployed incrementally.

### Performance Targets

| Phase | Target Speedup | Pipeline Time | Status |
|-------|---------------|---------------|--------|
| Phase 1 | 5-10x | 3.1s | Ready |
| Phase 2 | 10-20x | 0.6s | Ready |
| Phase 3 | 20-40x | 0.3s | Ready |
| Phase 4 | 40-100x | 0.15s | Ready |

**Cumulative: 12s → 0.15s (80x speedup)**

---

## Phase 1: Quick Wins (Week 1-2)

**Plan:** `2026-04-17-phase1-quick-wins.md`

### Goals
- NumPy vectorization for physics/metrics (10-50x speedup)
- Async R2 I/O with httpx (1.5-2x speedup)
- Batch ONNX inference for 3D lifting (1.5-2x speedup)

### Key Changes
- `analysis/physics_engine.py` - Vectorized CoM and inertia
- `utils/geometry.py` - Vectorized angle functions
- `utils/gap_filling.py` - Vectorized interpolation
- `pose_3d/onnx_extractor.py` - Batch inference
- `backend/app/storage.py` - Async I/O

### Success Criteria
- 5-10x overall speedup (12s → 2-3s)
- 13 new tests, all passing
- No regression in accuracy

---

## Phase 2: Multi-GPU & Pipeline Parallelism (Week 3-4)

**Plan:** `2026-04-17-phase2-multi-gpu-pipeline.md`

### Goals
- Multi-GPU batch pose extraction (2-3x speedup)
- Async pipeline with parallel stages (1.7x speedup)
- Profiling utilities for bottleneck identification

### Key Changes
- `device.py` - Multi-GPU detection and configuration
- `pose_estimation/multi_gpu_extractor.py` - NEW
- `pipeline.py` - Async analyze with parallel stages
- `utils/profiler.py` - NEW

### Success Criteria
- 10-20x overall speedup (12s → 0.6s)
- Multi-GPU speedup > 1.5x
- Async pipeline speedup > 1.2x
- 10 new tests, all passing

### Dependencies
- Phase 1 complete
- Multi-GPU system (or single GPU with simulation)

---

## Phase 3: Model Optimization (Month 2)

**Plan:** `2026-04-17-phase3-model-optimization.md`

### Goals
- TensorRT conversion (2-3x speedup)
- FP16/INT8 quantization (1.5-3x speedup)
- Distributed Vast.ai processing (5-15x speedup)

### Key Changes
- `optimization/tensorrt_converter.py` - NEW
- `optimization/quantization.py` - NEW
- `optimization/validation.py` - NEW
- `vastai/distributed.py` - NEW
- `gpu_server/server.py` - Chunk processing

### Success Criteria
- 20-40x overall speedup (12s → 0.3s)
- TensorRT speedup > 2x
- FP16 maintains > 99% accuracy
- 10 new tests, all passing

### Dependencies
- Phase 2 complete
- TensorRT 8.x+ installed
- Vast.ai API access

---

## Phase 4: Advanced Optimizations (Month 3)

**Plan:** `2026-04-17-phase4-advanced-optimizations.md`

### Goals
- Custom CUDA kernels for physics (10-20x speedup)
- Numba JIT for hot loops (1.5-3x speedup)
- WebSocket real-time progress streaming (UX improvement)

### Key Changes
- `cuda/physics.cu` - NEW
- `cuda/kernels.py` - NEW
- `analysis/physics_engine_cuda.py` - NEW
- `utils/numba_jit.py` - NEW
- `backend/app/routes/websockets.py` - NEW

### Success Criteria
- 40-100x overall speedup (12s → 0.15s)
- CUDA speedup > 10x over original
- WebSocket latency < 100ms
- 9 new tests, all passing

### Dependencies
- Phase 3 complete
- CUDA toolkit installed
- WebSocket client library

---

## Execution Strategy

### Recommended Approach

1. **Start with Phase 1** - Quick wins with minimal risk
2. **Profile after each phase** - Measure actual speedup
3. **Adjust targets** - Based on hardware constraints
4. **Deploy incrementally** - Each phase is production-ready

### Execution Options

**Option 1: Subagent-Driven Development (Recommended)**
```bash
# Use subagent-driven-development skill
# Each task executed by fresh subagent
# Fast iteration with code review between tasks
```

**Option 2: Inline Execution**
```bash
# Use executing-plans skill
# Execute tasks in current session
# Batch execution with checkpoints
```

### Testing Strategy

```bash
# Run all tests
uv run pytest ml/tests/ backend/tests/ -v

# Run specific phase tests
uv run pytest ml/tests/analysis/test_physics_engine.py -v
uv run pytest ml/tests/benchmark/ -v -s --benchmark-only

# Run with markers
uv run pytest ml/tests/ -v -m "not cuda"  # Skip CUDA tests
uv run pytest ml/tests/ -v -m "tensorrt"  # Only TensorRT tests
```

---

## Performance Baseline

### Current Performance (RTX 3050 Ti)

| Video Length | Frames | Processing Time | Real-time Factor |
|--------------|--------|-----------------|------------------|
| 5s | 150 | 4.2s | 0.83x |
| 15s | 450 | 12.0s | 0.80x |
| 30s | 900 | 24.5s | 0.78x |

### Target Performance (After All Phases)

| Video Length | Frames | Processing Time | Real-time Factor |
|--------------|--------|-----------------|------------------|
| 5s | 150 | 0.05s | **100x** |
| 15s | 450 | 0.15s | **100x** |
| 30s | 900 | 0.30s | **100x** |

---

## Risk Assessment

### High-Risk Items

| Risk | Mitigation | Phase |
|------|------------|-------|
| TensorRT conversion fails | Keep ONNX fallback | 3 |
| Multi-GPU VRAM fragmentation | Dynamic batch sizing | 2 |
| CUDA kernel bugs | Extensive testing | 4 |
| Accuracy drop from quantization | Validate > 99% | 3 |

### Rollback Strategy

Each phase can be independently rolled back:

```bash
# Revert specific phase
git revert <phase-commit-range>

# Or disable via environment
export SKATING_DISABLE_CUDA=1
export SKATING_DISABLE_TENSORRT=1
export SKATING_DISABLE_MULTIGPU=1
```

---

## Success Metrics

### Phase Completion Criteria

- [ ] All tests pass
- [ ] Performance targets met
- [ ] No regression in accuracy
- [ ] Documentation updated
- [ ] Code review approved

### Final Success Criteria

- [ ] 80-100x overall speedup achieved
- [ ] Sub-second processing for 15s videos
- [ ] Real-time preview capability
- [ ] Cost-effective scaling (<$0.05/video)
- [ ] 100% test coverage maintained

---

## Dependencies

### Required Software

- Python 3.11+
- CUDA 12.x (for GPU support)
- TensorRT 8.x+ (Phase 3)
- ONNX Runtime 1.16+
- NumPy, SciPy, pytest

### Optional Software

- CuPy or PyCUDA (Phase 4)
- Numba (Phase 4)
- NVIDIA nvidia-smi (for GPU detection)

### Hardware Requirements

- Minimum: RTX 3050 Ti (6GB VRAM)
- Recommended: RTX 3060+ (12GB VRAM)
- Multi-GPU: 2x RTX 3060+ (Phase 2)

---

## Next Steps

1. **Review all plans** - Read through each phase plan
2. **Choose execution approach** - Subagent or inline
3. **Start with Phase 1** - Execute tasks sequentially
4. **Measure progress** - Run benchmarks after each phase
5. **Adjust as needed** - Based on actual hardware performance

---

## Contact & Support

For questions or issues during implementation:

- Review the spec: `docs/specs/2026-04-17-parallelization-design.md`
- Check research findings: `docs/research/`
- Run benchmarks: `uv run pytest ml/tests/benchmark/ -v -s`

**Remember:** Each phase is independently valuable. Even completing just Phase 1 yields 5-10x speedup with minimal risk.
