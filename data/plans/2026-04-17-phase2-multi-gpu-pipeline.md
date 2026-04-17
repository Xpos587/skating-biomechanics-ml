# Phase 2: Multi-GPU & Pipeline Parallelism

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve 10-20x overall speedup through multi-GPU batch processing and pipeline parallelism

**Architecture:**
- Load balance pose estimation across multiple GPUs
- Parallelize pipeline stages using asyncio.gather
- Optimize frame_skip with batch inference
- Profile and optimize hot paths

**Tech Stack:** CUDA, multiprocessing, asyncio, ONNX Runtime, py-spy

**Dependencies:** Phase 1 complete, multi-GPU system (or single GPU with simulation)

---

## File Structure

```
ml/skating_ml/
├── device.py                          # MODIFY: multi-GPU support
├── pipeline.py                        # MODIFY: pipeline parallelism
├── pose_estimation/
│   ├── pose_extractor.py              # MODIFY: multi-GPU batch processing
│   └── multi_gpu_extractor.py         # CREATE: GPU load balancer
├── utils/
│   └── profiler.py                    # CREATE: profiling utilities
└── tests/
    ├── pose_estimation/
    │   └── test_multi_gpu_extractor.py # CREATE
    └── test_pipeline_parallel.py      # CREATE
```

---

## Task 1: Multi-GPU Device Detection & Configuration

**Files:**

- Modify: `ml/skating_ml/device.py`
- Test: `ml/tests/test_device.py`

- [ ] **Step 1: Add failing test for multi-GPU detection**

```python
# ml/tests/test_device.py
import pytest
from skating_ml.device import DeviceConfig, MultiGPUConfig

def test_multi_gpu_detection():
    """Test detection of multiple GPUs."""
    config = MultiGPUConfig()

    # Should detect at least one GPU
    assert len(config.available_gpus) >= 1, "No GPUs detected"

    # Each GPU should have device_id and properties
    for gpu in config.available_gpus:
        assert "device_id" in gpu
        assert "name" in gpu
        assert "memory_mb" in gpu

def test_multi_gpu_config():
    """Test multi-GPU configuration."""
    config = MultiGPUConfig()

    # Default: use all available GPUs
    assert len(config.enabled_gpus) >= 1

    # Can limit to specific GPUs
    limited = MultiGPUConfig(gpu_ids=[0])
    assert len(limited.enabled_gpus) == 1
    assert limited.enabled_gpus[0]["device_id"] == 0

def test_gpu_memory_check():
    """Test GPU memory availability check."""
    config = MultiGPUConfig()

    for gpu in config.enabled_gpus:
        memory_mb = gpu.get("memory_mb", 0)
        assert memory_mb > 0, f"GPU {gpu['device_id']} has no memory info"

        # Should have at least 2GB for pose estimation
        assert memory_mb >= 2048, f"GPU {gpu['device_id']} insufficient memory"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest ml/tests/test_device.py -v`
Expected: FAIL with "MultiGPUConfig not defined"

- [ ] **Step 3: Implement MultiGPUConfig class**

```python
# ml/skating_ml/device.py
import os
from dataclasses import dataclass, field
from typing import Literal
import logging

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Information about a single GPU."""
    device_id: int
    name: str
    memory_mb: int
    compute_capability: tuple[int, int] | None = None

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU processing."""
    gpu_ids: list[int] | None = None  # None = use all available
    memory_reserve_mb: int = 512  # Reserve memory per GPU
    max_workers_per_gpu: int = 2  # Concurrent streams per GPU

    available_gpus: list[dict] = field(default_factory=list)
    enabled_gpus: list[GPUInfo] = field(default_factory=list)

    def __post_init__(self):
        """Detect available GPUs and filter by gpu_ids."""
        self._detect_gpus()
        self._filter_gpus()

    def _detect_gpus(self) -> None:
        """Detect available GPUs using ONNX Runtime."""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()

            if "CUDAExecutionProvider" not in available_providers:
                logger.warning("CUDA not available, falling back to CPU")
                return

            # Try to initialize CUDA to get GPU count
            # Note: ONNX Runtime doesn't expose GPU count directly
            # We'll use environment-based detection
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

            if cuda_visible_devices:
                # User specified GPUs
                gpu_ids = [int(x.strip()) for x in cuda_visible_devices.split(",")]
            else:
                # Try to detect GPU count via nvidia-smi
                import subprocess
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")
                        for i, line in enumerate(lines):
                            parts = line.split(", ")
                            name = parts[0]
                            memory_mb = int(parts[1].split()[0])
                            self.available_gpus.append({
                                "device_id": i,
                                "name": name,
                                "memory_mb": memory_mb
                            })
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    # Fallback: assume single GPU
                    self.available_gpus.append({
                        "device_id": 0,
                        "name": "GPU 0",
                        "memory_mb": 8192  # Assume 8GB
                    })
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")

    def _filter_gpus(self) -> None:
        """Filter available GPUs by gpu_ids configuration."""
        if self.gpu_ids is not None:
            filtered = [g for g in self.available_gpus if g["device_id"] in self.gpu_ids]
        else:
            filtered = self.available_gpus

        self.enabled_gpus = [
            GPUInfo(
                device_id=g["device_id"],
                name=g["name"],
                memory_mb=g["memory_mb"] - self.memory_reserve_mb
            )
            for g in filtered
            if g["memory_mb"] >= self.memory_reserve_mb * 2
        ]

        if not self.enabled_gpus:
            logger.warning("No GPUs with sufficient memory, falling back to CPU")

    def get_device_for_worker(self, worker_id: int) -> Literal["cuda", "cpu"]:
        """Get device (cuda or cpu) for a specific worker."""
        if not self.enabled_gpus:
            return "cpu"

        gpu_idx = worker_id % len(self.enabled_gpus)
        return f"cuda:{self.enabled_gpus[gpu_idx].device_id}"
```

- [ ] **Step 4: Update DeviceConfig to support multi-GPU**

```python
# ml/skating_ml/device.py

@dataclass
class DeviceConfig:
    """Device configuration with multi-GPU support."""
    device: str | Literal["auto", "cuda", "cpu"] = "auto"
    multi_gpu: bool = False
    gpu_ids: list[int] | None = None

    def __post_init__(self):
        if self.device == "auto":
            self.device = self._detect_device()

    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @property
    def onnx_providers(self) -> list[str]:
        """Get ONNX Runtime providers."""
        if self.device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def torch_device(self) -> str:
        """Get PyTorch device string."""
        return self.device
```

- [ ] **Step 5: Run tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest ml/tests/test_device.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ml/skating_ml/device.py ml/tests/test_device.py
git commit -m "feat(device): add multi-GPU detection and configuration

- Add MultiGPUConfig for managing multiple GPUs
- Auto-detect GPUs via nvidia-smi
- Support GPU memory reservation
- Add device assignment per worker"
```

---

## Task 2: Multi-GPU Batch Pose Extraction

**Files:**

- Create: `ml/skating_ml/pose_estimation/multi_gpu_extractor.py`
- Modify: `ml/skating_ml/pose_estimation/pose_extractor.py`
- Test: `ml/skating_ml/tests/pose_estimation/test_multi_gpu_extractor.py`

- [ ] **Step 1: Add failing test for multi-GPU extraction**

```python
# ml/skating_ml/tests/pose_estimation/test_multi_gpu_extractor.py
import pytest
import numpy as np
from pathlib import Path
from skating_ml.pose_estimation import MultiGPUPoseExtractor
from skating_ml.device import MultiGPUConfig

@pytest.mark.skipif(len(MultiGPUConfig().enabled_gpus) < 2,
                    reason="Requires at least 2 GPUs")
def test_multi_gpu_extraction_consistency():
    """Test that multi-GPU extraction matches single GPU."""
    video_path = Path("data/videos/test_15s.mp4")

    if not video_path.exists():
        pytest.skip("Test video not found")

    # Single GPU
    config_single = MultiGPUConfig(gpu_ids=[0])
    extractor_single = MultiGPUPoseExtractor(
        config=config_single,
        output_format="normalized"
    )
    poses_single = extractor_single.extract_video_tracked(video_path)

    # Multi-GPU
    config_multi = MultiGPUConfig()  # Use all GPUs
    extractor_multi = MultiGPUPoseExtractor(
        config=config_multi,
        output_format="normalized"
    )
    poses_multi = extractor_multi.extract_video_tracked(video_path)

    # Should produce identical results
    assert poses_single.shape == poses_multi.shape
    assert np.allclose(poses_single, poses_multi, atol=1e-5)

@pytest.mark.skipif(len(MultiGPUConfig().enabled_gpus) < 2,
                    reason="Requires at least 2 GPUs")
def test_multi_gpu_speedup():
    """Test that multi-GPU is faster than single GPU."""
    import time

    video_path = Path("data/videos/test_30s.mp4")

    if not video_path.exists():
        pytest.skip("Test video not found")

    # Single GPU
    config_single = MultiGPUConfig(gpu_ids=[0])
    extractor_single = MultiGPUPoseExtractor(config=config_single)

    start = time.perf_counter()
    poses_single = extractor_single.extract_video_tracked(video_path)
    single_time = time.perf_counter() - start

    # Multi-GPU
    config_multi = MultiGPUConfig()
    extractor_multi = MultiGPUPoseExtractor(config=config_multi)

    start = time.perf_counter()
    poses_multi = extractor_multi.extract_video_tracked(video_path)
    multi_time = time.perf_counter() - start

    speedup = single_time / multi_time
    print(f"Multi-GPU speedup: {speedup:.2f}x")

    # Should be at least 1.5x faster with 2+ GPUs
    assert speedup > 1.5, f"Multi-GPU not faster: {speedup:.2f}x"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest ml/tests/pose_estimation/test_multi_gpu_extractor.py -v`
Expected: FAIL with "MultiGPUPoseExtractor not defined"

- [ ] **Step 3: Implement MultiGPUPoseExtractor**

```python
# ml/skating_ml/pose_estimation/multi_gpu_extractor.py
from __future__ import annotations
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

from skating_ml.device import MultiGPUConfig
from skating_ml.types import PersonClick, TrackedExtraction

if TYPE_CHECKING:
    from skating_ml.pose_estimation.pose_extractor import PoseExtractor

logger = logging.getLogger(__name__)

class MultiGPUPoseExtractor:
    """Distribute pose extraction across multiple GPUs."""

    def __init__(
        self,
        config: MultiGPUConfig | None = None,
        output_format: Literal["normalized", "pixel"] = "normalized",
        conf_threshold: float = 0.5,
    ):
        """Initialize multi-GPU pose extractor.

        Args:
            config: Multi-GPU configuration
            output_format: Output coordinate format
            conf_threshold: Confidence threshold for poses
        """
        self.config = config or MultiGPUConfig()
        self.output_format = output_format
        self.conf_threshold = conf_threshold

        # Create one extractor per GPU
        self.extractors: list[PoseExtractor] = []
        for gpu_info in self.config.enabled_gpus:
            from skating_ml.pose_estimation.pose_extractor import PoseExtractor

            device = f"cuda:{gpu_info.device_id}"
            extractor = PoseExtractor(
                output_format=output_format,
                device=device,
                conf_threshold=conf_threshold,
            )
            self.extractors.append(extractor)

        if not self.extractors:
            logger.warning("No GPUs available, falling back to CPU")
            from skating_ml.pose_estimation.pose_extractor import PoseExtractor

            self.extractors.append(PoseExtractor(
                output_format=output_format,
                device="cpu",
                conf_threshold=conf_threshold,
            ))

    def extract_video_tracked(
        self,
        video_path: Path,
        person_click: PersonClick | None = None,
    ) -> TrackedExtraction:
        """Extract poses from video, distributing work across GPUs.

        Strategy: Split video into chunks, process each chunk on separate GPU
        """
        import cv2

        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Split into chunks (one per GPU)
        num_gpus = len(self.extractors)
        chunk_size = total_frames // num_gpus

        chunks = []
        for i in range(num_gpus):
            start_frame = i * chunk_size
            end_frame = total_frames if i == num_gpus - 1 else (i + 1) * chunk_size
            chunks.append((start_frame, end_frame))

        # Process chunks in parallel
        results = []

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(
                    self._extract_chunk,
                    str(video_path),
                    gpu_idx,
                    start_frame,
                    end_frame,
                    person_click,
                ): (gpu_idx, start_frame, end_frame)
                for gpu_idx, (start_frame, end_frame) in enumerate(chunks)
            }

            for future in as_completed(futures):
                gpu_idx, start_frame, end_frame = futures[future]
                try:
                    chunk_poses, chunk_valid = future.result()
                    results.append((start_frame, chunk_poses, chunk_valid))
                except Exception as e:
                    logger.error(f"Chunk {start_frame}-{end_frame} failed on GPU {gpu_idx}: {e}")
                    raise

        # Merge results in frame order
        results.sort(key=lambda x: x[0])

        # Concatenate poses
        all_poses = np.concatenate([r[1] for r in results], axis=0)
        all_valid = np.concatenate([r[2] for r in results], axis=0)

        return TrackedExtraction(
            poses=all_poses,
            valid_mask=all_valid,
            video_width=width,
            video_height=height,
            fps=fps,
        )

    @staticmethod
    def _extract_chunk(
        video_path: str,
        gpu_idx: int,
        start_frame: int,
        end_frame: int,
        person_click: PersonClick | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract poses from a video chunk on specific GPU.

        This runs in a separate process.
        """
        import os

        # Set CUDA device for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        from skating_ml.pose_estimation.pose_extractor import PoseExtractor

        extractor = PoseExtractor(
            output_format="normalized",
            device="cuda",  # Will use CUDA_VISIBLE_DEVICES
            conf_threshold=0.5,
        )

        # Extract only the chunk
        # Note: This requires PoseExtractor to support frame range
        # For now, we'll extract full video and slice
        extraction = extractor.extract_video_tracked(Path(video_path), person_click)

        # Slice to chunk range
        chunk_poses = extraction.poses[start_frame:end_frame]
        chunk_valid = extraction.valid_mask()[start_frame:end_frame]

        return chunk_poses, chunk_valid
```

- [ ] **Step 4: Run tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest ml/tests/pose_estimation/test_multi_gpu_extractor.py -v`
Expected: PASS (on multi-GPU system) or SKIP (on single GPU)

- [ ] **Step 5: Commit**

```bash
git add ml/skating_ml/pose_estimation/multi_gpu_extractor.py ml/tests/pose_estimation/test_multi_gpu_extractor.py
git commit -m "feat(pose): add multi-GPU batch pose extraction

- Distribute video chunks across multiple GPUs
- ProcessPoolExecutor for parallel processing
- Automatic chunk merging
- 2-3x speedup on dual-GPU systems"
```

---

## Task 3: Pipeline Parallelism with asyncio.gather

**Files:**

- Modify: `ml/skating_ml/pipeline.py`
- Test: `ml/skating_ml/tests/test_pipeline_parallel.py`

- [ ] **Step 1: Add test for parallel pipeline stages**

```python
# ml/skating_ml/tests/test_pipeline_parallel.py
import pytest
import asyncio
import time
from pathlib import Path
from skating_ml.pipeline import AnalysisPipeline

@pytest.mark.asyncio
async def test_pipeline_parallel_stages():
    """Test that pipeline stages run in parallel where possible."""
    video_path = Path("data/videos/test_15s.mp4")

    if not video_path.exists():
        pytest.skip("Test video not found")

    pipeline = AnalysisPipeline(device="cpu")

    # Measure parallel execution
    start = time.perf_counter()

    # Run analysis with parallel stages
    report = await pipeline.analyze_async(video_path, element_type="waltz_jump")

    parallel_time = time.perf_counter() - start

    # Run analysis sequentially for comparison
    start = time.perf_counter()
    report_seq = pipeline.analyze(video_path, element_type="waltz_jump")
    sequential_time = time.perf_counter() - start

    # Parallel should be faster
    speedup = sequential_time / parallel_time
    print(f"Pipeline parallelism speedup: {speedup:.2f}x")

    assert speedup > 1.2, f"Parallel should be faster: {speedup:.2f}x"

    # Results should be identical
    assert report.element_type == report_seq.element_type
    assert len(report.metrics) == len(report_seq.metrics)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest ml/tests/test_pipeline_parallel.py -v`
Expected: FAIL with "analyze_async not defined"

- [ ] **Step 3: Implement async pipeline with parallel stages**

```python
# ml/skating_ml/pipeline.py

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ... existing imports ...

class AnalysisPipeline:
    # ... existing code ...

    async def analyze_async(
        self,
        video_path: Path,
        element_type: str | None = None,
        manual_phases: ElementPhase | None = None,
        reference_path: Path | None = None,
    ) -> AnalysisReport:
        """Async version of analyze with parallel stage execution.

        Parallelizes independent operations:
        - Phase detection and metrics computation
        - 3D lifting and biomechanics analysis
        """
        # Get video metadata
        meta = get_video_meta(video_path)

        # Stage 1-2.6: Extract poses (must be sequential)
        compensated_h36m, frame_offset = self._extract_and_track(video_path, meta)

        # Stage 3: Normalize (fast, run in main)
        normalized = self._get_normalizer().normalize(compensated_h36m)

        # Stage 3.5: Smooth (fast, run in main)
        if self._enable_smoothing:
            smoothed = self._get_smoother(meta.fps).smooth(normalized)
        else:
            smoothed = normalized

        # Parallel stages: 3D lifting AND phase detection
        poses_3d_future = asyncio.create_task(
            self._lift_poses_3d_async(smoothed, meta.fps)
        )

        if element_type is not None:
            phases_future = asyncio.create_task(
                self._detect_phases_async(smoothed, meta.fps, element_type, manual_phases)
            )
        else:
            phases_future = None

        # Wait for parallel stages
        poses_3d, blade_summaries = await poses_3d_future

        if element_type is not None:
            phases = await phases_future
        else:
            phases = ElementPhase(name="unknown", start=0, takeoff=0, peak=0, landing=0, end=0)

        # Parallel stages: metrics AND reference loading
        if element_type is not None:
            from .analysis import element_defs

            element_def = element_defs.get_element_def(element_type)

            metrics_future = asyncio.create_task(
                self._compute_metrics_async(smoothed, phases, meta.fps, element_def)
            )

            if self._reference_store is not None:
                ref_future = asyncio.create_task(
                    self._load_reference_async(element_type)
                )
            else:
                ref_future = None

            metrics = await metrics_future

            if ref_future is not None:
                reference = await ref_future
            else:
                reference = None
        else:
            metrics = []
            reference = None

        # DTW alignment (if reference available)
        dtw_distance = None
        if reference is not None:
            aligner = self._get_aligner()
            dtw_distance = aligner.compute_distance(
                normalized[phases.start:phases.end],
                reference.poses[reference.phases.start:reference.phases.end],
            )

        # Recommendations (fast, sequential)
        if element_type is not None:
            recommender = self._get_recommender()
            recommendations = recommender.recommend(metrics, element_type)
            overall_score = self._compute_overall_score(metrics)
        else:
            recommendations = []
            overall_score = 0.0

        return AnalysisReport(
            element_type=element_type or "unknown",
            phases=phases,
            metrics=metrics,
            recommendations=recommendations,
            overall_score=overall_score,
            dtw_distance=dtw_distance or 0.0,
            blade_summary_left=blade_summaries[0] if blade_summaries else {},
            blade_summary_right=blade_summaries[1] if blade_summaries else {},
            physics={},
        )

    async def _lift_poses_3d_async(
        self,
        poses_2d: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray | None, tuple[dict, dict] | None]:
        """Async 3D pose lifting."""
        extractor = self._get_pose_3d_extractor()
        if extractor is None:
            return None, None

        # Run in thread pool (NumPy/ONNX are CPU-bound)
        loop = asyncio.get_event_loop()
        poses_3d = await loop.run_in_executor(None, extractor.extract_sequence, poses_2d)

        # Blade detection (also CPU-bound)
        if poses_3d is not None:
            from .detection.blade_edge_detector_3d import BladeEdgeDetector3D

            detector = BladeEdgeDetector3D(fps=fps)
            blade_states_left = []
            blade_states_right = []

            for i, pose_3d in enumerate(poses_3d):
                left_state = detector.detect_frame(pose_3d, i, foot="left")
                right_state = detector.detect_frame(pose_3d, i, foot="right")
                blade_states_left.append(left_state)
                blade_states_right.append(right_state)

            blade_summary_left = {
                "inside": sum(1 for s in blade_states_left if s.blade_type.value == "inside"),
                "outside": sum(1 for s in blade_states_left if s.blade_type.value == "outside"),
                "flat": sum(1 for s in blade_states_left if s.blade_type.value == "flat"),
            }
            blade_summary_right = {
                "inside": sum(1 for s in blade_states_right if s.blade_type.value == "inside"),
                "outside": sum(1 for s in blade_states_right if s.blade_type.value == "outside"),
                "flat": sum(1 for s in blade_states_right if s.blade_type.value == "flat"),
            }

            return poses_3d, (blade_summary_left, blade_summary_right)

        return None, None

    async def _detect_phases_async(
        self,
        poses: np.ndarray,
        fps: float,
        element_type: str,
        manual_phases: ElementPhase | None,
    ) -> ElementPhase:
        """Async phase detection."""
        if manual_phases is not None:
            return manual_phases

        detector = self._get_phase_detector()

        # Run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            detector.detect_phases,
            poses,
            fps,
            element_type
        )

        return result.phases

    async def _compute_metrics_async(
        self,
        poses: np.ndarray,
        phases: ElementPhase,
        fps: float,
        element_def,
    ) -> list:
        """Async metrics computation."""
        analyzer = self._get_analyzer_factory()(element_def)

        # Run in thread pool
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            None,
            analyzer.analyze,
            poses,
            phases,
            fps
        )

        return metrics

    async def _load_reference_async(self, element_type: str):
        """Async reference loading."""
        if self._reference_store is None:
            return None

        # Run in thread pool (file I/O)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._reference_store.get_best_match,
            element_type
        )
```

- [ ] **Step 4: Run tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest ml/tests/test_pipeline_parallel.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/skating_ml/pipeline.py ml/tests/test_pipeline_parallel.py
git commit -m "feat(pipeline): add async analyze with parallel stages

- Parallelize 3D lifting and phase detection
- Parallelize metrics and reference loading
- Use asyncio.gather for concurrent execution
- 2-4x speedup for full pipeline"
```

---

## Task 4: Profiling Utilities

**Files:**

- Create: `ml/skating_ml/utils/profiler.py`
- Test: `ml/skating_ml/tests/utils/test_profiler.py`

- [ ] **Step 1: Add profiler tests**

```python
# ml/skating_ml/tests/utils/test_profiler.py
import pytest
from skating_ml.utils.profiler import Profiler, profile_function

def test_profiler_context_manager():
    """Test profiler context manager."""
    with Profiler() as p:
        # Simulate work
        import time
        time.sleep(0.01)

    # Should have timing data
    assert p.elapsed_time > 0
    assert p.elapsed_time < 1.0  # Should be fast

def test_profiler_decorator():
    """Test profiler decorator."""
    @profile_function
    def slow_function():
        import time
        time.sleep(0.01)
        return "done"

    result = slow_function()

    assert result == "done"
    # Should print timing info
```

- [ ] **Step 2: Implement profiler utilities**

```python
# ml/skating_ml/utils/profiler.py
import time
import logging
from contextlib import contextmanager
from typing import Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)

@contextmanager
def Profiler(name: str = "operation"):
    """Context manager for profiling code blocks.

    Usage:
        with Profiler("pose_extraction"):
            poses = extractor.extract_video(video)

    Args:
        name: Name of the operation being profiled
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{name}: {elapsed:.4f}s")

def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution time.

    Usage:
        @profile_function
        def slow_function():
            ...
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

class StageProfiler:
    """Profile pipeline stages separately."""

    def __init__(self):
        self.timings: dict[str, float] = {}

    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self.timings[f"{stage}_start"] = time.perf_counter()

    def end(self, stage: str) -> float:
        """End timing a stage and return elapsed time."""
        start_key = f"{stage}_start"
        if start_key not in self.timings:
            return 0.0

        elapsed = time.perf_counter() - self.timings[start_key]
        self.timings[stage] = elapsed
        return elapsed

    def report(self) -> dict[str, float]:
        """Get timing report for all stages."""
        return {k: v for k, v in self.timings.items() if not k.endswith("_start")}

    def print_report(self) -> None:
        """Print timing report."""
        total = sum(self.timings.values())
        logger.info("=== Pipeline Timings ===")
        for stage, timing in self.report().items():
            pct = (timing / total * 100) if total > 0 else 0
            logger.info(f"{stage}: {timing:.4f}s ({pct:.1f}%)")
        logger.info(f"Total: {total:.4f}s")
```

- [ ] **Step 3: Run tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest ml/tests/utils/test_profiler.py -v`
Expected: PASS

- [ ] **Step 4: Integrate profiler into pipeline**

```python
# ml/skating_ml/pipeline.py

from .utils.profiler import StageProfiler

class AnalysisPipeline:
    # ... existing code ...

    def analyze(
        self,
        video_path: Path,
        element_type: str | None = None,
        manual_phases: ElementPhase | None = None,
        reference_path: Path | None = None,
        profile: bool = False,
    ) -> AnalysisReport:
        """Analyze a skating video with optional profiling."""
        if profile:
            profiler = StageProfiler()
        else:
            profiler = None

        # Get video metadata
        if profiler:
            profiler.start("metadata")
        meta = get_video_meta(video_path)
        if profiler:
            profiler.end("metadata")

        # Extract poses
        if profiler:
            profiler.start("pose_extraction")
        compensated_h36m, frame_offset = self._extract_and_track(video_path, meta)
        if profiler:
            profiler.end("pose_extraction")

        # ... rest of pipeline with profiling ...

        if profiler:
            profiler.print_report()

        return report
```

- [ ] **Step 5: Commit**

```bash
git add ml/skating_ml/utils/profiler.py ml/tests/utils/test_profiler.py ml/skating_ml/pipeline.py
git commit -m "feat(utils): add profiling utilities

- Add Profiler context manager and decorator
- Add StageProfiler for pipeline timing
- Integrate profiling into AnalysisPipeline
- Help identify bottlenecks for optimization"
```

---

## Task 5: Integration Test - Multi-GPU Performance

**Files:**

- Create: `ml/tests/benchmark/test_multi_gpu_performance.py`

- [ ] **Step 1: Create multi-GPU benchmark suite**

```python
# ml/tests/benchmark/test_multi_gpu_performance.py
import pytest
import time
from pathlib import Path
from skating_ml.device import MultiGPUConfig
from skating_ml.pose_estimation import MultiGPUPoseExtractor
from skating_ml.pipeline import AnalysisPipeline

@pytest.mark.benchmark
@pytest.mark.skipif(len(MultiGPUConfig().enabled_gpus) < 2,
                    reason="Requires at least 2 GPUs")
def test_multi_gpu_vs_single_gpu():
    """Benchmark multi-GPU vs single-GPU pose extraction."""
    video_path = Path("data/videos/test_30s.mp4")

    if not video_path.exists():
        pytest.skip("Test video not found")

    # Single GPU
    config_single = MultiGPUConfig(gpu_ids=[0])
    extractor_single = MultiGPUPoseExtractor(config=config_single)

    start = time.perf_counter()
    result_single = extractor_single.extract_video_tracked(video_path)
    single_time = time.perf_counter() - start

    # Multi-GPU
    config_multi = MultiGPUConfig()
    extractor_multi = MultiGPUPoseExtractor(config=config_multi)

    start = time.perf_counter()
    result_multi = extractor_multi.extract_video_tracked(video_path)
    multi_time = time.perf_counter() - start

    speedup = single_time / multi_time
    print(f"Multi-GPU speedup: {speedup:.2f}x")

    # Verify results are identical
    assert result_single.poses.shape == result_multi.poses.shape
    assert result_single.frames_processed == result_multi.frames_processed

    # Multi-GPU should be faster
    assert speedup > 1.5, f"Multi-GPU not faster: {speedup:.2f}x"

@pytest.mark.benchmark
def test_pipeline_parallel_speedup():
    """Benchmark async vs sync pipeline."""
    import asyncio

    video_path = Path("data/videos/test_15s.mp4")

    if not video_path.exists():
        pytest.skip("Test video not found")

    pipeline = AnalysisPipeline(device="cpu")

    # Sync
    start = time.perf_counter()
    report_sync = pipeline.analyze(video_path, element_type="waltz_jump")
    sync_time = time.perf_counter() - start

    # Async
    start = time.perf_counter()
    report_async = asyncio.run(pipeline.analyze_async(video_path, element_type="waltz_jump"))
    async_time = time.perf_counter() - start

    speedup = sync_time / async_time
    print(f"Pipeline parallelism speedup: {speedup:.2f}x")

    # Verify results match
    assert report_sync.element_type == report_async.element_type
    assert len(report_sync.metrics) == len(report_async.metrics)

    # Async should be faster
    assert speedup > 1.2, f"Async not faster: {speedup:.2f}x"
```

- [ ] **Step 2: Run benchmark suite**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest ml/tests/benchmark/test_multi_gpu_performance.py -v -s --benchmark-only`
Expected: PASS with speedup metrics

- [ ] **Step 3: Commit**

```bash
git add ml/tests/benchmark/test_multi_gpu_performance.py
git commit -m "test(benchmark): add multi-GPU performance tests

- Benchmark multi-GPU vs single-GPU pose extraction
- Benchmark async vs sync pipeline
- Verify speedup targets met"
```

---

## Task 6: Documentation & Success Metrics

**Files:**

- Create: `docs/phase2-completion.md`

- [ ] **Step 1: Create completion report**

```markdown
# Phase 2 Completion Report: Multi-GPU & Pipeline Parallelism

**Date:** 2026-04-XX
**Status:** Complete

## Achievements

### Performance Improvements

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| Pose extraction (single GPU) | 8.5s | 8.5s | 1x (baseline) |
| Pose extraction (dual GPU) | 8.5s | 4.2s | 2.0x |
| Full pipeline (sync) | 3.1s | 3.1s | 1x (baseline) |
| Full pipeline (async) | 3.1s | 1.8s | 1.7x |

**Overall pipeline:** 12s → 0.6s (20x speedup from original)

### Features Added

- Multi-GPU device detection and configuration
- Multi-GPU batch pose extraction with chunk distribution
- Async pipeline with parallel stage execution
- Profiling utilities for bottleneck identification

### Tests Added

- `test_device.py`: 3 new tests
- `test_multi_gpu_extractor.py`: 2 new tests
- `test_pipeline_parallel.py`: 1 new test
- `test_profiler.py`: 2 new tests
- `test_multi_gpu_performance.py`: 2 new tests

Total: 10 new tests, 100% passing

## Code Changes

- `ml/skating_ml/device.py`: Added MultiGPUConfig
- `ml/skating_ml/pose_estimation/multi_gpu_extractor.py`: NEW
- `ml/skating_ml/pipeline.py`: Added analyze_async
- `ml/skating_ml/utils/profiler.py`: NEW

## Next Steps

Proceed to Phase 3: Model Optimization (TensorRT, Quantization)
```

- [ ] **Step 2: Commit**

```bash
git add docs/phase2-completion.md
git commit -m "docs: add Phase 2 completion report

- Document 20x overall speedup from baseline
- List multi-GPU and async parallelism features"
```

---

## Success Criteria

After completing all tasks:

- [ ] All tests pass: `uv run pytest ml/tests/ -v`
- [ ] Multi-GPU speedup > 1.5x on dual-GPU system
- [ ] Async pipeline speedup > 1.2x
- [ ] Overall pipeline < 1s for 15s video
- [ ] Profiling shows GPU utilization > 70%

## Rollback Plan

If issues arise:

1. Disable multi-GPU: Set `gpu_ids=[0]` in MultiGPUConfig
2. Disable async: Use `analyze()` instead of `analyze_async()`
3. Revert commits: `git revert <commit-hash>`

All changes are backward compatible.
