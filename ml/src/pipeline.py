"""Main analysis pipeline orchestrator.

H3.6M Architecture:
    This pipeline uses H3.6M 17-keypoint format as the primary format.
    2D extraction: PoseExtractor (rtmlib RTMO Body)
    3D lifting: AthletePose3DExtractor (MotionAGFormer)

Pipeline stages:
    1. Extract & track: PoseExtractor.extract_video_tracked() + gap fill + spatial ref
    2. Normalization
    3. Temporal smoothing (One-Euro Filter)
    4. Phase detection
    5. Biomechanics metrics
    6. 3D pose estimation (optional, for blade detection & physics)
    7. Reference comparison (DTW)
    8. Recommendations
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .device import DeviceConfig
from .types import AnalysisReport, ElementPhase, PersonClick, SegmentationResult
from .utils.geometry import calculate_com_trajectory
from .utils.profiling import PipelineProfiler
from .utils.video import VideoMeta, get_video_meta

if TYPE_CHECKING:
    from .alignment import MotionAligner, MotionDTWAligner
    from .analysis.phase_detector import PhaseDetector
    from .analysis.recommender import Recommender
    from .detection import PersonDetector
    from .pose_3d import AthletePose3DExtractor
    from .pose_estimation.normalizer import PoseNormalizer
    from .pose_estimation.pose_extractor import PoseExtractor
    from .references import ReferenceStore
    from .utils.smoothing import OneEuroFilterConfig, PoseSmoother


class AnalysisPipeline:
    """Main pipeline for skating technique analysis.

    H3.6M Architecture:
        - 2D poses: PoseExtractor (17 keypoints, normalized [0,1], rtmlib backend)
        - 3D poses: AthletePose3DExtractor (MotionAGFormer)
    """

    def __init__(
        self,
        reference_store: ReferenceStore | None = None,  # type: ignore[valid-type]
        device: str | DeviceConfig = "auto",
        enable_smoothing: bool = True,
        smoothing_config: OneEuroFilterConfig | None = None,  # type: ignore[valid-type]
        person_click: PersonClick | None = None,
        reestimate_camera: bool = False,
        profiler: PipelineProfiler | None = None,
        compute_3d: bool = False,
    ) -> None:
        """Initialize analysis pipeline.

        Args:
            reference_store: ReferenceStore for loading expert references.
            device: Device configuration — ``"auto"`` (default), ``"cuda"``, ``"cpu"``,
                or a DeviceConfig instance for custom behavior.
            enable_smoothing: Whether to apply One-Euro Filter temporal smoothing.
            smoothing_config: Optional custom smoothing configuration.
            person_click: Optional click point to select target person in multi-person videos.
            reestimate_camera: Enable per-frame camera re-estimation for moving cameras.
            profiler: Optional PipelineProfiler for recording stage timings.
            compute_3d: Whether to run 3D pose lifting (CorrectiveLens + blade detection).
                Disabled by default since 3D lifting adds significant compute
                and the model file is often not present.
        """
        self._reference_store = reference_store
        self._device_config = DeviceConfig(device) if isinstance(device, str) else device
        self._enable_smoothing = enable_smoothing
        self._smoothing_config = smoothing_config
        self._person_click = person_click
        self._reestimate_camera = reestimate_camera
        self._profiler = profiler or PipelineProfiler()
        self._compute_3d = compute_3d

        # Components will be lazy-loaded
        self._detector: PersonDetector | None = None  # type: ignore[valid-type]
        self._pose_2d_extractor: PoseExtractor | None = None  # type: ignore[valid-type]
        self._pose_3d_extractor: AthletePose3DExtractor | None = None  # type: ignore[valid-type]
        self._normalizer: PoseNormalizer | None = None  # type: ignore[valid-type]
        self._smoother: PoseSmoother | None = None  # type: ignore[valid-type]
        self._phase_detector: PhaseDetector | None = None  # type: ignore[valid-type]
        self._analyzer_factory: type | None = None
        self._aligner: MotionAligner | MotionDTWAligner | None = None  # type: ignore[valid-type]
        self._recommender: Recommender | None = None  # type: ignore[valid-type]

    def _extract_and_track(self, video_path: Path, meta: VideoMeta) -> tuple[np.ndarray, int]:
        """Extract poses with tracking, gap filling, and spatial compensation.

        Combines extraction, pre-roll trimming, gap filling, and camera
        compensation into a single method.

        Args:
            video_path: Path to video file.
            meta: Video metadata (width, height, fps, etc.).

        Returns:
            (compensated_h36m, frame_offset) — poses (N, 17, 3) normalized,
            frame_offset = first_detection_frame index.
        """
        # 1. Lazy-init extractor (model download + ONNX session)
        t0 = time.perf_counter()
        extractor = self._get_pose_2d_extractor()
        if extractor is None:
            raise RuntimeError("2D pose extractor not initialized")
        self._profiler.record("extractor_init", time.perf_counter() - t0)

        # 2. Tracked extraction (per-frame RTMO inference + tracking)
        t0 = time.perf_counter()
        extraction = extractor.extract_video_tracked(video_path, person_click=self._person_click)
        self._profiler.record("rtmo_inference_loop", time.perf_counter() - t0)

        # 3. Skip pre-roll (trim leading NaN frames before first detection)
        frame_offset = extraction.first_detection_frame
        poses = extraction.poses[frame_offset:]
        valid = extraction.valid_mask()[frame_offset:]
        first_frame = extraction.first_frame

        # 4. Gap filling
        t0 = time.perf_counter()
        from .utils.gap_filling import GapFiller

        filler = GapFiller()
        filled, _report = filler.fill_gaps(poses, valid)
        self._profiler.record("gap_filling", time.perf_counter() - t0)

        # 5. Spatial reference / camera compensation
        t0 = time.perf_counter()
        if self._reestimate_camera:
            from .detection.spatial_reference import (
                compensate_poses_per_frame,
                estimate_pose_sequence,
            )

            camera_poses = estimate_pose_sequence(str(video_path), interval=30, fps=meta.fps)
            compensated = compensate_poses_per_frame(
                filled,
                camera_poses,
                video_width=meta.width,
                video_height=meta.height,
            )
        else:
            # Single-frame estimation (use cached first_frame from extraction)
            import cv2

            from .detection.spatial_reference import CameraPose, SpatialReferenceDetector

            spatial_detector = SpatialReferenceDetector()
            if first_frame is not None:
                camera_pose = spatial_detector.estimate_pose(first_frame)
            else:
                # Fallback: read first frame from video
                cap = cv2.VideoCapture(str(video_path))
                ret, first_frame_fallback = cap.read()
                if ret:
                    camera_pose = spatial_detector.estimate_pose(first_frame_fallback)
                else:
                    camera_pose = CameraPose()
                cap.release()

            if camera_pose.confidence > 0.1:
                # Convert to pixels, compensate, convert back
                poses_px = filled[:, :, :2] * np.array([meta.width, meta.height])
                poses_with_conf = np.dstack([poses_px, filled[:, :, 2]])
                compensated_px = spatial_detector.compensate_poses(poses_with_conf, camera_pose)
                compensated = compensated_px[:, :, :2] / np.array([meta.width, meta.height])
                compensated = np.dstack([compensated, compensated_px[:, :, 2:3]])
            else:
                compensated = filled
        self._profiler.record("spatial_reference", time.perf_counter() - t0)

        return compensated, frame_offset

    def analyze(
        self,
        video_path: Path,
        element_type: str | None = None,
        manual_phases: ElementPhase | None = None,
        reference_path: Path | None = None,
    ) -> AnalysisReport:
        """Analyze a skating video.

        Args:
            video_path: Path to user's video file.
            element_type: Type of skating element (e.g., 'three_turn', 'waltz_jump').
                If None, only pose extraction + visualization is performed
                (no metrics, DTW, or recommendations).
            manual_phases: Optional manual phase boundaries (auto-detect if None).
            reference_path: Optional path to reference video (use store if None).

        Returns:
            AnalysisReport with metrics, recommendations, and scores.

        Raises:
            ValueError: If video cannot be processed or element type not supported.
        """
        # Validate element type (only when specified)
        from .analysis import element_defs

        element_def = None
        if element_type is not None:
            element_def = element_defs.get_element_def(element_type)
            if element_def is None:
                raise ValueError(f"Unknown element type: {element_type}")

        t0 = time.perf_counter()
        meta = get_video_meta(video_path)
        self._profiler.record("video_meta", time.perf_counter() - t0)

        t0 = time.perf_counter()
        compensated_h36m, _frame_offset = self._extract_and_track(video_path, meta)
        self._profiler.record("extract_and_track", time.perf_counter() - t0)

        t0 = time.perf_counter()
        normalized = self._get_normalizer().normalize(compensated_h36m)
        self._profiler.record("normalize", time.perf_counter() - t0)

        # Stage 3.5: Smooth poses (temporal filtering)
        t0 = time.perf_counter()
        if self._enable_smoothing:
            if manual_phases is not None:
                boundaries = [manual_phases.takeoff, manual_phases.peak, manual_phases.landing]
                boundaries = [b for b in boundaries if b > 0]
                smoothed = self._get_smoother(meta.fps).smooth_phase_aware(normalized, boundaries)
            else:
                smoothed = self._get_smoother(meta.fps).smooth(normalized)
        else:
            smoothed = normalized
        self._profiler.record("smooth", time.perf_counter() - t0)

        # Stage 3.6: 3D pose estimation (for blade detection & physics)
        t0 = time.perf_counter()
        poses_3d = None
        blade_summary_left = None
        blade_summary_right = None
        if self._compute_3d:
            try:
                # Use MotionAGFormer for 3D lifting (H3.6M format)
                poses_3d = self._get_pose_3d_extractor().extract_sequence(smoothed)

                from .detection.blade_edge_detector_3d import BladeEdgeDetector3D

                # Detect blade edge states using 3D poses
                blade_detector_3d = BladeEdgeDetector3D(fps=meta.fps)
                blade_states_left = []
                blade_states_right = []
                for i, pose_3d in enumerate(poses_3d):
                    left_state = blade_detector_3d.detect_frame(pose_3d, i, foot="left")
                    right_state = blade_detector_3d.detect_frame(pose_3d, i, foot="right")
                    blade_states_left.append(left_state)
                    blade_states_right.append(right_state)

                blade_summary_left = {
                    "inside": sum(1 for s in blade_states_left if s.blade_type.value == "inside"),
                    "outside": sum(1 for s in blade_states_left if s.blade_type.value == "outside"),
                    "flat": sum(1 for s in blade_states_left if s.blade_type.value == "flat"),
                }
                blade_summary_right = {
                    "inside": sum(1 for s in blade_states_right if s.blade_type.value == "inside"),
                    "outside": sum(
                        1 for s in blade_states_right if s.blade_type.value == "outside"
                    ),
                    "flat": sum(1 for s in blade_states_right if s.blade_type.value == "flat"),
                }
            except Exception:
                # 3D lifting is optional, don't fail if it errors
                pass
        self._profiler.record("3d_lift_and_blade", time.perf_counter() - t0)

        # Stage 4-7: Element-specific analysis (only when element_type provided)
        if element_type is not None and element_def is not None:
            # Stage 4: Detect phases (or use manual)
            t0 = time.perf_counter()
            if manual_phases is not None:
                phases = manual_phases
            else:
                phase_result = self._get_phase_detector().detect_phases(
                    smoothed, meta.fps, element_type
                )
                phases = phase_result.phases
            self._profiler.record("phase_detection", time.perf_counter() - t0)

            # Stage 5: Compute biomechanics metrics
            t0 = time.perf_counter()
            analyzer = self._get_analyzer_factory()(element_def)
            metrics = analyzer.analyze(smoothed, phases, meta.fps)
            self._profiler.record("metrics", time.perf_counter() - t0)

            # Stage 6: Load reference and align (if available)
            t0 = time.perf_counter()
            dtw_distance: float | None = None
            if self._reference_store is not None:
                reference = self._reference_store.get_best_match(element_type)
                if reference is not None:
                    aligner = self._get_aligner()
                    dtw_distance = aligner.compute_distance(
                        normalized[phases.start : phases.end],
                        reference.poses[reference.phases.start : reference.phases.end],
                    )
            self._profiler.record("dtw_alignment", time.perf_counter() - t0)

            # Stage 6.5: Physics calculations (3D pose + biomechanics)
            t0 = time.perf_counter()
            physics_dict: dict = {}
            if poses_3d is not None:
                try:
                    from .analysis.physics_engine import PhysicsEngine

                    physics_engine = PhysicsEngine(body_mass=60.0)

                    if phases.takeoff > 0 and phases.landing > 0:
                        trajectory = physics_engine.fit_jump_trajectory(
                            poses_3d, phases.takeoff, phases.landing
                        )
                        physics_dict["jump_height"] = trajectory["height"]
                        physics_dict["flight_time"] = trajectory["flight_time"]
                        physics_dict["takeoff_velocity"] = trajectory["takeoff_velocity"]
                        physics_dict["fit_quality"] = trajectory["fit_quality"]

                    inertia = physics_engine.calculate_moment_of_inertia(
                        poses_3d[phases.start : phases.end]
                    )
                    physics_dict["avg_inertia"] = float(np.mean(inertia))
                except Exception:
                    pass
            self._profiler.record("physics", time.perf_counter() - t0)

            # Stage 7: Generate recommendations
            t0 = time.perf_counter()
            recommender = self._get_recommender()
            recommendations = recommender.recommend(metrics, element_type)
            self._profiler.record("recommendations", time.perf_counter() - t0)

            # Stage 8: Compute overall score
            overall_score = self._compute_overall_score(metrics)
        else:
            # No element type specified — poses + visualization only
            phases = ElementPhase(name="unknown", start=0, takeoff=0, peak=0, landing=0, end=0)
            metrics = []
            recommendations = []
            overall_score = None
            dtw_distance = None
            physics_dict = {}

        return AnalysisReport(
            element_type=element_type or "unknown",
            phases=phases,
            metrics=metrics,
            recommendations=recommendations,
            overall_score=overall_score if overall_score is not None else 0.0,
            dtw_distance=dtw_distance if dtw_distance is not None else 0.0,
            blade_summary_left=blade_summary_left if blade_summary_left is not None else {},
            blade_summary_right=blade_summary_right if blade_summary_right is not None else {},
            physics=physics_dict,
            profiling=self._profiler.to_dict(),
        )

    def segment_video(
        self,
        video_path: Path,
    ) -> SegmentationResult:
        """Segment video into individual skating elements.

        Args:
            video_path: Path to training video with multiple elements.

        Returns:
            SegmentationResult with detected elements.
        """
        from .analysis import element_segmenter

        ElementSegmenter = element_segmenter.ElementSegmenter

        # Get video metadata
        meta = get_video_meta(video_path)

        # Stage 1-2.6: Extract poses with tracking, gap filling, spatial compensation
        compensated_h36m, _frame_offset = self._extract_and_track(video_path, meta)

        # Stage 2: Normalize poses

        normalized = self._get_normalizer().normalize(compensated_h36m)

        # Stage 3: Smooth poses
        if self._enable_smoothing:
            smoothed = self._get_smoother(meta.fps).smooth(normalized)
        else:
            smoothed = normalized

        # Stage 4: Segment video into elements
        segmenter = ElementSegmenter()
        segmentation = segmenter.segment(smoothed, video_path, meta)

        return segmentation

    def _get_detector(self) -> PersonDetector:  # type: ignore[valid-type]
        """Lazy-load person detector."""
        if self._detector is None:
            from .detection import person_detector

            PersonDetector = person_detector.PersonDetector

            self._detector = PersonDetector(model_size="n", confidence=0.5)
        return self._detector

    def _get_pose_2d_extractor(self) -> PoseExtractor:  # type: ignore[valid-type]
        """Lazy-load PoseExtractor (sole 2D pose backend)."""
        if self._pose_2d_extractor is None:
            from .pose_estimation.pose_extractor import PoseExtractor

            self._pose_2d_extractor = PoseExtractor(
                output_format="normalized",
                device=self._device_config.device,
            )
        return self._pose_2d_extractor  # type: ignore[return-value]

    def _get_pose_3d_extractor(self) -> AthletePose3DExtractor | None:  # type: ignore[valid-type]
        """Lazy-load 3D pose lifter (MotionAGFormer). Returns None if model not found."""
        if self._pose_3d_extractor is None:
            model_path = Path("data/models/motionagformer-s-ap3d.onnx")
            if not model_path.exists():
                return None
            from .pose_3d import AthletePose3DExtractor

            self._pose_3d_extractor = AthletePose3DExtractor(
                model_path=model_path,
                device=self._device_config.device,
            )
        return self._pose_3d_extractor

    def _get_normalizer(self) -> PoseNormalizer:  # type: ignore[valid-type]
        """Lazy-load pose normalizer."""
        if self._normalizer is None:
            from .pose_estimation import normalizer

            PoseNormalizer = normalizer.PoseNormalizer

            self._normalizer = PoseNormalizer(target_spine_length=0.4)
        return self._normalizer

    def _get_smoother(self, fps: float = 30.0) -> PoseSmoother:  # type: ignore[valid-type]
        """Lazy-load pose smoother with One-Euro Filter."""
        if not self._enable_smoothing:
            from .utils.smoothing import OneEuroFilterConfig, PoseSmoother

            config = OneEuroFilterConfig(min_cutoff=100.0, beta=0.0, freq=fps)
            return PoseSmoother(config=config, freq=fps)

        if self._smoother is None:
            from .utils.smoothing import (
                PoseSmoother,
                get_skating_optimized_config,
            )

            config = self._smoothing_config or get_skating_optimized_config(fps)
            self._smoother = PoseSmoother(config=config, freq=fps)
        return self._smoother

    def _get_phase_detector(self) -> PhaseDetector:  # type: ignore[valid-type]
        """Lazy-load phase detector."""
        if self._phase_detector is None:
            from .analysis import phase_detector

            PhaseDetector = phase_detector.PhaseDetector

            self._phase_detector = PhaseDetector()
        return self._phase_detector

    def _get_analyzer_factory(self) -> type:
        """Get analyzer factory (returns BiomechanicsAnalyzer class)."""
        if self._analyzer_factory is None:
            from .analysis import metrics

            BiomechanicsAnalyzer = metrics.BiomechanicsAnalyzer

            self._analyzer_factory = BiomechanicsAnalyzer
        return self._analyzer_factory

    def _get_aligner(self) -> MotionAligner | MotionDTWAligner:  # type: ignore[valid-type]
        """Lazy-load motion aligner (using phase-aware MotionDTW)."""
        if self._aligner is None:
            from .alignment import motion_dtw

            MotionDTWAligner = motion_dtw.MotionDTWAligner

            self._aligner = MotionDTWAligner(window_type="sakoechiba", window_size=0.2)
        return self._aligner

    def _get_recommender(self) -> Recommender:  # type: ignore[valid-type]
        """Lazy-load recommender."""
        if self._recommender is None:
            from .analysis import recommender

            Recommender = recommender.Recommender

            self._recommender = Recommender()
        return self._recommender

    def _compute_overall_score(self, metrics: list) -> float:  # type: ignore[valid-type]
        """Compute overall quality score 0-10 from metrics.

        Args:
            metrics: List of MetricResult.

        Returns:
            Overall score 0-10.
        """
        if not metrics:
            return 5.0

        good_count = sum(1 for m in metrics if m.is_good)
        total_count = len(metrics)

        if total_count == 0:
            return 5.0

        ratio = good_count / total_count
        score = ratio * 10

        return float(round(score, 1))

    def format_report(self, report: AnalysisReport) -> str:
        """Format analysis report as human-readable text.

        Args:
            report: AnalysisReport to format.

        Returns:
            Formatted text report in Russian.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"АНАЛИЗ: {report.element_type.upper()}")
        lines.append("=" * 60)

        # Phases
        lines.append("\n--- Фазы элемента ---")
        phase = report.phases
        if phase and (phase.takeoff > 0 or phase.start > 0):  # Only show if valid
            lines.append(f"  Начало:     {phase.start}")
            lines.append(f"  Отрыв:      {phase.takeoff}")
            lines.append(f"  Пик:        {phase.peak}")
            lines.append(f"  Приземление: {phase.landing}")
            lines.append(f"  Конец:       {phase.end}")

        # Metrics
        lines.append("\n--- Биомеханические метрики ---")
        for metric in report.metrics:
            status = (
                "\u2713 \u041e\u041a" if metric.is_good else "\u2717 \u041f\u041b\u041e\u0425\u041e"
            )
            ref_min, ref_max = metric.reference_range
            lines.append(
                f"  {metric.name}: {metric.value:.2f} {metric.unit} [{status}] "
                f"(референс: {ref_min:.2f}-{ref_max:.2f})"
            )

        # DTW distance
        if report.dtw_distance is not None:
            lines.append(
                "\n--- \u0421\u0445\u043e\u0434\u0441\u0442\u0432\u043e \u0441 \u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043e\u043c ---"
            )
            lines.append(f"  DTW-расстояние: {report.dtw_distance:.3f} (0 = идеально)")

        # Blade edge information
        if report.blade_summary_left or report.blade_summary_right:
            lines.append("\n--- Состояние лезвия ---")
            if report.blade_summary_left:
                lines.append(
                    f"  Левая нога: {report.blade_summary_left.get('dominant_edge', 'unknown')}"
                )
                if "type_percentages" in report.blade_summary_left:
                    types = report.blade_summary_left["type_percentages"]
                    lines.append(f"    Распределение: {types}")
            if report.blade_summary_right:
                lines.append(
                    f"  Правая нога: {report.blade_summary_right.get('dominant_edge', 'unknown')}"
                )
                if "type_percentages" in report.blade_summary_right:
                    types = report.blade_summary_right["type_percentages"]
                    lines.append(f"    Распределение: {types}")

        # Recommendations
        if report.recommendations:
            lines.append("\n--- РЕКОМЕНДАЦИИ ---")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        else:
            lines.append("\n--- РЕКОМЕНДАЦИИ ---")
            lines.append("  Отличное выполнение! Продолжай в том же духе.")

        # Overall score
        if report.overall_score is not None:
            lines.append(
                f"\n\u041e\u0431\u0449\u0438\u0439 \u0431\u0430\u043b\u043b: {report.overall_score:.1f} / 10"
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Async pipeline with parallel stage execution
    # ------------------------------------------------------------------

    async def analyze_async(
        self,
        video_path: Path,
        element_type: str | None = None,
        manual_phases: ElementPhase | None = None,
        reference_path: Path | None = None,
    ) -> AnalysisReport:
        """Async version of analyze with parallel stage execution.

        Parallelizes independent operations:
        - 3D lifting and blade detection run in parallel with phase detection
        - Metrics computation runs in parallel with reference loading

        Args:
            video_path: Path to user's video file.
            element_type: Type of skating element (e.g., 'three_turn', 'waltz_jump').
            manual_phases: Optional manual phase boundaries (auto-detect if None).
            reference_path: Optional path to reference video (use store if None).

        Returns:
            AnalysisReport with metrics, recommendations, and scores.
        """
        # Validate element type
        from .analysis import element_defs

        element_def = None
        if element_type is not None:
            element_def = element_defs.get_element_def(element_type)
            if element_def is None:
                raise ValueError(f"Unknown element type: {element_type}")

        # Get video metadata
        meta = get_video_meta(video_path)

        # Stage 1-2.6: Extract poses with tracking (must be sequential)
        compensated_h36m, _frame_offset = self._extract_and_track(video_path, meta)

        # Stage 3: Normalize poses (fast, run in main)
        normalized = self._get_normalizer().normalize(compensated_h36m)

        # Stage 3.5: Smooth poses (fast, run in main)
        if self._enable_smoothing:
            if manual_phases is not None:
                boundaries = [manual_phases.takeoff, manual_phases.peak, manual_phases.landing]
                boundaries = [b for b in boundaries if b > 0]
                smoothed = self._get_smoother(meta.fps).smooth_phase_aware(normalized, boundaries)
            else:
                smoothed = self._get_smoother(meta.fps).smooth(normalized)
        else:
            smoothed = normalized

        # Default phases for non-element analysis
        poses_3d: np.ndarray | None = None
        blade_summaries: tuple | None = None
        phases = ElementPhase(name="unknown", start=0, takeoff=0, peak=0, landing=0, end=0)
        reference: np.ndarray | None = None

        # Element-specific analysis with wave-based parallelism
        if element_type is not None and element_def is not None:
            # Pre-compute CoM once (shared by metrics + 3D physics)
            com_trajectory = calculate_com_trajectory(smoothed)

            # === Wave 1: 3D lift, phase detection, reference load in parallel ===
            wave1_tasks: list[asyncio.Task] = []

            if self._compute_3d:
                wave1_tasks.append(
                    asyncio.create_task(self._lift_poses_3d_async(smoothed, meta.fps))
                )

            wave1_tasks.append(
                asyncio.create_task(
                    self._detect_phases_async(smoothed, meta.fps, element_type, manual_phases)
                )
            )

            if self._reference_store is not None:
                wave1_tasks.append(asyncio.create_task(self._load_reference_async(element_type)))

            wave1_results = await asyncio.gather(*wave1_tasks)

            # Unpack wave 1 results
            result_idx = 0
            if self._compute_3d:
                poses_3d, blade_summaries = wave1_results[result_idx]
                result_idx += 1

            phases = wave1_results[result_idx]
            result_idx += 1

            reference = wave1_results[result_idx] if result_idx < len(wave1_results) else None

            # === Wave 2: physics, metrics in parallel ===
            wave2_tasks: list[asyncio.Task] = []

            if poses_3d is not None:
                wave2_tasks.append(
                    asyncio.create_task(self._compute_physics_async(poses_3d, phases))
                )

            wave2_tasks.append(
                asyncio.create_task(
                    self._compute_metrics_async(
                        smoothed,
                        phases,
                        meta.fps,
                        element_def,
                        com_trajectory=com_trajectory,
                    )
                )
            )

            wave2_results = await asyncio.gather(*wave2_tasks)

            # Unpack wave 2 results
            result_idx = 0
            if poses_3d is not None:
                physics_dict = wave2_results[result_idx] or {}
                result_idx += 1
            else:
                physics_dict = {}

            metrics = wave2_results[result_idx]

            # DTW alignment (needs phases + reference)
            dtw_distance = None
            if reference is not None:
                aligner = self._get_aligner()
                dtw_distance = aligner.compute_distance(
                    normalized[phases.start : phases.end],
                    reference.poses[reference.phases.start : reference.phases.end],
                )

            recommender = self._get_recommender()
            recommendations = recommender.recommend(metrics, element_type)
            overall_score = self._compute_overall_score(metrics)
        else:
            # No element type specified
            metrics = []
            recommendations = []
            overall_score = None
            dtw_distance = None
            physics_dict = {}

        # Extract blade summaries
        blade_summary_left = blade_summaries[0] if blade_summaries and blade_summaries[0] else {}
        blade_summary_right = blade_summaries[1] if blade_summaries and blade_summaries[1] else {}

        return AnalysisReport(
            element_type=element_type or "unknown",
            phases=phases,
            metrics=metrics,
            recommendations=recommendations,
            overall_score=overall_score if overall_score is not None else 0.0,
            dtw_distance=dtw_distance if dtw_distance is not None else 0.0,
            blade_summary_left=blade_summary_left,
            blade_summary_right=blade_summary_right,
            physics=physics_dict,
        )

    async def _lift_poses_3d_async(
        self,
        poses_2d: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray | None, tuple[dict, dict] | None]:
        """Async 3D pose lifting with blade detection.

        Args:
            poses_2d: (N, 17, 2) normalized 2D poses.
            fps: Video frame rate.

        Returns:
            Tuple of (poses_3d, (blade_summary_left, blade_summary_right)).
        """
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
        """Async phase detection.

        Args:
            poses: (N, 17, 2) normalized poses.
            fps: Video frame rate.
            element_type: Element type for detection.
            manual_phases: Manual phases if provided.

        Returns:
            ElementPhase with detected boundaries.
        """
        if manual_phases is not None:
            return manual_phases

        # Run in thread pool
        loop = asyncio.get_event_loop()
        detector = self._get_phase_detector()
        result = await loop.run_in_executor(None, detector.detect_phases, poses, fps, element_type)
        return result.phases

    async def _compute_metrics_async(
        self,
        poses: np.ndarray,
        phases: ElementPhase,
        fps: float,
        element_def,
        com_trajectory: np.ndarray | None = None,
    ) -> list:
        """Async biomechanics metrics computation.

        Args:
            poses: (N, 17, 2) normalized poses.
            phases: Element phases.
            fps: Video frame rate.
            element_def: Element definition.
            com_trajectory: Pre-computed CoM trajectory (optional, for caching).

        Returns:
            List of MetricResult.
        """
        # Run in thread pool
        loop = asyncio.get_event_loop()
        analyzer = self._get_analyzer_factory()(element_def)
        metrics = await loop.run_in_executor(
            None, analyzer.analyze, poses, phases, fps, com_trajectory
        )
        return metrics

    async def _load_reference_async(self, element_type: str):
        """Async reference loading from store.

        Args:
            element_type: Element type to load reference for.

        Returns:
            ReferenceData or None.
        """
        if self._reference_store is None:
            return None

        # Run in thread pool
        loop = asyncio.get_event_loop()
        reference = await loop.run_in_executor(
            None, self._reference_store.get_best_match, element_type
        )
        return reference

    async def _compute_physics_async(
        self,
        poses_3d: np.ndarray,
        phases: ElementPhase,
    ) -> dict | None:
        """Async physics calculations with CoM caching.

        Args:
            poses_3d: (N, 17, 3) 3D pose array.
            phases: Element phase boundaries.

        Returns:
            Physics dict with jump_height, flight_time, avg_inertia, takeoff_velocity,
            fit_quality, or None on error.
        """
        try:
            from .analysis.physics_engine import PhysicsEngine

            physics_engine = PhysicsEngine(body_mass=60.0)
            result = physics_engine.analyze(
                poses_3d, takeoff_idx=phases.takeoff, landing_idx=phases.landing
            )
            physics_dict: dict = {
                "avg_inertia": float(np.mean(result.moment_of_inertia)),
            }
            if result.jump_height is not None:
                physics_dict["jump_height"] = result.jump_height
                physics_dict["flight_time"] = result.flight_time

            # Restore trajectory fit details for types.py report rendering
            if phases.takeoff is not None and phases.landing is not None:
                trajectory = physics_engine._fit_jump_trajectory_with_com(
                    poses_3d, phases.takeoff, phases.landing, result.center_of_mass
                )
                physics_dict["takeoff_velocity"] = trajectory["takeoff_velocity"]
                physics_dict["fit_quality"] = trajectory["fit_quality"]

            return physics_dict
        except Exception:
            return None
