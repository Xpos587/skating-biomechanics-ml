"""Main analysis pipeline orchestrator.

This module combines all pipeline stages into a unified system:
detection -> 2D pose -> normalization -> analysis -> alignment -> recommendations.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from skating_biomechanics_ml.types import AnalysisReport, ElementPhase, SegmentationResult
from skating_biomechanics_ml.utils.video import get_video_meta

if TYPE_CHECKING:
    from skating_biomechanics_ml.alignment import MotionAligner, MotionDTWAligner
    from skating_biomechanics_ml.analysis import PhaseDetector, Recommender
    from skating_biomechanics_ml.detection import PersonDetector
    from skating_biomechanics_ml.pose_2d import PoseNormalizer
    from skating_biomechanics_ml.references import ReferenceStore
    from skating_biomechanics_ml.utils.smoothing import OneEuroFilterConfig, PoseSmoother


class AnalysisPipeline:
    """Main pipeline for skating technique analysis.

    Coordinates all stages:
    1. Person detection (YOLOv11)
    2. 2D pose extraction (BlazePose)
    3. Normalization
    4. Temporal smoothing (One-Euro Filter)
    5. Phase detection
    6. Biomechanics metrics
    7. Reference comparison (DTW)
    8. Recommendations
    """

    def __init__(
        self,
        reference_store: "ReferenceStore | None" = None,  # type: ignore[valid-type]
        use_gpu: bool = True,
        enable_smoothing: bool = True,
        smoothing_config: "OneEuroFilterConfig | None" = None,  # type: ignore[valid-type]
    ) -> None:
        """Initialize analysis pipeline.

        Args:
            reference_store: ReferenceStore for loading expert references.
            use_gpu: Whether to use GPU acceleration (when available).
            enable_smoothing: Whether to apply One-Euro Filter temporal smoothing.
            smoothing_config: Optional custom smoothing configuration.
        """
        self._reference_store = reference_store
        self._use_gpu = use_gpu
        self._enable_smoothing = enable_smoothing
        self._smoothing_config = smoothing_config

        # Components will be lazy-loaded
        self._detector: "PersonDetector | None" = None  # type: ignore[valid-type]
        self._pose_extractor: "BlazePoseExtractor | None" = None  # type: ignore[valid-type]
        self._normalizer: "PoseNormalizer | None" = None  # type: ignore[valid-type]
        self._smoother: "PoseSmoother | None" = None  # type: ignore[valid-type]
        self._phase_detector: "PhaseDetector | None" = None  # type: ignore[valid-type]
        self._analyzer_factory: type | None = None
        self._aligner: "MotionAligner | MotionDTWAligner | None" = None  # type: ignore[valid-type]
        self._recommender: "Recommender | None" = None  # type: ignore[valid-type]

    def analyze(
        self,
        video_path: Path,
        element_type: str,
        manual_phases: ElementPhase | None = None,
        reference_path: Path | None = None,  # noqa: ARG002
    ) -> AnalysisReport:
        """Analyze a skating video.

        Args:
            video_path: Path to user's video file.
            element_type: Type of skating element (e.g., 'three_turn', 'waltz_jump').
            manual_phases: Optional manual phase boundaries (auto-detect if None).
            reference_path: Optional path to reference video (use store if None).

        Returns:
            AnalysisReport with metrics, recommendations, and scores.

        Raises:
            ValueError: If video cannot be processed or element type not supported.
        """
        # Validate element type
        from skating_biomechanics_ml.references import get_element_def

        element_def = get_element_def(element_type)
        if element_def is None:
            raise ValueError(f"Unknown element type: {element_type}")

        # Get video metadata
        meta = get_video_meta(video_path)

        # Stage 1: Detect person (optional if video has single person)
        bbox = self._get_detector().detect_first_frame(video_path)

        # Stage 2: Extract 2D poses
        raw_poses = self._get_pose_extractor().extract_video(video_path, crop=bbox)

        # Stage 3: Normalize poses
        normalized = self._get_normalizer().normalize(raw_poses)

        # Stage 3.5: Smooth poses (temporal filtering)
        if self._enable_smoothing:
            # Use phase-aware smoothing if manual phases are provided
            if manual_phases is not None:
                boundaries = [manual_phases.takeoff, manual_phases.peak, manual_phases.landing]
                boundaries = [b for b in boundaries if b > 0]  # Filter out zeros
                smoothed = self._get_smoother(meta.fps).smooth_phase_aware(normalized, boundaries)
            else:
                smoothed = self._get_smoother(meta.fps).smooth(normalized)
        else:
            smoothed = normalized

        # Stage 3.6: Detect blade edge states (both feet)
        from skating_biomechanics_ml.utils import BladeEdgeDetector

        blade_detector = BladeEdgeDetector(smoothing_window=3)
        blade_states_left = blade_detector.detect_sequence(smoothed, meta.fps, foot="left")
        blade_states_right = blade_detector.detect_sequence(smoothed, meta.fps, foot="right")
        blade_summary_left = blade_detector.get_blade_summary(blade_states_left)
        blade_summary_right = blade_detector.get_blade_summary(blade_states_right)

        # Stage 4: Detect phases (or use manual)
        if manual_phases is not None:
            phases = manual_phases
        else:
            phase_result = self._get_phase_detector().detect_phases(
                smoothed, meta.fps, element_type
            )
            phases = phase_result.phases

        # Stage 5: Compute biomechanics metrics
        analyzer = self._get_analyzer_factory()(element_def)
        metrics = analyzer.analyze(smoothed, phases, meta.fps)

        # Stage 6: Load reference and align (if available)
        dtw_distance: float | None = None
        if self._reference_store is not None:
            reference = self._reference_store.get_best_match(element_type)
            if reference is not None:
                aligner = self._get_aligner()
                dtw_distance = aligner.compute_distance(
                    normalized[phases.start : phases.end],
                    reference.poses[reference.phases.start : reference.phases.end],
                )

        # Stage 7: Generate recommendations
        recommender = self._get_recommender()
        recommendations = recommender.recommend(metrics, element_type)

        # Stage 8: Compute overall score
        overall_score = self._compute_overall_score(metrics)

        return AnalysisReport(
            element_type=element_type,
            phases=[phases],
            metrics=metrics,
            recommendations=recommendations,
            overall_score=overall_score,
            dtw_distance=dtw_distance,
            blade_summary_left=blade_summary_left,
            blade_summary_right=blade_summary_right,
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

        Usage:
            result = pipeline.segment_video(Path("coach_tutorial.mp4"))
            # Result contains list of ElementSegment objects
            # Can export each segment as reference .npz file
        """
        from skating_biomechanics_ml.segmentation import ElementSegmenter

        # Get video metadata
        meta = get_video_meta(video_path)

        # Stage 1: Detect person (optional if video has single person)
        bbox = self._get_detector().detect_first_frame(video_path)

        # Stage 2: Extract 2D poses
        raw_poses = self._get_pose_extractor().extract_video(video_path, crop=bbox)

        # Stage 3: Normalize poses
        normalized = self._get_normalizer().normalize(raw_poses)

        # Stage 3.5: Smooth poses (temporal filtering)
        if self._enable_smoothing:
            smoothed = self._get_smoother(meta.fps).smooth(normalized)
        else:
            smoothed = normalized

        # Stage 4: Segment video into elements
        segmenter = ElementSegmenter()
        segmentation = segmenter.segment(smoothed, video_path, meta)

        return segmentation

    def _get_detector(self) -> "PersonDetector":  # type: ignore[valid-type]
        """Lazy-load person detector."""
        if self._detector is None:
            from skating_biomechanics_ml.detection import PersonDetector

            self._detector = PersonDetector(model_size="n", confidence=0.5)
        return self._detector

    def _get_pose_extractor(self) -> "PoseExtractor":  # type: ignore[valid-type]
        """Lazy-load pose extractor (BlazePose with 33 keypoints)."""
        if self._pose_extractor is None:
            from skating_biomechanics_ml.pose_2d import BlazePoseExtractor

            # BlazePose parameters: model_path, min_detection_confidence, min_presence_confidence, num_poses
            self._pose_extractor = BlazePoseExtractor(
                min_detection_confidence=0.5,
                min_presence_confidence=0.5,
                num_poses=1,
            )
        return self._pose_extractor

    def _get_normalizer(self) -> "PoseNormalizer":  # type: ignore[valid-type]
        """Lazy-load pose normalizer."""
        if self._normalizer is None:
            from skating_biomechanics_ml.pose_2d import PoseNormalizer

            self._normalizer = PoseNormalizer(target_spine_length=0.4)
        return self._normalizer

    def _get_smoother(self, fps: float = 30.0) -> "PoseSmoother":  # type: ignore[valid-type]
        """Lazy-load pose smoother with One-Euro Filter."""
        if not self._enable_smoothing:
            # Return a no-op smoother that just returns input unchanged
            from skating_biomechanics_ml.utils.smoothing import PoseSmoother

            # Create a minimal config that doesn't smooth (high cutoff)
            from skating_biomechanics_ml.utils.smoothing import OneEuroFilterConfig

            config = OneEuroFilterConfig(min_cutoff=100.0, beta=0.0, freq=fps)
            return PoseSmoother(config=config, freq=fps)

        if self._smoother is None:
            from skating_biomechanics_ml.utils.smoothing import PoseSmoother, get_skating_optimized_config

            config = self._smoothing_config or get_skating_optimized_config(fps)
            self._smoother = PoseSmoother(config=config, freq=fps)
        return self._smoother

    def _get_phase_detector(self) -> "PhaseDetector":  # type: ignore[valid-type]
        """Lazy-load phase detector."""
        if self._phase_detector is None:
            from skating_biomechanics_ml.analysis import PhaseDetector

            self._phase_detector = PhaseDetector()
        return self._phase_detector

    def _get_analyzer_factory(self) -> type:
        """Get analyzer factory (returns BiomechanicsAnalyzer class)."""
        if self._analyzer_factory is None:
            from skating_biomechanics_ml.analysis import BiomechanicsAnalyzer

            self._analyzer_factory = BiomechanicsAnalyzer
        return self._analyzer_factory

    def _get_aligner(self) -> "MotionAligner | MotionDTWAligner":  # type: ignore[valid-type]
        """Lazy-load motion aligner (using phase-aware MotionDTW)."""
        if self._aligner is None:
            from skating_biomechanics_ml.alignment import MotionDTWAligner

            self._aligner = MotionDTWAligner(window_type="sakoechiba", window_size=0.2)
        return self._aligner

    def _get_recommender(self) -> "Recommender":  # type: ignore[valid-type]
        """Lazy-load recommender."""
        if self._recommender is None:
            from skating_biomechanics_ml.analysis import Recommender

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
            return 5.0  # Neutral score

        # Count good vs bad metrics
        good_count = sum(1 for m in metrics if m.is_good)
        total_count = len(metrics)

        if total_count == 0:
            return 5.0

        # Base score from ratio of good metrics
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
        for phase in report.phases:
            lines.append(f"  Начало:     {phase.start}")
            lines.append(f"  Отрыв:      {phase.takeoff}")
            lines.append(f"  Пик:        {phase.peak}")
            lines.append(f"  Приземление: {phase.landing}")
            lines.append(f"  Конец:       {phase.end}")

        # Metrics
        lines.append("\n--- Биомеханические метрики ---")
        for metric in report.metrics:
            status = "✓ ОК" if metric.is_good else "✗ ПЛОХО"
            ref_min, ref_max = metric.reference_range
            lines.append(
                f"  {metric.name}: {metric.value:.2f} {metric.unit} [{status}] "
                f"(референс: {ref_min:.2f}-{ref_max:.2f})"
            )

        # DTW distance
        if report.dtw_distance is not None:
            lines.append("\n--- Сходство с референсом ---")
            lines.append(f"  DTW-расстояние: {report.dtw_distance:.3f} (0 = идеально)")

        # Blade edge information
        if report.blade_summary_left or report.blade_summary_right:
            lines.append("\n--- Состояние лезвия ---")
            if report.blade_summary_left:
                lines.append(f"  Левая нога: {report.blade_summary_left.get('dominant_edge', 'unknown')}")
                if 'type_percentages' in report.blade_summary_left:
                    types = report.blade_summary_left['type_percentages']
                    lines.append(f"    Распределение: {types}")
            if report.blade_summary_right:
                lines.append(f"  Правая нога: {report.blade_summary_right.get('dominant_edge', 'unknown')}")
                if 'type_percentages' in report.blade_summary_right:
                    types = report.blade_summary_right['type_percentages']
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
            lines.append(f"\nОбщий балл: {report.overall_score:.1f} / 10")

        lines.append("=" * 60)

        return "\n".join(lines)
