"""Main analysis pipeline orchestrator.

H3.6M Architecture:
    This pipeline uses H3.6M 17-keypoint format as the primary format.
    2D extraction: H36MExtractor (YOLO26-Pose backend)
    3D lifting: AthletePose3DExtractor (MotionAGFormer)

Pipeline stages:
    1. Extract & track: H36MExtractor.extract_video_tracked() + gap fill + spatial ref
    2. Normalization
    3. Temporal smoothing (One-Euro Filter)
    4. Phase detection
    5. Biomechanics metrics
    6. 3D pose estimation (optional, for blade detection & physics)
    7. Reference comparison (DTW)
    8. Recommendations
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .types import AnalysisReport, ElementPhase, PersonClick, SegmentationResult
from .utils.video import VideoMeta, get_video_meta

if TYPE_CHECKING:
    from .alignment import MotionAligner, MotionDTWAligner
    from .analysis.phase_detector import PhaseDetector
    from .analysis.recommender import Recommender
    from .detection import PersonDetector
    from .pose_3d import AthletePose3DExtractor
    from .pose_estimation import H36MExtractor
    from .pose_estimation.normalizer import PoseNormalizer
    from .references import ReferenceStore
    from .utils.smoothing import OneEuroFilterConfig, PoseSmoother


class AnalysisPipeline:
    """Main pipeline for skating technique analysis.

    H3.6M Architecture:
        - 2D poses: H36MExtractor (17 keypoints, normalized [0,1], YOLO26-Pose backend)
        - 3D poses: AthletePose3DExtractor (MotionAGFormer)
        - No intermediate 33kp storage
    """

    def __init__(
        self,
        reference_store: "ReferenceStore | None" = None,  # type: ignore[valid-type]
        use_gpu: bool = True,
        enable_smoothing: bool = True,
        smoothing_config: "OneEuroFilterConfig | None" = None,  # type: ignore[valid-type]
        person_click: PersonClick | None = None,
        reestimate_camera: bool = False,
    ) -> None:
        """Initialize analysis pipeline.

        Args:
            reference_store: ReferenceStore for loading expert references.
            use_gpu: Whether to use GPU acceleration (when available).
            enable_smoothing: Whether to apply One-Euro Filter temporal smoothing.
            smoothing_config: Optional custom smoothing configuration.
            person_click: Optional click point to select target person in multi-person videos.
            reestimate_camera: Enable per-frame camera re-estimation for moving cameras.
        """
        self._reference_store = reference_store
        self._use_gpu = use_gpu
        self._enable_smoothing = enable_smoothing
        self._smoothing_config = smoothing_config
        self._person_click = person_click
        self._reestimate_camera = reestimate_camera

        # Components will be lazy-loaded
        self._detector: PersonDetector | None = None  # type: ignore[valid-type]
        self._pose_2d_extractor: H36MExtractor | None = None  # type: ignore[valid-type]
        self._pose_3d_extractor: AthletePose3DExtractor | None = None  # type: ignore[valid-type]
        self._normalizer: PoseNormalizer | None = None  # type: ignore[valid-type]
        self._smoother: PoseSmoother | None = None  # type: ignore[valid-type]
        self._phase_detector: PhaseDetector | None = None  # type: ignore[valid-type]
        self._analyzer_factory: type | None = None
        self._aligner: MotionAligner | MotionDTWAligner | None = None  # type: ignore[valid-type]
        self._recommender: Recommender | None = None  # type: ignore[valid-type]

    def _extract_and_track(
        self, video_path: Path, meta: VideoMeta
    ) -> tuple[np.ndarray, int]:
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
        # 1. Tracked extraction
        extraction = self._get_pose_2d_extractor().extract_video_tracked(
            video_path, person_click=self._person_click
        )

        # 2. Skip pre-roll (trim leading NaN frames before first detection)
        frame_offset = extraction.first_detection_frame
        poses = extraction.poses[frame_offset:]
        valid = extraction.valid_mask()[frame_offset:]

        # 3. Gap filling
        from .utils.gap_filling import GapFiller

        filler = GapFiller()
        filled, _report = filler.fill_gaps(poses, valid)

        # 4. Spatial reference / camera compensation
        if self._reestimate_camera:
            from .detection.spatial_reference import (
                compensate_poses_per_frame,
                estimate_pose_sequence,
            )

            camera_poses = estimate_pose_sequence(
                str(video_path), interval=30, fps=meta.fps
            )
            compensated = compensate_poses_per_frame(
                filled,
                camera_poses,
                video_width=meta.width,
                video_height=meta.height,
            )
        else:
            # Single-frame estimation (existing behavior)
            import cv2

            from .detection.spatial_reference import SpatialReferenceDetector

            spatial_detector = SpatialReferenceDetector()
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            if ret:
                camera_pose = spatial_detector.estimate_pose(first_frame)
            else:
                camera_pose = SpatialReferenceDetector.CameraPose()
            cap.release()

            if camera_pose.confidence > 0.1:
                # Convert to pixels, compensate, convert back
                poses_px = filled[:, :, :2] * np.array([meta.width, meta.height])
                poses_with_conf = np.dstack([poses_px, filled[:, :, 2]])
                compensated_px = spatial_detector.compensate_poses(
                    poses_with_conf, camera_pose
                )
                compensated = compensated_px[:, :, :2] / np.array(
                    [meta.width, meta.height]
                )
                compensated = np.dstack(
                    [compensated, compensated_px[:, :, 2:3]]
                )
            else:
                compensated = filled

        return compensated, frame_offset

    def analyze(  # noqa: PLR0912, PLR0915
        self,
        video_path: Path,
        element_type: str | None = None,
        manual_phases: ElementPhase | None = None,
        reference_path: Path | None = None,  # noqa: ARG002
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

        # Get video metadata
        meta = get_video_meta(video_path)

        # Stage 1-2.6: Extract poses with tracking, gap filling, spatial compensation
        compensated_h36m, _frame_offset = self._extract_and_track(video_path, meta)

        # Stage 3: Normalize poses

        normalized = self._get_normalizer().normalize(compensated_h36m)

        # Stage 3.5: Smooth poses (temporal filtering)
        if self._enable_smoothing:
            if manual_phases is not None:
                boundaries = [manual_phases.takeoff, manual_phases.peak, manual_phases.landing]
                boundaries = [b for b in boundaries if b > 0]
                smoothed = self._get_smoother(meta.fps).smooth_phase_aware(normalized, boundaries)
            else:
                smoothed = self._get_smoother(meta.fps).smooth(normalized)
        else:
            smoothed = normalized

        # Stage 3.6: 3D pose estimation (for blade detection & physics)
        poses_3d = None
        blade_summary_left = None
        blade_summary_right = None
        try:
            # Use MotionAGFormer for 3D lifting (H3.6M format)
            poses_3d = self._get_pose_3d_extractor().extract_sequence(smoothed)

            from .detection.blade_edge_detector_3d import BladeEdgeDetector3D  # noqa: PLC0415

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
                "outside": sum(1 for s in blade_states_right if s.blade_type.value == "outside"),
                "flat": sum(1 for s in blade_states_right if s.blade_type.value == "flat"),
            }
        except Exception:
            # 3D lifting is optional, don't fail if it errors
            pass

        # Stage 4-7: Element-specific analysis (only when element_type provided)
        if element_type is not None and element_def is not None:
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

            # Stage 6.5: Physics calculations (3D pose + biomechanics)
            physics_dict: dict = {}
            if poses_3d is not None:
                try:
                    from .analysis import PhysicsEngine  # noqa: PLC0415

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

            # Stage 7: Generate recommendations
            recommender = self._get_recommender()
            recommendations = recommender.recommend(metrics, element_type)

            # Stage 8: Compute overall score
            overall_score = self._compute_overall_score(metrics)
        else:
            # No element type specified — poses + visualization only
            phases = None
            metrics = []
            recommendations = []
            overall_score = None
            dtw_distance = None
            physics_dict = {}

        return AnalysisReport(
            element_type=element_type or "unknown",
            phases=[phases] if phases else [],
            metrics=metrics,
            recommendations=recommendations,
            overall_score=overall_score,
            dtw_distance=dtw_distance if dtw_distance else 0.0,
            blade_summary_left=blade_summary_left,
            blade_summary_right=blade_summary_right,
            physics=physics_dict,
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
        from .analysis import element_segmenter  # noqa: PLC0415

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

    def _get_detector(self) -> "PersonDetector":  # type: ignore[valid-type]
        """Lazy-load person detector."""
        if self._detector is None:
            from .detection import person_detector  # noqa: PLC0415

            PersonDetector = person_detector.PersonDetector

            self._detector = PersonDetector(model_size="n", confidence=0.5)
        return self._detector

    def _get_pose_2d_extractor(self) -> "H36MExtractor":  # type: ignore[valid-type]
        """Lazy-load 2D pose extractor (H3.6M 17kp format with YOLO26-Pose backend)."""
        if self._pose_2d_extractor is None:
            from .pose_estimation import H36MExtractor  # noqa: PLC0415

            self._pose_2d_extractor = H36MExtractor(
                output_format="normalized",  # [0,1] coordinates
            )
        return self._pose_2d_extractor

    def _get_pose_3d_extractor(self) -> "AthletePose3DExtractor":  # type: ignore[valid-type]
        """Lazy-load 3D pose lifter (MotionAGFormer)."""
        if self._pose_3d_extractor is None:
            from .pose_3d import AthletePose3DExtractor  # noqa: PLC0415

            model_path = "data/models/motionagformer-s-ap3d.pth.tr"
            self._pose_3d_extractor = AthletePose3DExtractor(
                model_path=Path(model_path) if Path(model_path).exists() else None,
                use_simple=True,  # Fallback to biomechanics estimator
            )
        return self._pose_3d_extractor

    def _get_normalizer(self) -> "PoseNormalizer":  # type: ignore[valid-type]
        """Lazy-load pose normalizer."""
        if self._normalizer is None:
            from .pose_estimation import normalizer  # noqa: PLC0415

            PoseNormalizer = normalizer.PoseNormalizer

            self._normalizer = PoseNormalizer(target_spine_length=0.4)
        return self._normalizer

    def _get_smoother(self, fps: float = 30.0) -> "PoseSmoother":  # type: ignore[valid-type]
        """Lazy-load pose smoother with One-Euro Filter."""
        if not self._enable_smoothing:
            from .utils.smoothing import OneEuroFilterConfig, PoseSmoother  # noqa: PLC0415

            config = OneEuroFilterConfig(min_cutoff=100.0, beta=0.0, freq=fps)
            return PoseSmoother(config=config, freq=fps)

        if self._smoother is None:
            from .utils.smoothing import (  # noqa: PLC0415
                PoseSmoother,
                get_skating_optimized_config,
            )

            config = self._smoothing_config or get_skating_optimized_config(fps)
            self._smoother = PoseSmoother(config=config, freq=fps)
        return self._smoother

    def _get_phase_detector(self) -> "PhaseDetector":  # type: ignore[valid-type]
        """Lazy-load phase detector."""
        if self._phase_detector is None:
            from .analysis import phase_detector  # noqa: PLC0415

            PhaseDetector = phase_detector.PhaseDetector

            self._phase_detector = PhaseDetector()
        return self._phase_detector

    def _get_analyzer_factory(self) -> type:
        """Get analyzer factory (returns BiomechanicsAnalyzer class)."""
        if self._analyzer_factory is None:
            from .analysis import metrics  # noqa: PLC0415

            BiomechanicsAnalyzer = metrics.BiomechanicsAnalyzer

            self._analyzer_factory = BiomechanicsAnalyzer
        return self._analyzer_factory

    def _get_aligner(self) -> "MotionAligner | MotionDTWAligner":  # type: ignore[valid-type]
        """Lazy-load motion aligner (using phase-aware MotionDTW)."""
        if self._aligner is None:
            from .alignment import motion_dtw  # noqa: PLC0415

            MotionDTWAligner = motion_dtw.MotionDTWAligner

            self._aligner = MotionDTWAligner(window_type="sakoechiba", window_size=0.2)
        return self._aligner

    def _get_recommender(self) -> "Recommender":  # type: ignore[valid-type]
        """Lazy-load recommender."""
        if self._recommender is None:
            from .analysis import recommender  # noqa: PLC0415

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
        for phase in report.phases:
            lines.append(f"  Начало:     {phase.start}")
            lines.append(f"  Отрыв:      {phase.takeoff}")
            lines.append(f"  Пик:        {phase.peak}")
            lines.append(f"  Приземление: {phase.landing}")
            lines.append(f"  Конец:       {phase.end}")

        # Metrics
        lines.append("\n--- Биомеханические метрики ---")
        for metric in report.metrics:
            status = "\u2713 \u041e\u041a" if metric.is_good else "\u2717 \u041f\u041b\u041e\u0425\u041e"
            ref_min, ref_max = metric.reference_range
            lines.append(
                f"  {metric.name}: {metric.value:.2f} {metric.unit} [{status}] "
                f"(референс: {ref_min:.2f}-{ref_max:.2f})"
            )

        # DTW distance
        if report.dtw_distance is not None:
            lines.append("\n--- \u0421\u0445\u043e\u0434\u0441\u0442\u0432\u043e \u0441 \u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043e\u043c ---")
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
            lines.append(f"\n\u041e\u0431\u0449\u0438\u0439 \u0431\u0430\u043b\u043b: {report.overall_score:.1f} / 10")

        lines.append("=" * 60)

        return "\n".join(lines)
