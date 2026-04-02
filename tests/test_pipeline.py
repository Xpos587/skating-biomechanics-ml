"""Integration tests for the full analysis pipeline.

H3.6M Migration:
    Pipeline now uses H3.6M 17-keypoint format as primary.
    2D: H36MExtractor, 3D: AthletePose3DExtractor
"""

import pytest

from src.pipeline import AnalysisPipeline
from src.types import ElementPhase


@pytest.mark.integration
class TestAnalysisPipeline:
    """Integration tests for the full pipeline."""

    def test_pipeline_initialization(self):
        """Should initialize without errors."""
        pipeline = AnalysisPipeline()

        assert pipeline is not None

    def test_pipeline_without_reference(self):
        """Should work without reference store."""
        pipeline = AnalysisPipeline(reference_store=None)

        assert pipeline._reference_store is None

    def test_analyze_with_mock_data(self, sample_normalized_poses):
        """Should analyze mock pose data."""
        pipeline = AnalysisPipeline(reference_store=None)

        # Create mock phases
        ElementPhase(
            name="three_turn",
            start=0,
            takeoff=0,
            peak=1,
            landing=0,
            end=2,
        )

        # Note: This test would require mocking video loading
        # For now, just test the pipeline structure
        assert pipeline is not None

    def test_format_report(self):
        """Should format report correctly."""
        pipeline = AnalysisPipeline()

        from src.types import AnalysisReport, MetricResult

        # Create mock report
        phases = ElementPhase(
            name="test",
            start=0,
            takeoff=10,
            peak=20,
            landing=30,
            end=40,
        )

        metrics = [
            MetricResult(
                name="test_metric",
                value=0.5,
                unit="s",
                is_good=True,
                reference_range=(0.3, 0.7),
            )
        ]

        report = AnalysisReport(
            element_type="test_element",
            phases=[phases],
            metrics=metrics,
            recommendations=["Test recommendation"],
            overall_score=7.5,
            dtw_distance=0.15,
        )

        formatted = pipeline.format_report(report)

        assert "АНАЛИЗ" in formatted
        assert "TEST_ELEMENT" in formatted
        assert "7.5" in formatted
        assert "Test recommendation" in formatted

    def test_compute_overall_score(self):
        """Should compute overall score correctly."""
        pipeline = AnalysisPipeline()

        from src.types import MetricResult

        # All good metrics
        metrics_good = [
            MetricResult(
                name="metric1",
                value=0.5,
                unit="s",
                is_good=True,
                reference_range=(0.3, 0.7),
            ),
            MetricResult(
                name="metric2",
                value=100,
                unit="deg",
                is_good=True,
                reference_range=(90, 110),
            ),
        ]

        score = pipeline._compute_overall_score(metrics_good)
        assert score == 10.0

        # Half good, half bad
        metrics_mixed = [
            MetricResult(
                name="metric1",
                value=0.5,
                unit="s",
                is_good=True,
                reference_range=(0.3, 0.7),
            ),
            MetricResult(
                name="metric2",
                value=50,
                unit="deg",
                is_good=False,
                reference_range=(90, 110),
            ),
        ]

        score = pipeline._compute_overall_score(metrics_mixed)
        assert score == 5.0

        # All bad metrics
        metrics_bad = [
            MetricResult(
                name="metric1",
                value=0.1,
                unit="s",
                is_good=False,
                reference_range=(0.3, 0.7),
            ),
            MetricResult(
                name="metric2",
                value=50,
                unit="deg",
                is_good=False,
                reference_range=(90, 110),
            ),
        ]

        score = pipeline._compute_overall_score(metrics_bad)
        assert score == 0.0

    def test_compute_overall_score_empty_metrics(self):
        """Should return neutral score for empty metrics."""
        pipeline = AnalysisPipeline()

        score = pipeline._compute_overall_score([])
        assert score == 5.0


@pytest.mark.integration
class TestPipelineLazyLoading:
    """Test lazy loading of pipeline components.

    H3.6M Migration: Updated for new variable names.
    """

    def test_detector_lazy_load(self):
        """Should lazy-load person detector."""
        pipeline = AnalysisPipeline()

        assert pipeline._detector is None
        detector = pipeline._get_detector()
        assert detector is not None
        assert pipeline._detector is not None

    def test_pose_2d_extractor_lazy_load(self):
        """Should lazy-load 2D pose extractor (H3.6M format)."""
        from pathlib import Path

        # Skip if YOLO model not available
        model_file = Path("yolo26n-pose.pt")
        if not model_file.exists():
            pytest.skip("YOLO model not available")

        pipeline = AnalysisPipeline()

        assert pipeline._pose_2d_extractor is None
        extractor = pipeline._get_pose_2d_extractor()
        assert extractor is not None
        assert pipeline._pose_2d_extractor is not None

    def test_pose_3d_extractor_lazy_load(self):
        """Should lazy-load 3D pose lifter (MotionAGFormer)."""
        pipeline = AnalysisPipeline()

        assert pipeline._pose_3d_extractor is None
        extractor = pipeline._get_pose_3d_extractor()
        assert extractor is not None
        assert pipeline._pose_3d_extractor is not None

    def test_normalizer_lazy_load(self):
        """Should lazy-load normalizer."""
        pipeline = AnalysisPipeline()

        assert pipeline._normalizer is None
        normalizer = pipeline._get_normalizer()
        assert normalizer is not None
        assert pipeline._normalizer is not None

    def test_phase_detector_lazy_load(self):
        """Should lazy-load phase detector."""
        pipeline = AnalysisPipeline()

        assert pipeline._phase_detector is None
        detector = pipeline._get_phase_detector()
        assert detector is not None
        assert pipeline._phase_detector is not None

    def test_aligner_lazy_load(self):
        """Should lazy-load aligner."""
        pipeline = AnalysisPipeline()

        assert pipeline._aligner is None
        aligner = pipeline._get_aligner()
        assert aligner is not None
        assert pipeline._aligner is not None

    def test_recommender_lazy_load(self):
        """Should lazy-load recommender."""
        pipeline = AnalysisPipeline()

        assert pipeline._recommender is None
        recommender = pipeline._get_recommender()
        assert recommender is not None
        assert pipeline._recommender is not None
