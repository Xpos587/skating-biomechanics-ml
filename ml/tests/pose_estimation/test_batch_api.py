"""Simple API tests for batch RTMO integration."""

from __future__ import annotations


def test_extract_video_tracked_has_batch_params():
    """extract_video_tracked should accept use_batch parameter."""
    import inspect

    from src.pose_estimation.pose_extractor import PoseExtractor

    sig = inspect.signature(PoseExtractor.extract_video_tracked)
    params = sig.parameters

    assert "use_batch" in params, "use_batch parameter missing"
    assert "batch_size" in params, "batch_size parameter missing"

    # Check default values
    assert params["use_batch"].default is False, "use_batch should default to False"
    assert params["batch_size"].default == 8, "batch_size should default to 8"


def test_extract_batch_method_exists():
    """PoseExtractor should have _extract_batch method."""
    from src.pose_estimation.pose_extractor import PoseExtractor

    assert hasattr(PoseExtractor, "_extract_batch"), "_extract_batch method missing"


def test_extract_batch_signature():
    """_extract_batch should have correct signature."""
    import inspect

    from src.pose_estimation.pose_extractor import PoseExtractor

    sig = inspect.signature(PoseExtractor._extract_batch)
    params = list(sig.parameters.keys())

    expected_params = ["self", "video_path", "person_click", "progress_cb", "batch_size"]
    for param in expected_params:
        assert param in params, f"Parameter {param} missing from _extract_batch"


def test_batch_method_returns_tracked_extraction():
    """_extract_batch should return TrackedExtraction type."""
    import inspect

    from src.pose_estimation.pose_extractor import PoseExtractor
    from src.types import TrackedExtraction

    sig = inspect.signature(PoseExtractor._extract_batch)
    return_annotation = sig.return_annotation

    # Check if return annotation mentions TrackedExtraction
    assert "TrackedExtraction" in str(return_annotation), (
        f"_extract_batch should return TrackedExtraction, got {return_annotation}"
    )


if __name__ == "__main__":
    test_extract_video_tracked_has_batch_params()
    test_extract_batch_method_exists()
    test_extract_batch_signature()
    test_batch_method_returns_tracked_extraction()
    print("All tests passed!")
