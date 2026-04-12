"""Reference data builder from expert videos.

H3.6M Migration:
    Uses H3.6M 17-keypoint format as the primary format.
    2D extraction: RTMPoseExtractor (rtmlib BodyWithFeet)

This module provides tools to create reference datasets from
expert skating videos for comparison with user performances.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..types import (
    ElementPhase,
    ReferenceData,
    VideoMeta,
)  # type: ignore[import-untyped]
from ..utils.video import get_video_meta

if TYPE_CHECKING:
    from .normalizer import PoseNormalizer  # type: ignore[import-untyped]
    from .pose_estimation import RTMPoseExtractor as PoseExtractor  # type: ignore[import-untyped]


class ReferenceBuilder:
    """Build reference data from expert skating videos.

    H3.6M Architecture:
        - 2D poses: RTMPoseExtractor (17 keypoints, normalized [0,1])
    """

    def __init__(
        self,
        pose_extractor: "PoseExtractor",  # type: ignore[valid-type]
        normalizer: "PoseNormalizer",  # type: ignore[valid-type]
    ) -> None:
        """Initialize reference builder.

        Args:
            pose_extractor: RTMPoseExtractor instance.
            normalizer: PoseNormalizer instance.
        """
        self._pose_extractor = pose_extractor
        self._normalizer = normalizer

    def build_from_video(
        self,
        video_path: Path,
        element_type: str,
        phases: ElementPhase,
    ) -> ReferenceData:
        """Build reference data from a video.

        Args:
            video_path: Path to expert video.
            element_type: Type of skating element.
            phases: Phase boundaries (manually annotated).

        Returns:
            ReferenceData with normalized poses (H3.6M 17kp format) and metadata.
        """
        # Extract video metadata
        meta = get_video_meta(video_path)

        # Extract poses in H3.6M format (normalized [0,1]) with tracking
        extraction = self._pose_extractor.extract_video_tracked(video_path)
        normalized = self._normalizer.normalize(extraction.poses)

        return ReferenceData(
            element_type=element_type,
            name=video_path.name,
            poses=normalized,
            phases=phases,
            fps=meta.fps,
            meta=meta,
            source=str(video_path),
        )

    def save_reference(self, ref: ReferenceData, output_dir: Path) -> Path:
        """Save reference data to .npz file.

        Args:
            ref: ReferenceData to save.
            output_dir: Directory to save reference file.

        Returns:
            Path to saved .npz file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from element type and source
        source_path = Path(ref.source)
        filename = f"{ref.element_type}_{source_path.name}.npz"
        output_path = output_dir / filename

        # Save to .npz format
        np.savez_compressed(
            output_path,
            element_type=ref.element_type,
            poses=ref.poses,
            meta_fps=ref.meta.fps if ref.meta else 30.0,
            meta_width=ref.meta.width if ref.meta else 1920,
            meta_height=ref.meta.height if ref.meta else 1080,
            meta_num_frames=ref.meta.num_frames if ref.meta else len(ref.poses),
            meta_path=str(ref.meta.path) if ref.meta else "",
            phases_name=ref.phases.name,
            phases_start=ref.phases.start,
            phases_takeoff=ref.phases.takeoff,
            phases_peak=ref.phases.peak,
            phases_landing=ref.phases.landing,
            phases_end=ref.phases.end,
            source=ref.source,
        )

        return output_path

    def load_reference(self, path: Path) -> ReferenceData:
        """Load reference data from .npz file.

        Args:
            path: Path to .npz file.

        Returns:
            ReferenceData.
        """
        data = np.load(path, allow_pickle=True)

        # Reconstruct VideoMeta
        meta = VideoMeta(
            path=Path(str(data["meta_path"])),
            fps=float(data["meta_fps"]),
            width=int(data["meta_width"]),
            height=int(data["meta_height"]),
            num_frames=int(data["meta_num_frames"]),
        )

        # Reconstruct ElementPhase
        phases = ElementPhase(
            name=str(data["phases_name"]),
            start=int(data["phases_start"]),
            takeoff=int(data["phases_takeoff"]),
            peak=int(data["phases_peak"]),
            landing=int(data["phases_landing"]),
            end=int(data["phases_end"]),
        )

        return ReferenceData(
            element_type=str(data["element_type"]),
            name=str(data["source"]),
            poses=data["poses"],
            phases=phases,
            fps=float(data.get("fps", meta.fps if meta else 30.0)),
            meta=meta,
            source=str(data["source"]),
        )
