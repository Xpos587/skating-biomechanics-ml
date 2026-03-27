"""Reference data builder from expert videos.

This module provides tools to create reference datasets from
expert skating videos for comparison with user performances.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from skating_biomechanics_ml.types import (
    ElementPhase,
    NormalizedPose,
    ReferenceData,
    VideoMeta,
)
from skating_biomechanics_ml.utils.video import get_video_meta

if TYPE_CHECKING:
    from skating_biomechanics_ml.pose_2d import PoseExtractor, PoseNormalizer


class ReferenceBuilder:
    """Build reference data from expert skating videos."""

    def __init__(
        self,
        pose_extractor: "PoseExtractor",  # type: ignore[valid-type]
        normalizer: "PoseNormalizer",  # type: ignore[valid-type]
    ) -> None:
        """Initialize reference builder.

        Args:
            pose_extractor: PoseExtractor instance.
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
            ReferenceData with normalized poses and metadata.
        """
        # Extract video metadata
        meta = get_video_meta(video_path)

        # Extract poses
        raw_poses = self._pose_extractor.extract_video(video_path)
        normalized = self._normalizer.normalize(raw_poses)

        return ReferenceData(
            element_type=element_type,
            poses=normalized,
            meta=meta,
            phases=phases,
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
        filename = f"{ref.element_type}_{ref.source.name}.npz"
        output_path = output_dir / filename

        # Save to .npz format
        np.savez_compressed(
            output_path,
            element_type=ref.element_type,
            poses=ref.poses,
            meta_fps=ref.meta.fps,
            meta_width=ref.meta.width,
            meta_height=ref.meta.height,
            meta_num_frames=ref.meta.num_frames,
            meta_path=str(ref.meta.path),
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
            poses=data["poses"],
            meta=meta,
            phases=phases,
            source=str(data["source"]),
        )
