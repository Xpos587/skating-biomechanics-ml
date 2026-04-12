"""Skeleton data validation utilities."""

import numpy as np


class ValidationError:
    """Represents a single validation error."""

    def __init__(self, sample_id: str, field: str, message: str):
        """Initialize a validation error.

        Args:
            sample_id: Identifier for the sample being validated
            field: The field that failed validation
            message: Human-readable error message
        """
        self.sample_id = sample_id
        self.field = field
        self.message = message

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ValidationError(sample_id={self.sample_id!r}, field={self.field!r}, message={self.message!r})"


def validate_skeleton(
    poses: np.ndarray,
    sample_id: str = "unknown",
    min_frames: int = 5,
    max_coord: float = 10.0,
) -> list[ValidationError]:
    """Validate a skeleton sequence.

    Args:
        poses: Skeleton data of shape (T, V, C) or (T, V, C, 1)
            where T = time frames, V = keypoints (17 for H3.6M), C = channels
        sample_id: Identifier for the sample being validated
        min_frames: Minimum number of frames required
        max_coord: Maximum absolute coordinate value allowed

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Squeeze trailing dimension of size 1 (for 4D arrays like T, V, C, 1)
    if poses.ndim == 4 and poses.shape[-1] == 1:
        poses = poses.reshape(poses.shape[:-1])

    # Check ndim == 3
    if poses.ndim != 3:
        errors.append(
            ValidationError(
                sample_id=sample_id,
                field="ndim",
                message=f"Expected 3D array (T, V, C), got {poses.ndim}D with shape {poses.shape}",
            )
        )
        return errors  # Can't continue validation if shape is wrong

    T, V, C = poses.shape

    # Check minimum frames
    if T < min_frames:
        errors.append(
            ValidationError(
                sample_id=sample_id,
                field="frames",
                message=f"Too few frames: {T} < {min_frames}",
            )
        )

    # Check keypoint count (H3.6M format has 17 keypoints)
    if V != 17:
        errors.append(
            ValidationError(
                sample_id=sample_id,
                field="keypoints",
                message=f"Wrong number of keypoints: {V}, expected 17",
            )
        )

    # Check channels (at least x, y)
    if C < 2:
        errors.append(
            ValidationError(
                sample_id=sample_id,
                field="channels",
                message=f"Too few channels: {C}, expected at least 2",
            )
        )

    # Check for NaN values
    if np.isnan(poses).any():
        nan_count = np.isnan(poses).sum()
        errors.append(
            ValidationError(
                sample_id=sample_id,
                field="nan",
                message=f"Found {nan_count} NaN values",
            )
        )

    # Check for Inf values
    if np.isinf(poses).any():
        inf_count = np.isinf(poses).sum()
        errors.append(
            ValidationError(
                sample_id=sample_id,
                field="inf",
                message=f"Found {inf_count} Inf values",
            )
        )

    # Check not all zeros
    if np.all(poses == 0):
        errors.append(
            ValidationError(
                sample_id=sample_id,
                field="zeros",
                message="All values are zero",
            )
        )

    # Check coordinates within bounds
    if np.abs(poses).max() > max_coord:
        max_val = np.abs(poses).max()
        errors.append(
            ValidationError(
                sample_id=sample_id,
                field="bounds",
                message=f"Coordinate {max_val:.2f} exceeds bound ±{max_coord}",
            )
        )

    return errors
