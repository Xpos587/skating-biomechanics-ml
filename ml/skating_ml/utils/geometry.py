"""Geometric utilities for pose analysis."""

import numpy as np
from numpy.typing import NDArray

from ..types import FrameKeypoints, H36Key, NormalizedPose, TimeSeries


def angle_3pt(a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]) -> float:
    """Calculate angle ABC in degrees.

    Args:
        a: Point A coordinates (x, y).
        b: Vertex point B coordinates (x, y).
        c: Point C coordinates (x, y).

    Returns:
        Angle in degrees [0, 180].
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    # Vectors BA and BC
    ba = a - b
    bc = c - b

    # Cosine of angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)

    # Clamp to [-1, 1] to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return float(np.degrees(angle))


def distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Euclidean distance between two points.

    Args:
        a: Point A coordinates (x, y).
        b: Point B coordinates (x, y).

    Returns:
        Distance in same units as input.
    """
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def normalize_poses(
    raw: FrameKeypoints,
    spine_indices: tuple[int, int] = (H36Key.LSHOULDER, H36Key.LHIP),
    target_spine_length: float = 0.4,
) -> NormalizedPose:
    """Normalize poses via root-centering and scale normalization.

    1. Center pose at mid-hip (root) -> origin (0, 0)
    2. Scale so spine length equals target_spine_length

    Args:
        raw: Raw keypoints (num_frames, 17, 3) with x, y, confidence.
        spine_indices: (shoulder_idx, hip_idx) for spine length calculation.
        target_spine_length: Target spine length after normalization.

    Returns:
        NormalizedPose (num_frames, 17, 2) with centered, scaled coordinates.
    """
    if raw.shape[1] != 17:
        raise ValueError(f"Expected 17 keypoints (H3.6M format), got {raw.shape[1]}")

    num_frames = raw.shape[0]
    normalized = np.zeros((num_frames, 17, 2), dtype=np.float32)

    # Mid-hip point (between left and right hip)
    mid_hip_raw = (raw[:, H36Key.LHIP, :2] + raw[:, H36Key.RHIP, :2]) / 2

    for frame_idx in range(num_frames):
        frame_raw = raw[frame_idx]

        # 1. Root-centering: shift mid-hip to origin
        mid_hip = mid_hip_raw[frame_idx]
        centered = frame_raw[:, :2] - mid_hip

        # 2. Scale normalization
        shoulder_idx, hip_idx = spine_indices
        spine_vector = centered[shoulder_idx] - centered[hip_idx]
        spine_length = np.linalg.norm(spine_vector)

        scale = 1.0 if spine_length < 1e-6 else target_spine_length / spine_length

        normalized[frame_idx] = centered * scale

    return normalized


def smooth_signal(signal: TimeSeries, window: int = 5) -> TimeSeries:
    """Apply moving average smoothing to signal.

    Args:
        signal: Input signal (num_frames,).
        window: Window size for moving average (must be odd).

    Returns:
        Smoothed signal (num_frames,).
    """
    if window < 1:
        return signal

    if window % 2 == 0:
        window += 1

    if len(signal) < window:
        return signal.copy()

    # Use numpy convolution for efficient moving average
    kernel = np.ones(window) / window
    smoothed = np.convolve(signal, kernel, mode="same")

    return smoothed.astype(np.float32)


def get_mid_hip(poses: NormalizedPose) -> NDArray[np.float32]:
    """Calculate mid-hip point for each frame.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).

    Returns:
        Mid-hip coordinates (num_frames, 2).
    """
    return ((poses[:, H36Key.LHIP, :] + poses[:, H36Key.RHIP, :]) / 2).astype(np.float32)


def get_mid_shoulder(poses: NormalizedPose) -> NDArray[np.float32]:
    """Calculate mid-shoulder point for each frame.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).

    Returns:
        Mid-shoulder coordinates (num_frames, 2).
    """
    return ((poses[:, H36Key.LSHOULDER, :] + poses[:, H36Key.RSHOULDER, :]) / 2).astype(np.float32)


def calculate_center_of_mass(poses: NormalizedPose, frame_idx: int) -> float:
    """Calculate Center of Mass (CoM) Y-coordinate for a single frame.

    Uses anthropometric segment mass ratios from Dempster (1955) and
    Zatsiorsky (2002). The CoM is the weighted average of body segment
    positions: CoM = (1/M) * sum(m_i * p_i)

    This provides a physics-accurate measure of jump height, independent
    of landing pose. The hip-only method has 60% error for low jumps due
    to bent-knee landings artificially increasing flight time.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).
        frame_idx: Frame index to calculate CoM for.

    Returns:
        CoM Y-coordinate in normalized units (lower = higher position).

    Segment mass ratios (relative to total body mass):
        - Head: 0.081
        - Torso: 0.497
        - Upper arms: 0.050 each
        - Forearms+hands: 0.030 each
        - Thighs: 0.100 each
        - Shins+feet: 0.161 each
    """
    pose = poses[frame_idx]

    # Head (HEAD keypoint in H3.6M format)
    head = pose[H36Key.HEAD]
    head_mass = 0.081

    # Torso (mid-shoulder to mid-hip midpoint)
    torso = (
        pose[H36Key.LSHOULDER] + pose[H36Key.RSHOULDER] + pose[H36Key.LHIP] + pose[H36Key.RHIP]
    ) / 4
    torso_mass = 0.497

    # Arms (elbow-wrist midpoint for upper arm, wrist for forearm)
    l_upper_arm = (pose[H36Key.LSHOULDER] + pose[H36Key.LELBOW]) / 2
    r_upper_arm = (pose[H36Key.RSHOULDER] + pose[H36Key.RELBOW]) / 2
    l_forearm = (pose[H36Key.LELBOW] + pose[H36Key.LWRIST]) / 2
    r_forearm = (pose[H36Key.RELBOW] + pose[H36Key.RWRIST]) / 2
    arm_mass_each = 0.050

    # Thighs (hip-knee midpoint)
    l_thigh = (pose[H36Key.LHIP] + pose[H36Key.LKNEE]) / 2
    r_thigh = (pose[H36Key.RHIP] + pose[H36Key.RKNEE]) / 2
    thigh_mass_each = 0.100

    # Shins+feet (knee-foot midpoint)
    l_leg = (pose[H36Key.LKNEE] + pose[H36Key.LFOOT]) / 2
    r_leg = (pose[H36Key.RKNEE] + pose[H36Key.RFOOT]) / 2
    leg_mass_each = 0.161

    # Weighted sum of Y-coordinates only (for height)
    com_y = (
        head_mass * head[1]
        + torso_mass * torso[1]
        + arm_mass_each * (l_upper_arm[1] + r_upper_arm[1] + l_forearm[1] + r_forearm[1])
        + thigh_mass_each * (l_thigh[1] + r_thigh[1])
        + leg_mass_each * (l_leg[1] + r_leg[1])
    )

    return float(com_y)


def calculate_com_trajectory(poses: NormalizedPose) -> NDArray[np.float32]:
    """Calculate Center of Mass trajectory for entire pose sequence.

    Args:
        poses: NormalizedPose (num_frames, 17, 2).

    Returns:
        CoM Y-coordinates (num_frames,) in normalized units.
    """
    num_frames = len(poses)
    com_trajectory = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        com_trajectory[i] = calculate_center_of_mass(poses, i)

    return com_trajectory


# ---------------------------------------------------------------------------
# Foot angle functions for blade edge detection (HALPE26 foot keypoints)
# ---------------------------------------------------------------------------


def segment_angle(a: NDArray, b: NDArray) -> float:
    """Angle of segment AB relative to horizontal, anticlockwise, degrees.

    Convention matches Sports2D: positive = upward from horizontal.

    Args:
        a: Start point (x, y) — can be any shape broadcastable to (2,).
        b: End point (x, y).

    Returns:
        Angle in degrees [-180, 180].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return float(np.degrees(np.arctan2(-dy, dx)))  # negate dy: image coords y-down


def foot_angle(heel: NDArray, big_toe: NDArray) -> float:
    """Foot angle for blade edge detection.

    Measures big_toe -> heel relative to horizontal.  In image coordinates
    (y-axis pointing down), a foot pointing right returns ~0 degrees.

    Args:
        heel: Heel keypoint (x, y).
        big_toe: Big toe keypoint (x, y).

    Returns:
        Angle in degrees [-180, 180].
    """
    return segment_angle(big_toe, heel)


def ankle_dorsiflexion(
    knee: NDArray,
    ankle: NDArray,
    big_toe: NDArray,
    heel: NDArray,
) -> float:
    """Ankle dorsiflexion: knee -> ankle -> foot_midpoint angle (3-point).

    The foot midpoint is the average of big_toe and heel positions.

    Args:
        knee: Knee keypoint (x, y).
        ankle: Ankle keypoint (x, y).
        big_toe: Big toe keypoint (x, y).
        heel: Heel keypoint (x, y).

    Returns:
        Angle in degrees [0, 180].
    """
    foot_mid = (np.asarray(big_toe, dtype=np.float64) + np.asarray(heel, dtype=np.float64)) / 2
    return angle_3pt(knee, ankle, foot_mid)


def compute_foot_angles_series(
    foot_keypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute foot angles from a time series of HALPE26 foot keypoints.

    Args:
        foot_keypoints: (N, 6, 3) array with columns
            [L_Heel, L_BigToe, L_SmallToe, R_Heel, R_BigToe, R_SmallToe]
            and channels (x, y, confidence).

    Returns:
        Tuple of (left_angles, right_angles), each shape (N,) in degrees.
        Frames where confidence < 0.3 for heel or big_toe are set to NaN.
    """
    n = foot_keypoints.shape[0]
    left_angles = np.full(n, np.nan, dtype=np.float32)
    right_angles = np.full(n, np.nan, dtype=np.float32)

    # Left foot: L_Heel=0, L_BigToe=1
    l_heel = foot_keypoints[:, 0, :]  # (N, 3)
    l_big_toe = foot_keypoints[:, 1, :]

    # Right foot: R_Heel=3, R_BigToe=4
    r_heel = foot_keypoints[:, 3, :]
    r_big_toe = foot_keypoints[:, 4, :]

    # Confidence mask: both heel and big_toe must be > 0.3
    l_valid = (l_heel[:, 2] >= 0.3) & (l_big_toe[:, 2] >= 0.3)
    r_valid = (r_heel[:, 2] >= 0.3) & (r_big_toe[:, 2] >= 0.3)

    for i in range(n):
        if l_valid[i]:
            left_angles[i] = foot_angle(l_heel[i, :2], l_big_toe[i, :2])
        if r_valid[i]:
            right_angles[i] = foot_angle(r_heel[i, :2], r_big_toe[i, :2])

    return left_angles, right_angles


# ---------------------------------------------------------------------------
# Visible side detection (Sports2D-inspired)
# ---------------------------------------------------------------------------


def detect_visible_side(
    foot_keypoints: np.ndarray,
    conf_threshold: float = 0.3,
) -> str | None:
    """Detect which side of the body is facing the camera.

    Uses HALPE26 foot keypoints (heel + big_toe) to determine orientation.
    If big_toe is to the RIGHT of heel → right side visible.
    If big_toe is to the LEFT of heel → left side visible.

    Args:
        foot_keypoints: (1, 6, 3) or (N, 6, 3) foot keypoints in pixel coords.
            Columns: [L_Heel, L_BigToe, L_SmallToe, R_Heel, R_BigToe, R_SmallToe]
            Channels: [x, y, confidence].
        conf_threshold: Minimum confidence for valid keypoints.

    Returns:
        "left", "right", or None if insufficient data.
    """
    # Use first frame if multi-frame
    if foot_keypoints.ndim == 3:
        # Aggregate: median orientation across frames
        orientations = []
        for i in range(foot_keypoints.shape[0]):
            fp = foot_keypoints[i]
            l_conf = fp[1, 2]  # L_BigToe confidence
            r_conf = fp[4, 2]  # R_BigToe confidence
            if l_conf < conf_threshold or r_conf < conf_threshold:
                continue
            l_orientation = fp[1, 0] - fp[0, 0]  # L_BigToe.x - L_Heel.x
            r_orientation = fp[4, 0] - fp[3, 0]  # R_BigToe.x - R_Heel.x
            orientations.append(l_orientation + r_orientation)
        if not orientations:
            return None
        return "right" if np.median(orientations) >= 0 else "left"
    return None


# ---------------------------------------------------------------------------
# Floor angle estimation (Sports2D-inspired)
# ---------------------------------------------------------------------------


def estimate_floor_angle(
    foot_positions: np.ndarray,
) -> float:
    """Estimate floor angle from foot positions.

    Fits a line through foot positions and returns the angle of that line
    relative to horizontal. Used to correct segment angles for camera tilt.

    Args:
        foot_positions: (N, 2) array of foot (x, y) positions in pixels.

    Returns:
        Floor angle in degrees. 0 = horizontal. Positive = upward slope in image coords.
        Returns 0.0 if fewer than 2 points.
    """
    if len(foot_positions) < 2:
        return 0.0

    # Fit line: y = m*x + b
    coeffs = np.polyfit(foot_positions[:, 0], foot_positions[:, 1], 1)
    # coeffs[0] = slope (dy/dx), coeffs[1] = intercept
    # Angle of slope relative to horizontal (image coords: y-down)
    # In image coords: positive slope (dy/dx > 0) = Y increases with X = downward
    # We return the angle as it appears visually in the image
    angle = np.degrees(np.arctan(coeffs[0]))
    return float(angle)
