"""Biomechanics metrics computation for figure skating.

This module provides metrics for analyzing skating technique,
including joint angles, airtime, rotation speed, and edge quality.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from ..types import (
    ElementPhase,
    H36Key,
    MetricResult,
    NormalizedPose,
    TimeSeries,
)
from ..utils.geometry import (
    _angle_3pt_rad,
    angle_3pt,
    angle_3pt_batch,
    calculate_com_trajectory,
)

if TYPE_CHECKING:
    from .element_defs import ElementDef


@njit(cache=True, fastmath=True)
def _compute_knee_angle_series_numba(
    poses: np.ndarray,
    hip_idx: int,
    knee_idx: int,
    foot_idx: int,
) -> np.ndarray:
    """Compute knee angle series (jitted).

    Args:
        poses: (num_frames, 17, 2) pose array.
        hip_idx: Hip keypoint index (will be converted to int).
        knee_idx: Knee keypoint index (will be converted to int).
        foot_idx: Foot keypoint index (will be converted to int).

    Returns:
        (num_frames,) knee angles in degrees.
    """
    num_frames = poses.shape[0]
    angles = np.zeros(num_frames, dtype=np.float32)
    rad2deg = 180.0 / np.pi

    for i in range(num_frames):
        pose = poses[i]
        hip = pose[hip_idx]
        knee = pose[knee_idx]
        foot = pose[foot_idx]

        angle_rad = _angle_3pt_rad(hip, knee, foot)
        angles[i] = angle_rad * rad2deg

    return angles


@njit(cache=True, fastmath=True)
def _compute_trunk_lean_series_numba(poses: np.ndarray) -> np.ndarray:
    """Compute trunk lean angle series (jitted).

    Args:
        poses: (num_frames, 17, 2) pose array.

    Returns:
        (num_frames,) trunk lean angles in degrees.
    """
    num_frames = poses.shape[0]
    leans = np.zeros(num_frames, dtype=np.float32)
    rad2deg = 180.0 / np.pi

    # H36Key indices (hardcoded for Numba compatibility)
    # LHIP=4, RHIP=8, LSHOULDER=11, RSHOULDER=14
    l_hip = 4
    r_hip = 8
    l_shoulder = 11
    r_shoulder = 14

    for i in range(num_frames):
        pose = poses[i]
        # Mid-hip to mid-shoulder vector
        mid_hip = (pose[l_hip] + pose[r_hip]) * 0.5
        mid_shoulder = (pose[l_shoulder] + pose[r_shoulder]) * 0.5

        spine_vector = mid_shoulder - mid_hip

        # Angle from vertical (0, -1) - upward in normalized coords
        # atan2(x, -y) gives angle from vertical
        lean = np.arctan2(spine_vector[0], -spine_vector[1])
        leans[i] = lean * rad2deg

    return leans


class BiomechanicsAnalyzer:
    """Compute biomechanics metrics for skating technique analysis."""

    def __init__(self, element_def: "ElementDef") -> None:
        """Initialize analyzer with element definition.

        Args:
            element_def: ElementDef with ideal metric ranges.
        """
        self._element_def = element_def

    def analyze(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
        com_trajectory: NDArray[np.float32] | None = None,
    ) -> list[MetricResult]:
        """Compute all relevant metrics for the element.

        Args:
            poses: Normalized pose sequence (num_frames, 17, 2).
            phases: Element phase boundaries.
            fps: Video frame rate.
            com_trajectory: Pre-computed CoM trajectory (optional, for caching).

        Returns:
            List of MetricResult with computed values and goodness assessment.
        """
        results: list[MetricResult] = []

        # Compute metrics based on element type
        if self._element_def.rotations > 0:
            # Jump metrics
            results.extend(self._analyze_jump(poses, phases, fps, com_trajectory=com_trajectory))
        else:
            # Step/edge metrics
            results.extend(self._analyze_step(poses, phases, fps))

        # Common metrics for all elements
        results.extend(self._analyze_common(poses, phases, fps))

        # Mark goodness based on ideal ranges
        for result in results:
            if result.name in self._element_def.ideal_metrics:
                min_good, max_good = self._element_def.ideal_metrics[result.name]
                result.is_good = min_good <= result.value <= max_good
                # Store reference range
                object.__setattr__(result, "reference_range", (min_good, max_good))

        return results

    def _analyze_jump(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
        com_trajectory: NDArray[np.float32] | None = None,
    ) -> list[MetricResult]:
        """Analyze jump-specific metrics."""
        results: list[MetricResult] = []

        # Airtime
        airtime = self.compute_airtime(phases, fps)
        results.append(
            MetricResult(
                name="airtime",
                value=airtime,
                unit="s",
                is_good=False,  # Will be updated based on ideal range
                reference_range=(0, 0),
            )
        )

        # Jump height (CoM-based for physics accuracy)
        height = self.compute_jump_height_com(poses, phases, com_trajectory=com_trajectory)
        results.append(
            MetricResult(
                name="max_height",
                value=height,
                unit="norm",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        # Landing quality
        landing_quality = self.compute_landing_quality(poses, phases)
        results.append(
            MetricResult(
                name="landing_knee_angle",
                value=landing_quality,
                unit="deg",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        # Arm position
        arm_score = self.compute_arm_position(poses)
        results.append(
            MetricResult(
                name="arm_position_score",
                value=arm_score,
                unit="score",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        # Rotation speed
        rot_speed = self.compute_rotation_speed(poses, phases, fps)
        results.append(
            MetricResult(
                name="rotation_speed",
                value=rot_speed,
                unit="deg/s",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        # Landing quality (OOFSkate approach: camera-independent)
        landing_stab = self.compute_landing_knee_stability(poses, phases)
        results.append(
            MetricResult(
                name="landing_knee_stability",
                value=landing_stab,
                unit="score",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        landing_trunk = self.compute_landing_trunk_recovery(poses, phases)
        results.append(
            MetricResult(
                name="landing_trunk_recovery",
                value=landing_trunk,
                unit="score",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        rel_height = self.compute_relative_jump_height(poses, phases, com_trajectory=com_trajectory)
        results.append(
            MetricResult(
                name="relative_jump_height",
                value=rel_height,
                unit="ratio",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        # Landing CoM velocity (negative = hard landing)
        landing_vel = self.compute_landing_com_velocity(poses, phases, fps)
        results.append(
            MetricResult(
                name="landing_com_velocity",
                value=landing_vel,
                unit="norm/s",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        # Landing smoothness (post-landing CoM stability)
        landing_smooth = self.compute_landing_smoothness(poses, phases, fps)
        results.append(
            MetricResult(
                name="landing_smoothness",
                value=landing_smooth,
                unit="score",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        return results

    def _analyze_step(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
    ) -> list[MetricResult]:
        """Analyze step/edge-specific metrics.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries (used for duration calc).
            fps: Frame rate (used for duration calc).
        """
        results: list[MetricResult] = []

        # Knee angle (average during element)
        knee_angles = self.compute_knee_angle_series(poses, side="left")
        avg_knee_angle = float(np.mean(knee_angles))
        results.append(
            MetricResult(
                name="knee_angle",
                value=avg_knee_angle,
                unit="deg",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        # Trunk lean
        trunk_lean = self.compute_trunk_lean(poses)
        # Use average lean across element
        avg_lean = float(np.mean(trunk_lean))
        results.append(
            MetricResult(
                name="trunk_lean",
                value=avg_lean,
                unit="deg",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        # Edge indicator
        edge_ind = self.compute_edge_indicator(poses, side="left")
        # Measure edge change (variance)
        edge_change = float(np.std(edge_ind))
        results.append(
            MetricResult(
                name="edge_change_smoothness",
                value=edge_change,
                unit="score",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        return results

    def _analyze_common(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
    ) -> list[MetricResult]:
        """Analyze metrics common to all elements.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            fps: Frame rate (reserved for future use).
        """
        results: list[MetricResult] = []

        # Symmetry
        symmetry = self.compute_symmetry(poses, phases)
        results.append(
            MetricResult(
                name="symmetry",
                value=symmetry,
                unit="score",
                is_good=False,
                reference_range=(0, 0),
            )
        )

        return results

    def compute_angle_series(
        self,
        poses: NormalizedPose,
        joint_a: int,
        joint_b: int,
        joint_c: int,
    ) -> TimeSeries:
        """Compute angle ABC for each frame (vectorized).

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            joint_a: Index of first joint.
            joint_b: Index of vertex joint.
            joint_c: Index of third joint.

        Returns:
            TimeSeries of angles in degrees.
        """
        a = poses[:, joint_a]  # (N, 2)
        b = poses[:, joint_b]  # (N, 2)
        c = poses[:, joint_c]  # (N, 2)

        # Build triplet array for batch processing: (N, 3, 2)
        abc_triplets = np.stack([a, b, c], axis=1)
        return angle_3pt_batch(abc_triplets).astype(np.float32)

    def compute_angular_velocity(self, angle_series: TimeSeries, fps: float) -> TimeSeries:
        """Compute angular velocity from angle series.

        Args:
            angle_series: Angles in degrees (num_frames,).
            fps: Frame rate.

        Returns:
            Angular velocity in deg/s.
        """
        # Compute gradient
        gradient = np.gradient(angle_series)
        return gradient * fps

    def compute_airtime(self, phases: ElementPhase, fps: float) -> float:
        """Compute flight time.

        Args:
            phases: Element phase boundaries.
            fps: Frame rate.

        Returns:
            Airtime in seconds.
        """
        return phases.airtime_sec(fps)

    def compute_jump_height(self, hip_y_series: TimeSeries, phases: ElementPhase) -> float:
        """Compute maximum jump height using hip trajectory.

        .. deprecated::
            This method has ~60% error for low jumps due to landing knee flexion.
            Use compute_jump_height_com() for physics-accurate results using
            Center of Mass trajectory.

        Args:
            hip_y_series: Hip Y coordinates (lower = higher).
            phases: Element phase boundaries.

        Returns:
            Maximum height in normalized units.

        Warning:
            Deprecated - use compute_jump_height_com() for accurate results.
            The hip-only method overestimates low jumps by up to 60% because
            skaters land with bent knees, which affects hip position but not CoM.
        """
        warnings.warn(
            "compute_jump_height() is deprecated due to 60% error for low jumps. "
            "Use compute_jump_height_com() for physics-accurate results using "
            "Center of Mass trajectory instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Get landing hip Y (reference level)
        landing_y = hip_y_series[phases.landing]

        # Get minimum hip Y (peak height)
        peak_y = np.min(hip_y_series[phases.takeoff : phases.landing])

        return float(landing_y - peak_y)

    def compute_jump_height_com(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        com_trajectory: NDArray[np.float32] | None = None,
    ) -> float:
        """Compute jump height using Center of Mass trajectory.

        This method provides physics-accurate jump height independent of
        landing pose. During flight, the CoM follows a parabolic trajectory
        governed only by gravity: h(t) = h₀ + v₀t - ½gt²

        The hip-only method has ~60% error for low jumps because skaters
        land with bent knees, which artificially increases the measured
        "flight time" and therefore the computed height.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            com_trajectory: Pre-computed CoM trajectory (optional, for caching).

        Returns:
            Maximum jump height in normalized units (peak - takeoff CoM).

        Reference:
            - Dempster (1955) - Space requirements of the seated operator
            - Zatsiorsky (2002) - Kinetics of human motion
            - Gemini Research (2026) - 60% error in hip-only method
        """
        if com_trajectory is None:
            com_trajectory = calculate_com_trajectory(poses)

        # Get CoM at takeoff (baseline)
        takeoff_com = com_trajectory[phases.takeoff]

        # Find minimum CoM during flight (maximum height)
        # Y is inverted in normalized coords, so min Y = max height
        flight_com = com_trajectory[phases.takeoff : phases.landing + 1]
        peak_com = np.min(flight_com)

        # Height = takeoff CoM - peak CoM (both inverted, so difference is positive)
        return float(takeoff_com - peak_com)

    def compute_landing_quality(self, poses: NormalizedPose, phases: ElementPhase) -> float:
        """Compute landing knee angle.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.

        Returns:
            Knee angle in degrees at landing frame.
        """
        # Use left knee angle at landing frame
        landing_frame = min(phases.landing, len(poses) - 1)

        hip = poses[landing_frame, H36Key.LHIP]
        knee = poses[landing_frame, H36Key.LKNEE]
        foot = poses[landing_frame, H36Key.LFOOT]

        return angle_3pt(hip, knee, foot)

    def compute_landing_knee_stability(self, poses: NormalizedPose, phases: ElementPhase) -> float:
        """Compute post-landing knee stability score.

        Measures how stable the knees are after landing by analyzing the
        standard deviation of knee angles during the post-landing phase.
        Camera-independent: uses internal body angles only.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.

        Returns:
            Stability score in [0.0, 1.0] where 1.0 = perfectly stable.
            Formula: max(0.0, 1.0 - avg_std / 15.0)
            Returns 1.0 if no post-landing data available.
        """
        # Check if we have post-landing data
        if phases.end <= phases.landing + 1:
            return 1.0

        # Extract post-landing frames (landing+1 to end)
        post_landing_start = phases.landing + 1
        post_landing_poses = poses[post_landing_start : phases.end + 1]

        # Compute knee angle series for left and right
        left_knee_angles = self.compute_knee_angle_series(post_landing_poses, side="left")
        right_knee_angles = self.compute_knee_angle_series(post_landing_poses, side="right")

        # Calculate standard deviation of knee angles
        left_std = float(np.std(left_knee_angles))
        right_std = float(np.std(right_knee_angles))

        # Average standard deviation
        avg_std = (left_std + right_std) / 2.0

        # Convert to stability score: lower std = higher stability
        # 15 degrees is a reasonable threshold for "unstable"
        stability = max(0.0, 1.0 - avg_std / 15.0)

        return float(stability)

    def compute_landing_trunk_recovery(self, poses: NormalizedPose, phases: ElementPhase) -> float:
        """Compute post-landing trunk recovery score.

        Measures how upright the trunk is during the post-landing phase.
        Camera-independent: uses spine-to-hip angle relative to vertical.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.

        Returns:
            Recovery score in [0.0, 1.0] where 1.0 = perfectly upright.
            Formula: max(0.0, 1.0 - avg_lean / 30.0)
            Returns 1.0 if no post-landing data available.
        """
        # Check if we have post-landing data
        if phases.end <= phases.landing:
            return 1.0

        # Extract post-landing frames (landing+1 to end)
        post_landing_start = phases.landing + 1
        post_landing_poses = poses[post_landing_start : phases.end + 1]

        # Compute trunk lean for post-landing frames
        trunk_lean = self.compute_trunk_lean(post_landing_poses)

        # Calculate average absolute lean during post-landing
        avg_lean = float(np.mean(np.abs(trunk_lean)))

        # Convert to recovery score: lower lean = higher recovery
        # 30 degrees is a reasonable threshold for "poor recovery"
        recovery = max(0.0, 1.0 - avg_lean / 30.0)

        return float(recovery)

    def compute_landing_com_velocity(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
    ) -> float:
        """Compute CoM vertical velocity at landing frame.

        Negative value indicates downward motion (hard landing).
        Uses backward difference on CoM Y trajectory.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            fps: Frame rate.

        Returns:
            CoM vertical velocity in norm/s. Negative = downward.
            Returns 0.0 if landing frame is invalid or no previous frame.
        """
        if phases.landing <= 0 or phases.landing >= len(poses):
            return 0.0

        com_trajectory = calculate_com_trajectory(poses)
        # In normalized coords Y increases downward.
        # Backward difference: negate so downward = negative velocity.
        velocity = -(com_trajectory[phases.landing] - com_trajectory[phases.landing - 1]) * fps
        return float(velocity)

    def compute_landing_smoothness(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
    ) -> float:
        """Compute post-landing CoM velocity stability score.

        Measures how smooth the landing is by analyzing CoM velocity stability
        over a 0.5-second window after landing. Lower velocity variation = higher score.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            fps: Frame rate.

        Returns:
            Smoothness score in [0.0, 1.0] where 1.0 = perfectly stable.
            Returns 1.0 if no post-landing data available.
        """
        if phases.end <= phases.landing + 1:
            return 1.0

        post_landing_start = phases.landing + 1
        post_landing_end = min(phases.end + 1, len(poses))
        window_frames = int(0.5 * fps)
        post_landing_end = min(post_landing_start + window_frames, post_landing_end)

        if post_landing_end <= post_landing_start:
            return 1.0

        com_trajectory = calculate_com_trajectory(poses)
        post_com = com_trajectory[post_landing_start:post_landing_end]

        # Velocities in same convention as compute_landing_com_velocity
        velocities = -(post_com[1:] - post_com[:-1]) * fps

        if len(velocities) == 0:
            return 1.0

        std_velocity = float(np.std(velocities))
        # 0.2 norm/s std threshold for "unstable"
        smoothness = max(0.0, 1.0 - std_velocity / 0.2)
        return float(smoothness)

    def compute_arm_position(self, poses: NormalizedPose) -> float:
        """Compute arm position score.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).

        Returns:
            Score [0, 1] where 1 = arms close to body (good for jumps).
        """
        # Calculate average wrist-to-shoulder distance
        left_dist = np.linalg.norm(poses[:, H36Key.LWRIST] - poses[:, H36Key.LSHOULDER], axis=1)
        right_dist = np.linalg.norm(poses[:, H36Key.RWRIST] - poses[:, H36Key.RSHOULDER], axis=1)

        avg_dist = float(np.mean(left_dist + right_dist) / 2)

        return float(max(0, 1 - avg_dist))

    def compute_trunk_lean(self, poses: NormalizedPose) -> TimeSeries:
        """Compute trunk lean angle.

        Uses Numba-jitted implementation for performance.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).

        Returns:
            Trunk angle in degrees (positive = forward lean).
        """
        return _compute_trunk_lean_series_numba(poses)

    def compute_knee_angle_series(self, poses: NormalizedPose, side: str = "left") -> TimeSeries:
        """Compute knee angle series for step elements.

        Uses Numba-jitted implementation for performance.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            side: "left" or "right" knee.

        Returns:
            Knee angle in degrees (num_frames,).
        """
        if side == "left":
            hip_idx, knee_idx, foot_idx = int(H36Key.LHIP), int(H36Key.LKNEE), int(H36Key.LFOOT)
        else:
            hip_idx, knee_idx, foot_idx = int(H36Key.RHIP), int(H36Key.RKNEE), int(H36Key.RFOOT)

        return _compute_knee_angle_series_numba(poses, hip_idx, knee_idx, foot_idx)

    def compute_edge_indicator(
        self,
        poses: NormalizedPose,
        side: str = "left",
    ) -> TimeSeries:
        """Compute edge indicator using H3.6M 17-keypoint format.

        Vectorized implementation — processes all frames at once.

        Uses body lean angle and foot velocity to infer blade edge.
        - Inside edge: body leaning into turn (positive)
        - Outside edge: body leaning away from turn (negative)
        - Flat edge: body upright (near zero)

        Note: This is a simplified inference since H3.6M lacks detailed foot keypoints.
        For accurate blade detection, use BladeEdgeDetector3D with full 3D poses.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            side: "left" or "right" foot.

        Returns:
            Edge indicator: +1 = inside edge, -1 = outside edge, 0 = flat.
        """
        if side == "left":
            hip = poses[:, H36Key.LHIP]
            shoulder = poses[:, H36Key.LSHOULDER]
        else:
            hip = poses[:, H36Key.RHIP]
            shoulder = poses[:, H36Key.RSHOULDER]

        # Vector from hip to shoulder: (N, 2)
        spine_vector = shoulder - hip

        # Angle from vertical: atan2(x, -y)
        angle = np.arctan2(spine_vector[:, 0], -spine_vector[:, 1])

        # Normalize to [-1, 1]
        edge_indicator = np.clip(angle / (np.pi / 6), -1, 1).astype(np.float32)

        return edge_indicator

    def compute_rotation_speed(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        fps: float,
    ) -> float:
        """Compute peak rotation speed during jump.

        Vectorized implementation — processes all frames at once.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            fps: Frame rate.

        Returns:
            Peak rotation speed in deg/s.
        """
        # Vectorized shoulder axis angle: (N,)
        left_shoulder = poses[:, H36Key.LSHOULDER]
        right_shoulder = poses[:, H36Key.RSHOULDER]

        shoulder_vector = right_shoulder - left_shoulder
        angles = np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])
        angles_deg = np.degrees(angles)

        # Angular velocity
        velocity = self.compute_angular_velocity(angles_deg, fps)

        # Peak in flight phase
        if phases.takeoff < phases.landing and phases.landing < len(velocity):
            flight_velocity = velocity[phases.takeoff : phases.landing]
            return float(np.max(np.abs(flight_velocity)))

        return 0.0

    def compute_symmetry(self, poses: NormalizedPose, phases: ElementPhase) -> float:
        """Compute body symmetry score.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.

        Returns:
            Symmetry score [0, 1] where 1 = perfect symmetry.
        """
        # Get poses during element
        start = phases.start
        end = min(phases.end, len(poses))
        element_poses = poses[start:end]

        # Calculate left-right asymmetry for key joints
        joint_pairs = [
            (H36Key.LSHOULDER, H36Key.RSHOULDER),
            (H36Key.LELBOW, H36Key.RELBOW),
            (H36Key.LHIP, H36Key.RHIP),
            (H36Key.LKNEE, H36Key.RKNEE),
        ]

        asymmetries: list[float] = []

        for left_idx, right_idx in joint_pairs:
            # Mirror left side and compare to right
            left_joints = element_poses[:, left_idx]
            right_joints = element_poses[:, right_idx]

            # Mirror left across Y-axis: (x, y) -> (-x, y)
            mirrored_left = left_joints.copy()
            mirrored_left[:, 0] = -left_joints[:, 0]

            # Calculate average distance
            distances = np.linalg.norm(mirrored_left - right_joints, axis=1)
            asymmetries.append(float(np.mean(distances)))

        # Symmetry = 1 - average asymmetry
        avg_asymmetry = float(np.mean(asymmetries))
        return float(max(0, 1 - avg_asymmetry))

    def compute_relative_jump_height(
        self,
        poses: NormalizedPose,
        phases: ElementPhase,
        com_trajectory: NDArray[np.float32] | None = None,
    ) -> float:
        """Compute jump height normalized by spine length (camera-independent).

        This metric provides a camera-independent measure of jump height by
        normalizing the Center of Mass displacement by the athlete's spine length.
        This removes dependence on camera distance and zoom level.

        Typical values:
        - 0.0: No jump
        - ~0.5: Typical jump
        - ~1.0+: Elite jump (CoM displacement equal to spine length)

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            phases: Element phase boundaries.
            com_trajectory: Pre-computed CoM trajectory (optional, for caching).

        Returns:
            Relative jump height as ratio (CoM displacement / spine length).
            Returns 0.0 if takeoff >= landing (no jump detected).
        """
        # Guard against invalid phases
        if phases.takeoff >= phases.landing:
            return 0.0

        # Calculate average spine length around takeoff (vectorized)
        # Spine = distance from mid-hip to mid-shoulder
        start_frame = max(0, phases.takeoff - 2)
        end_frame = min(len(poses), phases.takeoff + 3)

        takeoff_window = poses[start_frame:end_frame]
        mid_hip = (takeoff_window[:, H36Key.LHIP] + takeoff_window[:, H36Key.RHIP]) / 2
        mid_shoulder = (
            takeoff_window[:, H36Key.LSHOULDER] + takeoff_window[:, H36Key.RSHOULDER]
        ) / 2
        spine_lengths = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
        valid_spines = spine_lengths[spine_lengths >= 0.01]

        if len(valid_spines) == 0:
            return 0.0

        avg_spine = float(np.mean(valid_spines))

        if com_trajectory is None:
            com_trajectory = calculate_com_trajectory(poses)

        # Get CoM at takeoff
        takeoff_com = com_trajectory[phases.takeoff]

        # Find minimum CoM during flight (maximum height)
        # Y is inverted in normalized coords, so min Y = max height
        flight_com = com_trajectory[phases.takeoff : phases.landing + 1]
        peak_com = np.min(flight_com)

        # CoM displacement = takeoff - peak (both inverted, so difference is positive)
        com_displacement = float(takeoff_com - peak_com)

        # Return normalized height
        return com_displacement / avg_spine


@dataclass
class PhaseDetectionResult:
    """Result of automatic phase detection."""

    phases: ElementPhase
    """Detected phase boundaries."""

    confidence: float
    """Detection confidence score [0, 1]."""
