"""Physics calculations for figure skating biomechanics analysis.

Implements:
- Center of Mass (CoM) calculation using anthropometric data
- Moment of Inertia (I) calculation
- Angular Momentum (L = I * w)
- Parabolic trajectory fitting for jump height

References:
- Dempster (1955) anthropometric tables
- Zatsiorsky (2002) biomechanics
- AthletePose3D: Monocular 3D pose for sports
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

# Anthropometric data (segment mass ratios)
# Based on Dempster (1955) - normalized to total body mass
SEGMENT_MASS_RATIOS = {
    "head": 0.081,
    "torso": 0.497,  # Includes thorax, abdomen, pelvis
    "left_upper_arm": 0.028,
    "left_forearm": 0.016,
    "left_hand": 0.006,
    "right_upper_arm": 0.028,
    "right_forearm": 0.016,
    "right_hand": 0.006,
    "left_thigh": 0.100,
    "left_shin": 0.0465,  # Lower leg
    "left_foot": 0.0145,
    "right_thigh": 0.100,
    "right_shin": 0.0465,
    "right_foot": 0.0145,
}

# Default body mass (kg) - can be overridden
DEFAULT_BODY_MASS = 60.0  # Typical figure skater


@dataclass
class PhysicsResult:
    """Result of physics calculations."""

    center_of_mass: np.ndarray  # (N, 3) CoM trajectory
    moment_of_inertia: np.ndarray  # (N,) I values
    angular_momentum: np.ndarray  # (N,) L values
    jump_height: float | None = None  # meters
    flight_time: float | None = None  # seconds
    rotation_rate: float | None = None  # deg/sec


class PhysicsEngine:
    """Physics calculations from 3D pose data.

    Uses anthropometric data to calculate:
    - Center of Mass (CoM) trajectory
    - Moment of Inertia (I)
    - Angular Momentum (L)
    - Jump height from parabolic fit
    """

    def __init__(self, body_mass: float = DEFAULT_BODY_MASS):
        """Initialize physics engine.

        Args:
            body_mass: Total body mass in kg
        """
        self.body_mass = body_mass
        self.segment_masses = {
            name: ratio * body_mass for name, ratio in SEGMENT_MASS_RATIOS.items()
        }

    def calculate_center_of_mass(
        self,
        poses_3d: np.ndarray,
    ) -> np.ndarray:
        """Calculate Center of Mass trajectory from 3D poses.

        CoM = (1/M) * sum(m_i * p_i)

        Where:
            M = total body mass
            mᵢ = segment mass
            pᵢ = segment center position

        Args:
            poses_3d: (N, 17, 3) array of H3.6M format poses
                - N = number of frames
                - 17 = H3.6M keypoints
                - 3 = (x, y, z) coordinates in meters

        Returns:
            com_trajectory: (N, 3) array of CoM positions
        """
        from ..pose_estimation import H36Key

        n_frames = poses_3d.shape[0]
        com_trajectory = np.zeros((n_frames, 3))

        for frame_idx in range(n_frames):
            pose = poses_3d[frame_idx]

            # Segment center positions (simplified from keypoints)
            # Head: nose/head approximation
            head_pos = pose[H36Key.HEAD]
            com_trajectory[frame_idx] += self.segment_masses["head"] * head_pos

            # Torso: weighted average of spine, thorax
            spine_pos = pose[H36Key.SPINE]
            thorax_pos = pose[H36Key.THORAX]
            torso_pos = (spine_pos + thorax_pos) / 2
            com_trajectory[frame_idx] += self.segment_masses["torso"] * torso_pos

            # Process arm segments
            for side, prefix in [("left", "l"), ("right", "r")]:
                shoulder_idx = getattr(H36Key, f"{prefix.upper()}SHOULDER")
                elbow_idx = getattr(H36Key, f"{prefix.upper()}ELBOW")
                wrist_idx = getattr(H36Key, f"{prefix.upper()}WRIST")

                # Upper arm: shoulder to elbow midpoint
                upper_arm_pos = (pose[shoulder_idx] + pose[elbow_idx]) / 2
                com_trajectory[frame_idx] += (
                    self.segment_masses[f"{side}_upper_arm"] * upper_arm_pos
                )

                # Forearm: elbow to wrist midpoint
                forearm_pos = (pose[elbow_idx] + pose[wrist_idx]) / 2
                com_trajectory[frame_idx] += self.segment_masses[f"{side}_forearm"] * forearm_pos

                # Hand: wrist position
                com_trajectory[frame_idx] += self.segment_masses[f"{side}_hand"] * pose[wrist_idx]

            # Process leg segments
            for side, prefix in [("left", "l"), ("right", "r")]:
                hip_idx = getattr(H36Key, f"{prefix.upper()}HIP")
                knee_idx = getattr(H36Key, f"{prefix.upper()}KNEE")
                ankle_idx = getattr(H36Key, f"{prefix.upper()}FOOT")

                # Thigh: hip to knee midpoint
                thigh_pos = (pose[hip_idx] + pose[knee_idx]) / 2
                com_trajectory[frame_idx] += self.segment_masses[f"{side}_thigh"] * thigh_pos

                # Shin: knee to ankle midpoint
                shin_pos = (pose[knee_idx] + pose[ankle_idx]) / 2
                com_trajectory[frame_idx] += self.segment_masses[f"{side}_shin"] * shin_pos

                # Foot: ankle position
                com_trajectory[frame_idx] += self.segment_masses[f"{side}_foot"] * pose[ankle_idx]

        # Normalize by total mass
        com_trajectory /= self.body_mass

        return com_trajectory

    def calculate_moment_of_inertia(
        self,
        poses_3d: np.ndarray,
        axis: str = "vertical",
    ) -> np.ndarray:
        """Calculate Moment of Inertia about vertical axis.

        I = sum(m_i * r_i^2)

        Where:
            mᵢ = segment mass
            rᵢ = perpendicular distance from rotation axis

        Args:
            poses_3d: (N, 17, 3) array of poses
            axis: Rotation axis ("vertical", "sagittal", "frontal")

        Returns:
            inertia: (N,) array of moment of inertia values (kg·m²)
        """
        from ..pose_estimation import H36Key

        n_frames = poses_3d.shape[0]
        inertia = np.zeros(n_frames)

        # Calculate CoM for each frame (reference point)
        com_trajectory = self.calculate_center_of_mass(poses_3d)

        for frame_idx in range(n_frames):
            pose = poses_3d[frame_idx]
            com = com_trajectory[frame_idx]

            # Calculate distance from each segment to CoM
            # Simplified: use keypoint positions as segment centers

            # Head
            r = np.linalg.norm(pose[H36Key.HEAD] - com)
            inertia[frame_idx] += self.segment_masses["head"] * r**2

            # Torso
            spine_pos = (pose[H36Key.SPINE] + pose[H36Key.THORAX]) / 2
            r = np.linalg.norm(spine_pos - com)
            inertia[frame_idx] += self.segment_masses["torso"] * r**2

            # Process all limb segments
            for side, prefix in [("left", "l"), ("right", "r")]:
                shoulder_idx = getattr(H36Key, f"{prefix.upper()}SHOULDER")
                elbow_idx = getattr(H36Key, f"{prefix.upper()}ELBOW")
                wrist_idx = getattr(H36Key, f"{prefix.upper()}WRIST")
                hip_idx = getattr(H36Key, f"{prefix.upper()}HIP")
                knee_idx = getattr(H36Key, f"{prefix.upper()}KNEE")
                ankle_idx = getattr(H36Key, f"{prefix.upper()}FOOT")

                # Upper arm
                upper_arm_pos = (pose[shoulder_idx] + pose[elbow_idx]) / 2
                r = np.linalg.norm(upper_arm_pos - com)
                inertia[frame_idx] += self.segment_masses[f"{side}_upper_arm"] * r**2

                # Forearm
                forearm_pos = (pose[elbow_idx] + pose[wrist_idx]) / 2
                r = np.linalg.norm(forearm_pos - com)
                inertia[frame_idx] += self.segment_masses[f"{side}_forearm"] * r**2

                # Thigh
                thigh_pos = (pose[hip_idx] + pose[knee_idx]) / 2
                r = np.linalg.norm(thigh_pos - com)
                inertia[frame_idx] += self.segment_masses[f"{side}_thigh"] * r**2

                # Shin
                shin_pos = (pose[knee_idx] + pose[ankle_idx]) / 2
                r = np.linalg.norm(shin_pos - com)
                inertia[frame_idx] += self.segment_masses[f"{side}_shin"] * r**2

        return inertia

    def calculate_angular_momentum(
        self,
        poses_3d: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """Calculate Angular Momentum.

        L = I * w

        Args:
            poses_3d: (N, 17, 3) array of poses
            angular_velocity: (N,) array of angular velocities (rad/s)

        Returns:
            angular_momentum: (N,) array of L values (kg·m²/s)
        """
        inertia = self.calculate_moment_of_inertia(poses_3d)
        return inertia * angular_velocity

    def fit_jump_trajectory(
        self,
        poses_3d: np.ndarray,
        takeoff_idx: int,
        landing_idx: int,
    ) -> dict:
        """Fit parabolic trajectory to CoM during flight.

        During flight, CoM follows: h(t) = h₀ + v₀t - ½gt²

        Args:
            poses_3d: (N, 17, 3) array of poses
            takeoff_idx: Frame index of takeoff
            landing_idx: Frame index of landing

        Returns:
            dict with:
                - height: Max jump height (meters)
                - flight_time: Time in air (seconds)
                - takeoff_velocity: Vertical velocity at takeoff (m/s)
                - fit_quality: R² of parabolic fit
        """
        # Get CoM trajectory
        com_trajectory = self.calculate_center_of_mass(poses_3d)

        # Extract flight phase (vertical component = Y axis)
        flight_com = com_trajectory[takeoff_idx : landing_idx + 1, 1]  # Y coordinate
        n_frames = len(flight_com)
        t = np.arange(n_frames) / 30.0  # Assume 30 fps

        # Parabolic fit: h(t) = at² + bt + c
        def parabola(t, a, b, c):
            return a * t**2 + b * t + c

        try:
            params, _ = curve_fit(parabola, t, flight_com)
            a, b, c = params

            # Calculate derived values
            # g = -2a (acceleration due to gravity)
            # v₀ = b (initial velocity)
            # h₀ = c (initial height)

            # Peak height occurs at t* = -b/(2a)
            t_peak = -b / (2 * a)
            h_peak = parabola(t_peak, a, b, c)
            h_takeoff = parabola(0, a, b, c)
            jump_height = h_peak - h_takeoff

            # Flight time
            flight_time = t[-1] - t[0]

            # R² for fit quality
            residuals = flight_com - parabola(t, a, b, c)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((flight_com - np.mean(flight_com)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                "height": abs(jump_height),  # meters
                "flight_time": flight_time,  # seconds
                "takeoff_velocity": b,  # m/s
                "fit_quality": r_squared,
            }

        except Exception:
            # Fallback: simple height difference
            return {
                "height": np.max(flight_com) - np.min(flight_com),
                "flight_time": n_frames / 30.0,
                "takeoff_velocity": 0.0,
                "fit_quality": 0.0,
            }

    def analyze(
        self,
        poses_3d: np.ndarray,
        takeoff_idx: int | None = None,
        landing_idx: int | None = None,
    ) -> PhysicsResult:
        """Run full physics analysis on 3D pose sequence.

        Args:
            poses_3d: (N, 17, 3) array of poses
            takeoff_idx: Optional takeoff frame index
            landing_idx: Optional landing frame index

        Returns:
            PhysicsResult with all calculated values
        """
        # Calculate CoM trajectory
        com = self.calculate_center_of_mass(poses_3d)

        # Calculate moment of inertia
        inertia = self.calculate_moment_of_inertia(poses_3d)

        # Angular momentum (assume zero angular velocity for now)
        angular_momentum = np.zeros_like(inertia)

        # Jump height (if takeoff/landing provided)
        jump_height = None
        flight_time = None

        if takeoff_idx is not None and landing_idx is not None:
            trajectory = self.fit_jump_trajectory(poses_3d, takeoff_idx, landing_idx)
            jump_height = trajectory["height"]
            flight_time = trajectory["flight_time"]

        return PhysicsResult(
            center_of_mass=com,
            moment_of_inertia=inertia,
            angular_momentum=angular_momentum,
            jump_height=jump_height,
            flight_time=flight_time,
            rotation_rate=None,
        )
