"""Export 3D skeleton poses to .glb for interactive Gradio Model3D viewer.

Converts H3.6M 17-keypoint 3D poses into a trimesh scene with:
- Cylinders for bones (colored by body region)
- Spheres for joints (colored by angle quality)
- Ground plane grid for spatial reference
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import trimesh

from src.types import H36M_SKELETON_EDGES
from src.utils.geometry import angle_3pt

# Body region colors (RGB for trimesh) keyed by H36Key index pairs from H36M_SKELETON_EDGES
_BONE_COLORS: dict[tuple[int, int], tuple[int, int, int]] = {
    # Torso/spine — light gray
    (0, 7): (180, 180, 180),  # HIP_CENTER → SPINE
    (7, 8): (180, 180, 180),  # SPINE → THORAX
    (8, 9): (180, 180, 180),  # THORAX → NECK
    (9, 10): (200, 200, 200),  # NECK → HEAD
    # Right leg — blue tint
    (0, 1): (100, 140, 220),  # HIP_CENTER → RHIP
    (1, 2): (100, 140, 220),  # RHIP → RKNEE
    (2, 3): (100, 140, 220),  # RKNEE → RFOOT
    # Left leg — blue tint
    (0, 4): (100, 140, 220),  # HIP_CENTER → LHIP
    (4, 5): (100, 140, 220),  # LHIP → LKNEE
    (5, 6): (100, 140, 220),  # LKNEE → LFOOT
    # Right arm — green tint
    (8, 14): (100, 200, 140),  # THORAX → RSHOULDER
    (14, 15): (100, 200, 140),  # RSHOULDER → RELBOW
    (15, 16): (100, 200, 140),  # RELBOW → RWRIST
    # Left arm — green tint
    (8, 11): (100, 200, 140),  # THORAX → LSHOULDER
    (11, 12): (100, 200, 140),  # LSHOULDER → LELBOW
    (12, 13): (100, 200, 140),  # LELBOW → LWRIST
}

# Joint angle specs for coloring: (point_a, vertex, point_c)
_ANGLE_JOINTS: list[tuple[int, int, int]] = [
    (1, 2, 3),  # R knee (RHIP, RKNEE, RFOOT)
    (4, 5, 6),  # L knee (LHIP, LKNEE, LFOOT)
    (14, 15, 16),  # R elbow (RSHOULDER, RELBOW, RWRIST)
    (11, 12, 13),  # L elbow (LSHOULDER, LELBOW, LWRIST)
    (8, 1, 2),  # R hip (THORAX, RHIP, RKNEE)
    (8, 4, 5),  # L hip (THORAX, LHIP, LKNEE)
]

_ANGLE_GOOD_RANGE = (90, 170)
_ANGLE_WARN_RANGE = (60, 190)


def _angle_color(angle: float) -> tuple[int, int, int]:
    """Return RGB color based on angle quality."""
    lo_g, hi_g = _ANGLE_GOOD_RANGE
    lo_w, hi_w = _ANGLE_WARN_RANGE
    if lo_g <= angle <= hi_g:
        return (50, 220, 50)  # green
    if lo_w <= angle <= hi_w:
        return (220, 220, 50)  # yellow
    return (220, 50, 50)  # red


def poses_to_glb(
    poses_3d: np.ndarray,
    frame_idx: int = 0,
    bone_radius: float = 0.015,
    joint_radius: float = 0.025,
) -> str:
    """Convert a single 3D pose frame to a .glb file.

    Args:
        poses_3d: (N, 17, 3) array of 3D poses in meters (hip-centered).
        frame_idx: Which frame to export.
        bone_radius: Radius of bone cylinders.
        joint_radius: Radius of joint spheres.

    Returns:
        Path to the exported .glb file (in tempdir).
    """
    if frame_idx >= len(poses_3d):
        frame_idx = len(poses_3d) - 1

    pose = poses_3d[frame_idx]  # (17, 3)

    scene = trimesh.Scene()

    # Compute joint angles for coloring
    joint_colors: dict[int, tuple[int, int, int]] = {}
    for pa_idx, v_idx, pc_idx in _ANGLE_JOINTS:
        a = pose[pa_idx]
        v = pose[v_idx]
        c = pose[pc_idx]
        if not (np.isnan(a).any() or np.isnan(v).any() or np.isnan(c).any()):
            angle = angle_3pt(a, v, c)
            if not np.isnan(angle):
                joint_colors[v_idx] = _angle_color(angle)

    # Draw bones as cylinders
    for joint_a, joint_b in H36M_SKELETON_EDGES:
        p_a = pose[joint_a]
        p_b = pose[joint_b]

        if np.isnan(p_a).any() or np.isnan(p_b).any():
            continue

        bone_vec = p_b - p_a
        bone_len = np.linalg.norm(bone_vec)
        if bone_len < 1e-4:
            continue

        color = _BONE_COLORS.get((joint_a, joint_b), (160, 160, 160))

        # Create cylinder between two points
        bone_mesh = trimesh.creation.cylinder(
            radius=bone_radius,
            height=bone_len,
            sections=8,
        )

        # Position at midpoint
        midpoint = (p_a + p_b) / 2
        bone_mesh.apply_translation(midpoint)

        # Align cylinder (default Y-axis) to bone direction
        bone_dir = bone_vec / bone_len
        # Default cylinder axis is [0, 1, 0]
        y_axis = np.array([0.0, 1.0, 0.0])
        dot = np.dot(y_axis, bone_dir)
        if abs(dot) < 0.9999:
            cross = np.cross(y_axis, bone_dir)
            cross_len = np.linalg.norm(cross)
            if cross_len > 1e-6:
                axis = cross / cross_len
                angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
                rot = trimesh.transformations.rotation_matrix(np.radians(angle), axis)
                bone_mesh.apply_transform(rot)

        bone_mesh.visual.face_colors = color
        scene.add_geometry(bone_mesh)

    # Draw joints as spheres
    for i in range(17):
        p = pose[i]
        if np.isnan(p).any():
            continue

        color = joint_colors.get(i, (200, 200, 200))
        joint_mesh = trimesh.creation.icosphere(radius=joint_radius, subdivisions=1)
        joint_mesh.apply_translation(p)
        joint_mesh.visual.face_colors = color
        scene.add_geometry(joint_mesh)

    # Add ground plane grid
    _add_ground_grid(scene, pose)

    # Export to .glb
    tmpdir = tempfile.mkdtemp()
    glb_path = Path(tmpdir) / "skeleton.glb"
    scene.export(str(glb_path), file_type="glb")
    return str(glb_path)


def poses_to_glb_sequence(
    poses_3d: np.ndarray,
    output_path: str | Path,
    bone_radius: float = 0.015,
    joint_radius: float = 0.025,
) -> str:
    """Export all frames as individual .glb files (for frame scrubbing).

    Args:
        poses_3d: (N, 17, 3) array of 3D poses.
        output_path: Directory to save .glb files.
        bone_radius: Radius of bone cylinders.
        joint_radius: Radius of joint spheres.

    Returns:
        Path to the output directory.
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute all frames
    paths: list[str] = []
    for i in range(len(poses_3d)):
        glb = poses_to_glb(poses_3d, i, bone_radius, joint_radius)
        dst = out_dir / f"frame_{i:04d}.glb"
        Path(glb).rename(dst)
        paths.append(str(dst))

    return str(out_dir)


def _add_ground_grid(scene: trimesh.Scene, pose: np.ndarray) -> None:
    """Add a subtle ground plane grid below the skeleton."""
    # Find the lowest point (feet)
    valid_y = pose[~np.isnan(pose[:, 1]), 1]
    if len(valid_y) == 0:
        return
    ground_y = valid_y.min()

    # Create a simple grid
    grid_size = 2.0
    grid_lines = 8
    step = grid_size / grid_lines
    half = grid_size / 2

    for i in range(grid_lines + 1):
        pos = -half + i * step
        # Lines along X
        points_x = np.array(
            [
                [pos, ground_y, -half],
                [pos, ground_y, half],
            ]
        )
        line_x = trimesh.load_path(points_x)
        line_x.visual.face_colors = (100, 100, 100, 80)
        scene.add_geometry(line_x)

        # Lines along Z
        points_z = np.array(
            [
                [-half, ground_y, pos],
                [half, ground_y, pos],
            ]
        )
        line_z = trimesh.load_path(points_z)
        line_z.visual.face_colors = (100, 100, 100, 80)
        scene.add_geometry(line_z)
