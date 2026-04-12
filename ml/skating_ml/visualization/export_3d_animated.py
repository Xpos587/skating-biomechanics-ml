"""Export 3D skeleton pose sequences to animated .glb files.

Converts H3.6M 17-keypoint 3D pose sequences into animated GLB files using pygltflib.
The GLB contains:
- Cylindrical bones between connected joints (using H36M_SKELETON_EDGES)
- Spherical joint markers at each of the 17 joints
- Animation: translation, rotation, scale keyframes for each node, LINEAR interpolation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pygltflib
from scipy.spatial.transform import Rotation as R

from skating_ml.types import H36M_SKELETON_EDGES

# Body region colors (RGB normalized [0,1])
_BONE_COLORS: dict[tuple[int, int], tuple[float, float, float]] = {
    # Torso/spine — light gray
    (0, 7): (0.71, 0.71, 0.71),  # HIP_CENTER → SPINE
    (7, 8): (0.71, 0.71, 0.71),  # SPINE → THORAX
    (8, 9): (0.71, 0.71, 0.71),  # THORAX → NECK
    (9, 10): (0.78, 0.78, 0.78),  # NECK → HEAD
    # Right leg — blue tint
    (0, 1): (0.39, 0.55, 0.86),  # HIP_CENTER → RHIP
    (1, 2): (0.39, 0.55, 0.86),  # RHIP → RKNEE
    (2, 3): (0.39, 0.55, 0.86),  # RKNEE → RFOOT
    # Left leg — blue tint
    (0, 4): (0.39, 0.55, 0.86),  # HIP_CENTER → LHIP
    (4, 5): (0.39, 0.55, 0.86),  # LHIP → LKNEE
    (5, 6): (0.39, 0.55, 0.86),  # LKNEE → LFOOT
    # Right arm — green tint
    (8, 14): (0.39, 0.78, 0.55),  # THORAX → RSHOULDER
    (14, 15): (0.39, 0.78, 0.55),  # RSHOULDER → RELBOW
    (15, 16): (0.39, 0.78, 0.55),  # RELBOW → RWRIST
    # Left arm — green tint
    (8, 11): (0.39, 0.78, 0.55),  # THORAX → LSHOULDER
    (11, 12): (0.39, 0.78, 0.55),  # LSHOULDER → LELBOW
    (12, 13): (0.39, 0.78, 0.55),  # LELBOW → LWRIST
}

_JOINT_COLOR = (0.78, 0.78, 0.78)  # Light gray


def _cylinder_geometry(
    radius: float = 1.0, height: float = 1.0, sections: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and indices for a cylinder centered at origin.

    Args:
        radius: Cylinder radius.
        height: Total height (cylinder extends from -height/2 to +height/2 along Y).
        sections: Number of radial segments.

    Returns:
        (vertices, indices) where vertices is (N, 3) float and indices is (M,) uint32.
    """
    # Top and bottom circles
    n_verts = sections * 2
    vertices = np.zeros((n_verts, 3), dtype=np.float32)

    for i in range(sections):
        theta = 2 * np.pi * i / sections
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)

        # Top vertex (y = +height/2)
        vertices[i * 2] = [x, height / 2, z]
        # Bottom vertex (y = -height/2)
        vertices[i * 2 + 1] = [x, -height / 2, z]

    # Generate triangle indices
    n_tri = sections * 2  # 2 triangles per section (side quads split)
    indices = np.zeros(n_tri * 3, dtype=np.uint32)

    for i in range(sections):
        i_next = (i + 1) % sections

        # Triangle 1: top_i -> bottom_i -> top_next
        idx = i * 6
        indices[idx + 0] = i * 2
        indices[idx + 1] = i * 2 + 1
        indices[idx + 2] = i_next * 2

        # Triangle 2: bottom_i -> bottom_next -> top_next
        indices[idx + 3] = i * 2 + 1
        indices[idx + 4] = i_next * 2 + 1
        indices[idx + 5] = i_next * 2

    return vertices, indices


def _icosphere_geometry(
    radius: float = 1.0, subdivisions: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and indices for an icosphere.

    Args:
        radius: Sphere radius.
        subdivisions: Number of subdivision iterations (0 = icosahedron, 1 = detailed).

    Returns:
        (vertices, indices) where vertices is (N, 3) float and indices is (M,) uint32.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Icosahedron vertices (12)
    verts = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    vertices = np.array(verts, dtype=np.float32)

    # Normalize to radius
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius

    # Icosahedron faces (20 triangles, 60 indices)
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    indices = np.array(faces, dtype=np.uint32).flatten()

    # Subdivide if requested
    for _ in range(subdivisions):
        vertices, indices = _subdivide_mesh(vertices, indices)
        # Re-normalize to sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius

    return vertices, indices


def _subdivide_mesh(vertices: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide mesh by splitting each triangle into 4 smaller triangles.

    Args:
        vertices: (N, 3) vertex positions.
        indices: (M,) triangle indices (flat array).

    Returns:
        (new_vertices, new_indices) for subdivided mesh.
    """
    # Reshape indices to (M/3, 3)
    faces = indices.reshape(-1, 3)

    # Dictionary to store edge midpoints
    midpoints: dict[tuple[int, int], int] = {}
    new_vertices = [vertices]  # Use list to collect new vertices

    def get_midpoint(i1: int, i2: int) -> int:
        """Get or create midpoint vertex between two vertices."""
        edge = tuple(sorted((i1, i2)))
        if edge in midpoints:
            return midpoints[edge]

        # Create new midpoint
        v1 = vertices[i1]
        v2 = vertices[i2]
        midpoint = (v1 + v2) / 2

        # Assign new index
        idx = len(vertices) + len(new_vertices) - 1
        new_vertices.append(midpoint)
        midpoints[edge] = idx
        return idx

    new_faces = []
    for f in faces:
        v0, v1, v2 = f
        a = get_midpoint(v0, v1)
        b = get_midpoint(v1, v2)
        c = get_midpoint(v2, v0)

        # 4 new triangles
        new_faces.extend(
            [
                [v0, a, c],
                [v1, b, a],
                [v2, c, b],
                [a, b, c],
            ]
        )

    # Combine original and new vertices
    all_vertices = np.vstack(new_vertices)
    return all_vertices, np.array(new_faces, dtype=np.uint32).flatten()


def _compute_trs(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    bone_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute translation, rotation (quaternion), and scale for a bone cylinder.

    Args:
        start_pos: (3,) start joint position.
        end_pos: (3,) end joint position.
        bone_radius: Radius of the bone cylinder.

    Returns:
        (translation, quaternion, scale) where:
        - translation is (3,) midpoint position
        - quaternion is (4,) [w, x, y, z] rotation from Y-axis to bone direction
        - scale is (3,) [radius, half_length, radius]
    """
    translation = (start_pos + end_pos) / 2

    # Bone vector and length
    bone_vec = end_pos - start_pos
    bone_length = np.linalg.norm(bone_vec)

    if bone_length < 1e-6:
        # Degenerate bone: no rotation, unit scale
        return (
            translation,
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([bone_radius, 0.5, bone_radius]),
        )

    bone_dir = bone_vec / bone_length

    # Rotation: from Y-axis [0, 1, 0] to bone_dir
    y_axis = np.array([0.0, 1.0, 0.0])

    # Check if already aligned
    if np.allclose(bone_dir, y_axis):
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    elif np.allclose(bone_dir, -y_axis):
        # Opposite direction: 180° rotation around X
        quaternion = np.array([0.0, 1.0, 0.0, 0.0])  # [w, x, y, z]
    else:
        # Compute rotation using scipy
        rotation = R.align_vectors([bone_dir], [y_axis])[0]
        quaternion = rotation.as_quat()  # [x, y, z, w]
        quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

    scale = np.array([bone_radius, bone_length / 2, bone_radius])

    return translation, quaternion, scale


def poses_to_animated_glb(
    poses_3d: np.ndarray,
    fps: float = 30.0,
    bone_radius: float = 0.012,
    joint_radius: float = 0.018,
) -> str:
    """Convert a 3D pose sequence to an animated GLB file.

    Args:
        poses_3d: (N, 17, 3) array of 3D poses in meters.
        fps: Frames per second for animation timing.
        bone_radius: Radius of bone cylinders.
        joint_radius: Radius of joint spheres.

    Returns:
        Path to the exported .glb file (in tempdir).
    """
    if poses_3d.ndim != 3 or poses_3d.shape[1] != 17 or poses_3d.shape[2] != 3:
        raise ValueError(f"poses_3d must be (N, 17, 3), got {poses_3d.shape}")

    n_frames = poses_3d.shape[0]

    # Create glTF object
    g = pygltflib.GLTF2()

    # Generate base geometry (unit cylinder and unit sphere)
    cylinder_verts, cylinder_idx = _cylinder_geometry(radius=1.0, height=1.0, sections=8)
    sphere_verts, sphere_idx = _icosphere_geometry(radius=1.0, subdivisions=1)

    # Build binary blob
    binary_data = bytearray()

    # Helper to align to 4-byte boundary
    def align(offset: int) -> int:
        return (offset + 3) & ~3

    # === GEOMETRY DATA ===
    # Cylinder vertices
    cylinder_verts_bytes = cylinder_verts.astype(np.float32).tobytes()
    cylinder_verts_offset = len(binary_data)
    binary_data += cylinder_verts_bytes
    cylinder_verts_offset_aligned = align(cylinder_verts_offset)

    # Cylinder indices
    cylinder_idx_bytes = cylinder_idx.astype(np.uint32).tobytes()
    cylinder_idx_offset = align(len(binary_data))
    binary_data += cylinder_idx_bytes

    # Sphere vertices
    sphere_verts_bytes = sphere_verts.astype(np.float32).tobytes()
    sphere_verts_offset = align(len(binary_data))
    binary_data += sphere_verts_bytes

    # Sphere indices
    sphere_idx_bytes = sphere_idx.astype(np.uint32).tobytes()
    sphere_idx_offset = align(len(binary_data))
    binary_data += sphere_idx_bytes

    # === ANIMATION DATA ===
    # Collect all keyframe data
    bone_translations: dict[int, list[np.ndarray]] = {}  # bone_idx -> list of (3,)
    bone_rotations: dict[int, list[np.ndarray]] = {}  # bone_idx -> list of (4,) quat
    bone_scales: dict[int, list[np.ndarray]] = {}  # bone_idx -> list of (3,)
    joint_translations: dict[int, list[np.ndarray]] = {}  # joint_idx -> list of (3,)

    # Initialize lists for all bones and joints
    for i, (_j_a, _j_b) in enumerate(H36M_SKELETON_EDGES):
        bone_translations[i] = []
        bone_rotations[i] = []
        bone_scales[i] = []

    for j in range(17):
        joint_translations[j] = []

    # Compute TRS for each frame
    for frame_idx in range(n_frames):
        pose = poses_3d[frame_idx]

        # Bones
        for i, (j_a, j_b) in enumerate(H36M_SKELETON_EDGES):
            p_a = pose[j_a]
            p_b = pose[j_b]

            if np.isnan(p_a).any() or np.isnan(p_b).any():
                # Use default values for NaN bones
                bone_translations[i].append(np.zeros(3, dtype=np.float32))
                bone_rotations[i].append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                bone_scales[i].append(np.array([bone_radius, 0.5, bone_radius], dtype=np.float32))
            else:
                trans, quat, scale = _compute_trs(p_a, p_b, bone_radius)
                bone_translations[i].append(trans.astype(np.float32))
                bone_rotations[i].append(quat.astype(np.float32))
                bone_scales[i].append(scale.astype(np.float32))

        # Joints
        for j in range(17):
            p = pose[j]
            if np.isnan(p).any():
                joint_translations[j].append(np.zeros(3, dtype=np.float32))
            else:
                joint_translations[j].append(p.astype(np.float32))

    # === WRITE ANIMATION DATA ===
    # Times (input for animation samplers)
    times = (np.arange(n_frames) / fps).astype(np.float32)
    times_bytes = times.tobytes()
    times_offset = align(len(binary_data))
    binary_data += times_bytes

    # Bone translation keyframes
    bone_trans_offsets: dict[int, int] = {}
    for i in range(len(H36M_SKELETON_EDGES)):
        data = np.array(bone_translations[i], dtype=np.float32).flatten()
        bone_trans_offsets[i] = align(len(binary_data))
        binary_data += data.tobytes()

    # Bone rotation keyframes
    bone_rot_offsets: dict[int, int] = {}
    for i in range(len(H36M_SKELETON_EDGES)):
        data = np.array(bone_rotations[i], dtype=np.float32).flatten()
        bone_rot_offsets[i] = align(len(binary_data))
        binary_data += data.tobytes()

    # Bone scale keyframes
    bone_scale_offsets: dict[int, int] = {}
    for i in range(len(H36M_SKELETON_EDGES)):
        data = np.array(bone_scales[i], dtype=np.float32).flatten()
        bone_scale_offsets[i] = align(len(binary_data))
        binary_data += data.tobytes()

    # Joint translation keyframes
    joint_trans_offsets: dict[int, int] = {}
    for j in range(17):
        data = np.array(joint_translations[j], dtype=np.float32).flatten()
        joint_trans_offsets[j] = align(len(binary_data))
        binary_data += data.tobytes()

    # Set binary blob
    g.set_binary_blob(bytes(binary_data))

    # === BUFFER ===
    g.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=cylinder_verts_offset_aligned,
            byteLength=len(cylinder_verts_bytes),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    cylinder_verts_bv = len(g.bufferViews) - 1

    g.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=cylinder_idx_offset,
            byteLength=len(cylinder_idx_bytes),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )
    cylinder_idx_bv = len(g.bufferViews) - 1

    g.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=sphere_verts_offset,
            byteLength=len(sphere_verts_bytes),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    sphere_verts_bv = len(g.bufferViews) - 1

    g.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=sphere_idx_offset,
            byteLength=len(sphere_idx_bytes),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )
    sphere_idx_bv = len(g.bufferViews) - 1

    # Times buffer view
    g.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=times_offset,
            byteLength=len(times_bytes),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    times_bv = len(g.bufferViews) - 1

    # === ACCESSORS ===
    # Cylinder vertices
    g.accessors.append(
        pygltflib.Accessor(
            bufferView=cylinder_verts_bv,
            componentType=pygltflib.FLOAT,
            count=len(cylinder_verts),
            type=pygltflib.VEC3,
        )
    )
    cylinder_verts_acc = len(g.accessors) - 1

    # Cylinder indices
    g.accessors.append(
        pygltflib.Accessor(
            bufferView=cylinder_idx_bv,
            componentType=pygltflib.UNSIGNED_INT,
            count=len(cylinder_idx),
            type=pygltflib.SCALAR,
        )
    )
    cylinder_idx_acc = len(g.accessors) - 1

    # Sphere vertices
    g.accessors.append(
        pygltflib.Accessor(
            bufferView=sphere_verts_bv,
            componentType=pygltflib.FLOAT,
            count=len(sphere_verts),
            type=pygltflib.VEC3,
        )
    )
    sphere_verts_acc = len(g.accessors) - 1

    # Sphere indices
    g.accessors.append(
        pygltflib.Accessor(
            bufferView=sphere_idx_bv,
            componentType=pygltflib.UNSIGNED_INT,
            count=len(sphere_idx),
            type=pygltflib.SCALAR,
        )
    )
    sphere_idx_acc = len(g.accessors) - 1

    # Times accessor
    g.accessors.append(
        pygltflib.Accessor(
            bufferView=times_bv,
            componentType=pygltflib.FLOAT,
            count=n_frames,
            type=pygltflib.SCALAR,
        )
    )
    times_acc = len(g.accessors) - 1

    # Bone translation accessors
    bone_trans_accs: dict[int, int] = {}
    for i in range(len(H36M_SKELETON_EDGES)):
        g.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=bone_trans_offsets[i],
                byteLength=n_frames * 3 * 4,  # 3 floats * 4 bytes
                target=pygltflib.ARRAY_BUFFER,
            )
        )
        bv = len(g.bufferViews) - 1

        g.accessors.append(
            pygltflib.Accessor(
                bufferView=bv,
                componentType=pygltflib.FLOAT,
                count=n_frames,
                type=pygltflib.VEC3,
            )
        )
        bone_trans_accs[i] = len(g.accessors) - 1

    # Bone rotation accessors
    bone_rot_accs: dict[int, int] = {}
    for i in range(len(H36M_SKELETON_EDGES)):
        g.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=bone_rot_offsets[i],
                byteLength=n_frames * 4 * 4,  # 4 floats * 4 bytes
                target=pygltflib.ARRAY_BUFFER,
            )
        )
        bv = len(g.bufferViews) - 1

        g.accessors.append(
            pygltflib.Accessor(
                bufferView=bv,
                componentType=pygltflib.FLOAT,
                count=n_frames,
                type=pygltflib.VEC4,
            )
        )
        bone_rot_accs[i] = len(g.accessors) - 1

    # Bone scale accessors
    bone_scale_accs: dict[int, int] = {}
    for i in range(len(H36M_SKELETON_EDGES)):
        g.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=bone_scale_offsets[i],
                byteLength=n_frames * 3 * 4,  # 3 floats * 4 bytes
                target=pygltflib.ARRAY_BUFFER,
            )
        )
        bv = len(g.bufferViews) - 1

        g.accessors.append(
            pygltflib.Accessor(
                bufferView=bv,
                componentType=pygltflib.FLOAT,
                count=n_frames,
                type=pygltflib.VEC3,
            )
        )
        bone_scale_accs[i] = len(g.accessors) - 1

    # Joint translation accessors
    joint_trans_accs: dict[int, int] = {}
    for j in range(17):
        g.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=joint_trans_offsets[j],
                byteLength=n_frames * 3 * 4,  # 3 floats * 4 bytes
                target=pygltflib.ARRAY_BUFFER,
            )
        )
        bv = len(g.bufferViews) - 1

        g.accessors.append(
            pygltflib.Accessor(
                bufferView=bv,
                componentType=pygltflib.FLOAT,
                count=n_frames,
                type=pygltflib.VEC3,
            )
        )
        joint_trans_accs[j] = len(g.accessors) - 1

    # === MESHES ===
    # Cylinder mesh
    g.meshes.append(
        pygltflib.Mesh(
            name="BoneCylinder",
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(POSITION=cylinder_verts_acc),
                    indices=cylinder_idx_acc,
                    mode=pygltflib.TRIANGLES,
                )
            ],
        )
    )
    cylinder_mesh = len(g.meshes) - 1

    # Sphere mesh
    g.meshes.append(
        pygltflib.Mesh(
            name="JointSphere",
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(POSITION=sphere_verts_acc),
                    indices=sphere_idx_acc,
                    mode=pygltflib.TRIANGLES,
                )
            ],
        )
    )
    sphere_mesh = len(g.meshes) - 1

    # === MATERIALS ===
    # Base material (white)
    g.materials.append(
        pygltflib.Material(
            name="BoneMaterial",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.5,
            ),
        )
    )
    base_material = len(g.materials) - 1

    # Assign materials to meshes
    g.meshes[cylinder_mesh].primitives[0].material = base_material
    g.meshes[sphere_mesh].primitives[0].material = base_material

    # === NODES ===
    # Create nodes for each bone and joint
    bone_nodes: list[int] = []
    joint_nodes: list[int] = []

    # Bone nodes
    for i, (j_a, j_b) in enumerate(H36M_SKELETON_EDGES):
        color = _BONE_COLORS.get((j_a, j_b), (0.63, 0.63, 0.63))

        # Clone material for colored bone
        g.materials.append(
            pygltflib.Material(
                name=f"BoneMaterial_{i}",
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorFactor=[color[0], color[1], color[2], 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.5,
                ),
            )
        )
        material_idx = len(g.materials) - 1

        # Clone mesh with this material
        g.meshes.append(
            pygltflib.Mesh(
                name=f"Bone_{i}",
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=cylinder_verts_acc),
                        indices=cylinder_idx_acc,
                        mode=pygltflib.TRIANGLES,
                        material=material_idx,
                    )
                ],
            )
        )
        mesh_idx = len(g.meshes) - 1

        g.nodes.append(
            pygltflib.Node(
                name=f"B_{j_a}_{j_b}",
                mesh=mesh_idx,
            )
        )
        bone_nodes.append(len(g.nodes) - 1)

    # Joint nodes
    for j in range(17):
        color = _JOINT_COLOR

        # Clone material for joint
        g.materials.append(
            pygltflib.Material(
                name=f"JointMaterial_{j}",
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorFactor=[color[0], color[1], color[2], 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.5,
                ),
            )
        )
        material_idx = len(g.materials) - 1

        # Clone mesh with this material
        g.meshes.append(
            pygltflib.Mesh(
                name=f"Joint_{j}",
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=sphere_verts_acc),
                        indices=sphere_idx_acc,
                        mode=pygltflib.TRIANGLES,
                        material=material_idx,
                    )
                ],
            )
        )
        mesh_idx = len(g.meshes) - 1

        g.nodes.append(
            pygltflib.Node(
                name=f"J_{j}",
                mesh=mesh_idx,
            )
        )
        joint_nodes.append(len(g.nodes) - 1)

    # Root node
    g.nodes.append(
        pygltflib.Node(
            name="SkeletonRoot",
            children=bone_nodes + joint_nodes,
        )
    )
    root_node = len(g.nodes) - 1

    # === SCENES ===
    g.scenes.append(pygltflib.Scene(name="SkeletonScene", nodes=[root_node]))
    g.scene = 0

    # === ANIMATIONS ===
    # Create animation samplers and channels for each bone and joint
    channels = []
    samplers = []

    # Bone animations
    for i, node_idx in enumerate(bone_nodes):
        # Translation sampler
        samplers.append(
            pygltflib.AnimationSampler(
                input=times_acc,
                output=bone_trans_accs[i],
                interpolation=pygltflib.LINEAR,
            )
        )
        trans_sampler = len(samplers) - 1

        # Rotation sampler
        samplers.append(
            pygltflib.AnimationSampler(
                input=times_acc,
                output=bone_rot_accs[i],
                interpolation=pygltflib.LINEAR,
            )
        )
        rot_sampler = len(samplers) - 1

        # Scale sampler
        samplers.append(
            pygltflib.AnimationSampler(
                input=times_acc,
                output=bone_scale_accs[i],
                interpolation=pygltflib.LINEAR,
            )
        )
        scale_sampler = len(samplers) - 1

        # Channels
        channels.append(
            pygltflib.AnimationChannel(
                sampler=trans_sampler,
                target=pygltflib.AnimationChannelTarget(node=node_idx, path="translation"),
            )
        )
        channels.append(
            pygltflib.AnimationChannel(
                sampler=rot_sampler,
                target=pygltflib.AnimationChannelTarget(node=node_idx, path="rotation"),
            )
        )
        channels.append(
            pygltflib.AnimationChannel(
                sampler=scale_sampler,
                target=pygltflib.AnimationChannelTarget(node=node_idx, path="scale"),
            )
        )

    # Joint animations
    for j, node_idx in enumerate(joint_nodes):
        # Translation sampler
        samplers.append(
            pygltflib.AnimationSampler(
                input=times_acc,
                output=joint_trans_accs[j],
                interpolation=pygltflib.LINEAR,
            )
        )
        trans_sampler = len(samplers) - 1

        # Channel
        channels.append(
            pygltflib.AnimationChannel(
                sampler=trans_sampler,
                target=pygltflib.AnimationChannelTarget(node=node_idx, path="translation"),
            )
        )

    # Create animation
    g.animations.append(
        pygltflib.Animation(
            name="SkeletonAnimation",
            channels=channels,
            samplers=samplers,
        )
    )

    # === BUFFERS ===
    g.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))

    # Save to file
    tmpdir = tempfile.mkdtemp()
    glb_path = Path(tmpdir) / "skeleton_animated.glb"
    g.save_binary(str(glb_path))

    return str(glb_path)
