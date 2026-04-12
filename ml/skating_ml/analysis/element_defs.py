"""Figure skating element definitions and ideal metrics.

This module defines the skating elements supported by the system,
including their biomechanical characteristics and ideal performance ranges.
"""

from dataclasses import dataclass

from ..types import H36Key


@dataclass(frozen=True)
class ElementDef:
    """Definition of a figure skating element.

    Attributes:
        name: Element identifier (e.g., 'three_turn', 'waltz_jump').
        name_ru: Russian name for display.
        rotations: Number of rotations (0 for steps, 1+ for jumps).
        has_toe_pick: True if takeoff uses toe pick (toe loop, flip, lutz).
        key_joints: List of H36Key indices relevant for analysis.
        ideal_metrics: Dict of metric_name -> (min_good, max_good) ranges.
    """

    name: str
    name_ru: str
    rotations: int
    has_toe_pick: bool
    key_joints: list[int]
    ideal_metrics: dict[str, tuple[float, float]]


# Element definitions ordered by complexity
ELEMENT_DEFS: dict[str, ElementDef] = {
    "three_turn": ElementDef(
        name="three_turn",
        name_ru="тройка",
        rotations=0,
        has_toe_pick=False,
        key_joints=[
            H36Key.LHIP,
            H36Key.RHIP,
            H36Key.LKNEE,
            H36Key.RKNEE,
            H36Key.LFOOT,
            H36Key.RFOOT,
            H36Key.LSHOULDER,
            H36Key.RSHOULDER,
        ],
        ideal_metrics={
            "knee_angle": (100, 140),  # Knee bend during entry (flexed knees)
            "trunk_lean": (-15, 20),  # Torso angle relative to vertical (slight forward OK)
            "edge_change_smoothness": (0.1, 0.5),  # Smooth edge transition (low std = smooth)
            "symmetry": (0.6, 1.0),  # Body symmetry score
        },
    ),
    "waltz_jump": ElementDef(
        name="waltz_jump",
        name_ru="вальсовый прыжок",
        rotations=1,  # Half jump, but treated as jump for analysis
        has_toe_pick=False,
        key_joints=[
            H36Key.LHIP,
            H36Key.RHIP,
            H36Key.LKNEE,
            H36Key.RKNEE,
            H36Key.LFOOT,
            H36Key.RFOOT,
            H36Key.LSHOULDER,
            H36Key.RSHOULDER,
            H36Key.LWRIST,
            H36Key.RWRIST,
        ],
        ideal_metrics={
            "airtime": (0.3, 0.7),  # Seconds in flight
            "max_height": (0.2, 0.5),  # Normalized height units
            "landing_knee_angle": (90, 130),  # Knee angle at landing (shock absorption)
            "arm_position_score": (0.6, 1.0),  # Arms controlled (close to body)
            "takeoff_angle": (70, 85),  # Takeoff angle relative to ice
            "landing_knee_stability": (0.5, 1.0),  # Knee stability after landing
            "landing_trunk_recovery": (0.5, 1.0),  # Trunk stays upright after landing
            "relative_jump_height": (0.3, 1.5),  # Height normalized by spine length
        },
    ),
    "toe_loop": ElementDef(
        name="toe_loop",
        name_ru="перекидной",
        rotations=1,
        has_toe_pick=True,
        key_joints=[
            H36Key.LHIP,
            H36Key.RHIP,
            H36Key.LKNEE,
            H36Key.RKNEE,
            H36Key.LFOOT,
            H36Key.RFOOT,
            H36Key.LSHOULDER,
            H36Key.RSHOULDER,
        ],
        ideal_metrics={
            "airtime": (0.35, 0.6),  # Seconds for single rotation
            "rotation_speed": (300, 500),  # Degrees per second
            "landing_knee_angle": (90, 125),  # Knee angle at landing
            "edge_quality": (0.7, 1.0),  # Clean edge on landing
            "toe_pick_timing": (0.1, 0.3),  # Time from toe pick to takeoff
            "landing_knee_stability": (0.5, 1.0),  # Knee stability after landing
            "landing_trunk_recovery": (0.5, 1.0),  # Trunk stays upright after landing
            "relative_jump_height": (0.3, 1.5),  # Height normalized by spine length
        },
    ),
    "flip": ElementDef(
        name="flip",
        name_ru="флип",
        rotations=1,
        has_toe_pick=True,
        key_joints=[
            H36Key.LHIP,
            H36Key.RHIP,
            H36Key.LKNEE,
            H36Key.RKNEE,
            H36Key.LFOOT,
            H36Key.RFOOT,
            H36Key.LSHOULDER,
            H36Key.RSHOULDER,
        ],
        ideal_metrics={
            "airtime": (0.35, 0.6),  # Seconds for single rotation
            "rotation_speed": (350, 550),  # Degrees per second
            "landing_knee_angle": (90, 125),  # Knee angle at landing
            "pick_quality": (0.7, 1.0),  # Clean toe pick
            "air_position": (0.7, 1.0),  # Body position in air (tight vs loose)
            "landing_knee_stability": (0.5, 1.0),  # Knee stability after landing
            "landing_trunk_recovery": (0.5, 1.0),  # Trunk stays upright after landing
            "relative_jump_height": (0.3, 1.5),  # Height normalized by spine length
        },
    ),
    "salchow": ElementDef(
        name="salchow",
        name_ru="перекидной",
        rotations=1,
        has_toe_pick=False,
        key_joints=[
            H36Key.LHIP,
            H36Key.RHIP,
            H36Key.LKNEE,
            H36Key.RKNEE,
            H36Key.LFOOT,
            H36Key.RFOOT,
            H36Key.LSHOULDER,
            H36Key.RSHOULDER,
        ],
        ideal_metrics={
            "airtime": (0.3, 0.6),
            "max_height": (0.15, 0.4),
            "landing_knee_angle": (90, 130),
            "rotation_speed": (300, 500),
            "takeoff_angle": (65, 85),
            "landing_knee_stability": (0.5, 1.0),  # Knee stability after landing
            "landing_trunk_recovery": (0.5, 1.0),  # Trunk stays upright after landing
            "relative_jump_height": (0.3, 1.5),  # Height normalized by spine length
        },
    ),
    "loop": ElementDef(
        name="loop",
        name_ru="петля",
        rotations=1,
        has_toe_pick=False,
        key_joints=[
            H36Key.LHIP,
            H36Key.RHIP,
            H36Key.LKNEE,
            H36Key.RKNEE,
            H36Key.LFOOT,
            H36Key.RFOOT,
            H36Key.LSHOULDER,
            H36Key.RSHOULDER,
        ],
        ideal_metrics={
            "airtime": (0.3, 0.6),
            "max_height": (0.15, 0.4),
            "landing_knee_angle": (90, 130),
            "rotation_speed": (300, 500),
            "landing_knee_stability": (0.5, 1.0),  # Knee stability after landing
            "landing_trunk_recovery": (0.5, 1.0),  # Trunk stays upright after landing
            "relative_jump_height": (0.3, 1.5),  # Height normalized by spine length
        },
    ),
    "lutz": ElementDef(
        name="lutz",
        name_ru="льютц",
        rotations=1,
        has_toe_pick=True,
        key_joints=[
            H36Key.LHIP,
            H36Key.RHIP,
            H36Key.LKNEE,
            H36Key.RKNEE,
            H36Key.LFOOT,
            H36Key.RFOOT,
            H36Key.LSHOULDER,
            H36Key.RSHOULDER,
        ],
        ideal_metrics={
            "airtime": (0.35, 0.6),
            "max_height": (0.15, 0.4),
            "landing_knee_angle": (90, 125),
            "pick_quality": (0.7, 1.0),
            "rotation_speed": (350, 550),
            "landing_knee_stability": (0.5, 1.0),  # Knee stability after landing
            "landing_trunk_recovery": (0.5, 1.0),  # Trunk stays upright after landing
            "relative_jump_height": (0.3, 1.5),  # Height normalized by spine length
        },
    ),
    "axel": ElementDef(
        name="axel",
        name_ru="аксель",
        rotations=int(1.5),
        has_toe_pick=False,
        key_joints=[
            H36Key.LHIP,
            H36Key.RHIP,
            H36Key.LKNEE,
            H36Key.RKNEE,
            H36Key.LFOOT,
            H36Key.RFOOT,
            H36Key.LSHOULDER,
            H36Key.RSHOULDER,
        ],
        ideal_metrics={
            "airtime": (0.4, 0.7),
            "max_height": (0.2, 0.5),
            "landing_knee_angle": (90, 130),
            "rotation_speed": (350, 550),
            "takeoff_angle": (65, 85),
            "landing_knee_stability": (0.5, 1.0),  # Knee stability after landing
            "landing_trunk_recovery": (0.5, 1.0),  # Trunk stays upright after landing
            "relative_jump_height": (0.3, 1.5),  # Height normalized by spine length
        },
    ),
}


def get_element_def(element_type: str) -> ElementDef | None:
    """Get element definition by type.

    Args:
        element_type: Element identifier (e.g., 'three_turn').

    Returns:
        ElementDef or None if not found.
    """
    return ELEMENT_DEFS.get(element_type)


def list_supported_elements() -> list[str]:
    """List all supported element types.

    Returns:
        List of element type identifiers.
    """
    return list(ELEMENT_DEFS.keys())


def is_jump(element_type: str) -> bool:
    """Check if element is a jump.

    Args:
        element_type: Element identifier.

    Returns:
        True if element has takeoff/flight phases.
    """
    element_def = get_element_def(element_type)
    return element_def.rotations > 0 if element_def else False
