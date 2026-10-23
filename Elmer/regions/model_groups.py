"""Semantic group contract shared by the mesh model and solver adapters."""

REGION_GROUP_IDS = {
    "WALL": 1,
}

REGION_NAMES = {
    "WALL": "wall",
}

FACE_GROUP_IDS = {
    "FOUNDATION": 1,
    "UPSTREAM": 2,
    "DOWNSTREAM": 3,
    "CREST": 4,
    "LEFT_ABUTMENT": 5,
    "RIGHT_ABUTMENT": 6,
    "OTHER_EXTERIOR": 7,
}

# Internal spring joints between wall thickness blocks (geometry.wall.joint_support).
# Joint k (0 = highest step) is tagged FIRST + 2k on its upper-block side and
# FIRST + 2k + 1 on its lower-block side; tags 10-16 belong to the rock model.
# Optional concrete trench body: wall=1, bedrock=2, foundation=3.
# Tags 11/12 retain the lower (foundation/bedrock) spring-pair role when present;
# 17/18 are the wall/foundation mortar slave/master pair. IDs in diagnostic.json
# are semantic source IDs; boundary_id_map maps them to saved mesh/SIF IDs.
FOUNDATION_REGION_ID = 3
WALL_FOUNDATION_BOUNDARY_IDS = (17, 18)
FOUNDATION_BEDROCK_BOUNDARY_IDS = (11, 12)
FIRST_WALL_JOINT_BOUNDARY_ID = 21
MAX_WALL_JOINTS = 20
# Connected near-planar foundation mortar groups: adjacent IDs from 1000.
# 11/12 and 17/18 remain the first group of each logical interface;
# additional groups use this namespace, safely above wall joints 21..60.
# IDs depend on deterministic face traversal; consume diagnostic spring_pairs.faces,
# never infer them from mesh ordering or assume legacy tags cover all faces.
FIRST_PLANAR_MORTAR_BOUNDARY_ID = 1000


def joint_boundary_ids(joint_index: int) -> tuple[int, int]:
    """(upper side, lower side) boundary tags of wall joint joint_index."""
    if not 0 <= joint_index < MAX_WALL_JOINTS:
        raise ValueError(f"Wall joint index must be in [0, {MAX_WALL_JOINTS})")
    upper = FIRST_WALL_JOINT_BOUNDARY_ID + 2 * joint_index
    return upper, upper + 1


def is_wall_joint_boundary_id(boundary_id: int) -> bool:
    return FIRST_WALL_JOINT_BOUNDARY_ID <= boundary_id < FIRST_WALL_JOINT_BOUNDARY_ID + 2 * MAX_WALL_JOINTS


GROUP_ENTITIES = {
    **{name: "volume" for name in REGION_GROUP_IDS},
    **{name: "face" for name in FACE_GROUP_IDS},
    "WALL_JOINT": "face",
}

REQUIRED_GROUPS = (
    "WALL",
    "FOUNDATION",
    "UPSTREAM",
    "DOWNSTREAM",
    "CREST",
)


def build_named_groups(mesh) -> dict[str, tuple[int, ...]]:
    """Return zero-based cell/face indices keyed by stable semantic group names."""
    groups: dict[str, list[int]] = {name: [] for name in GROUP_ENTITIES}
    region_names_by_id = {
        group_id: name
        for name, group_id in REGION_GROUP_IDS.items()
    }
    boundary_names_by_id = {
        group_id: name
        for name, group_id in FACE_GROUP_IDS.items()
    }

    for cell_index, region_id in enumerate(mesh.region_ids):
        try:
            group_name = region_names_by_id[region_id]
        except KeyError as error:
            raise ValueError(f"Unknown volume region ID: {region_id}") from error
        groups[group_name].append(cell_index)

    for face_index, (boundary_id, _) in enumerate(mesh.boundaries):
        if is_wall_joint_boundary_id(boundary_id):
            groups["WALL_JOINT"].append(face_index)
            continue
        try:
            group_name = boundary_names_by_id[boundary_id]
        except KeyError as error:
            raise ValueError(f"Unknown exterior boundary ID: {boundary_id}") from error
        groups[group_name].append(face_index)

    for group_name, region_name in REGION_NAMES.items():
        expected_count = mesh.region_cell_counts.get(region_name, 0)
        if len(groups[group_name]) != expected_count:
            raise ValueError(
                f"{group_name} group has {len(groups[group_name])} cells; "
                f"expected {expected_count}"
            )

    empty_required = [name for name in REQUIRED_GROUPS if not groups[name]]
    if empty_required:
        raise ValueError(f"Required mesh groups are empty: {', '.join(empty_required)}")

    return {name: tuple(indices) for name, indices in groups.items()}