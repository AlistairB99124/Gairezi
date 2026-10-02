"""Semantic group contract shared by the mesh model and solver adapters."""

REGION_GROUP_IDS = {
    "PLINTH": 1,
    "WALL": 2,
    "HAUNCH": 3,
}

REGION_NAMES = {
    "PLINTH": "plinth",
    "WALL": "wall",
    "HAUNCH": "haunch",
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

GROUP_ENTITIES = {
    **{name: "volume" for name in REGION_GROUP_IDS},
    **{name: "face" for name in FACE_GROUP_IDS},
}

REQUIRED_GROUPS = (
    "PLINTH",
    "WALL",
    "HAUNCH",
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