"""Build the concrete fill and mortar returns for the centre downstream step."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .model_groups import FACE_GROUP_IDS
from .mortar_interface import MORTAR_THICKNESS_M, MORTAR_WALL_BOUNDARY_ID
from .plinth import _hex_orientation
from .uniform_wall import (
    CENTER_STEP_END_CHAINAGE_M,
    CENTER_STEP_RADIUS_M,
    CENTER_STEP_START_CHAINAGE_M,
    CENTER_STEP_TOP_Z_M,
    DOWNSTREAM_BATTER_BASE_Z_M,
)


STEP_MORTAR_BOUNDARY_ID = 9
PLINTH_RADIUS_M = 74.0


@dataclass
class StepMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]


def _step_chainages(wall) -> list[float]:
    chainages_m = [
        chainage_m
        for chainage_m in wall.chainages_m
        if CENTER_STEP_START_CHAINAGE_M - 1.0e-9 <= chainage_m <= CENTER_STEP_END_CHAINAGE_M + 1.0e-9
    ]
    if not chainages_m or abs(chainages_m[0] - CENTER_STEP_START_CHAINAGE_M) > 1.0e-9 or abs(chainages_m[-1] - CENTER_STEP_END_CHAINAGE_M) > 1.0e-9:
        raise ValueError("Step fill requires exact centre-step chainage stations")
    return chainages_m


def _build_mesh(
    root: Path,
    wall,
    radial_bounds_m: tuple[float, float],
    z_levels_m: tuple[float, ...],
    interface_faces: dict[str, int],
) -> StepMesh:
    centerline_radius_m = float(json.loads((root / "config.json").read_text())["wall_centerline_radius_m"])
    chainages_m = _step_chainages(wall)
    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[float, float, float], int] = {}

    def node_id(chainage_m: float, radius_m: float, z_m: float) -> int:
        angle = chainage_m / centerline_radius_m
        point = (radius_m * math.sin(angle), radius_m * math.cos(angle), z_m)
        key = tuple(round(value, 9) for value in point)
        if key not in node_ids:
            node_ids[key] = len(nodes) + 1
            nodes.append(point)
        return node_ids[key]

    cells: list[tuple[int, tuple[int, ...]]] = []
    for start_m, end_m in zip(chainages_m, chainages_m[1:]):
        lower_radius_m, upper_radius_m = radial_bounds_m
        for lower_z_m, upper_z_m in zip(z_levels_m, z_levels_m[1:]):
            cell = (
                node_id(start_m, lower_radius_m, lower_z_m),
                node_id(end_m, lower_radius_m, lower_z_m),
                node_id(end_m, upper_radius_m, lower_z_m),
                node_id(start_m, upper_radius_m, lower_z_m),
                node_id(start_m, lower_radius_m, upper_z_m),
                node_id(end_m, lower_radius_m, upper_z_m),
                node_id(end_m, upper_radius_m, upper_z_m),
                node_id(start_m, upper_radius_m, upper_z_m),
            )
            if _hex_orientation(nodes, cell) < 0.0:
                cell = (cell[0], cell[3], cell[2], cell[1], cell[4], cell[7], cell[6], cell[5])
            cells.append((5, cell))

    face_patterns = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for _, cell in cells:
        for pattern in face_patterns:
            face = tuple(cell[index] for index in pattern)
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face

    boundaries = []
    tolerance = 1.0e-6
    for key, count in face_counts.items():
        if count != 1:
            continue
        face = oriented_faces[key]
        points = [nodes[node_id - 1] for node_id in face]
        chainages = [centerline_radius_m * math.atan2(x_m, y_m) for x_m, y_m, _ in points]
        radii = [math.hypot(x_m, y_m) for x_m, y_m, _ in points]
        z_values = [z_m for _, _, z_m in points]
        if max(abs(radius_m - radial_bounds_m[0]) for radius_m in radii) < tolerance and "radial_lower" in interface_faces:
            boundary_id = interface_faces["radial_lower"]
        elif max(abs(radius_m - radial_bounds_m[1]) for radius_m in radii) < tolerance and "radial_upper" in interface_faces:
            boundary_id = interface_faces["radial_upper"]
        elif max(abs(z_m - z_levels_m[0]) for z_m in z_values) < tolerance and "z_lower" in interface_faces:
            boundary_id = interface_faces["z_lower"]
        elif max(abs(z_m - z_levels_m[-1]) for z_m in z_values) < tolerance and "z_upper" in interface_faces:
            boundary_id = interface_faces["z_upper"]
        elif min(chainages) <= CENTER_STEP_START_CHAINAGE_M + tolerance or max(chainages) >= CENTER_STEP_END_CHAINAGE_M - tolerance:
            boundary_id = FACE_GROUP_IDS["OTHER_EXTERIOR"]
        else:
            boundary_id = FACE_GROUP_IDS["DOWNSTREAM"]
        boundaries.append((boundary_id, face))
    return StepMesh(nodes, cells, boundaries)


def build_step_fill(root: Path, wall) -> StepMesh:
    return _build_mesh(
        root,
        wall,
        (PLINTH_RADIUS_M, CENTER_STEP_RADIUS_M),
        (DOWNSTREAM_BATTER_BASE_Z_M, DOWNSTREAM_BATTER_BASE_Z_M + 0.5, CENTER_STEP_TOP_Z_M),
        {
            "z_lower": STEP_MORTAR_BOUNDARY_ID,
        },
    )


def build_step_mortar(root: Path, wall) -> StepMesh:
    return _build_mesh(
        root,
        wall,
        (PLINTH_RADIUS_M, CENTER_STEP_RADIUS_M),
        (DOWNSTREAM_BATTER_BASE_Z_M - MORTAR_THICKNESS_M, DOWNSTREAM_BATTER_BASE_Z_M),
        {
            "z_lower": FACE_GROUP_IDS["FOUNDATION"],
        },
    )