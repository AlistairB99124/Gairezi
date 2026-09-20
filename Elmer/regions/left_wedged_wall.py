"""Generate the standalone left wall with its integral smooth downstream toe."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

from regions.center_wall import write_gmsh, write_vtu
from regions.plinth import _monotone_values, _section_faces, load_contours, load_global_element_size


START_CHAINAGE_M = 62.0
END_CHAINAGE_M = 101.0
WEDGE_TOE_RADIUS_M = 74.0
DOWNSTREAM_WALL_RADIUS_M = 76.0
UPSTREAM_RADIUS_M = 80.0
WEDGE_SIZE_M = 2.0
CREST_Z_M = 0.0
BODY_ID = 1
BASE_BOUNDARY_ID = 1
UPSTREAM_BOUNDARY_ID = 2
DOWNSTREAM_BOUNDARY_ID = 3
CREST_BOUNDARY_ID = 4
LEFT_END_BOUNDARY_ID = 5
RIGHT_END_BOUNDARY_ID = 6


@dataclass
class LeftWedgedWallMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    element_size_m: float


def _chainages(element_size_m: float) -> list[float]:
    count = round((END_CHAINAGE_M - START_CHAINAGE_M) / element_size_m)
    return [START_CHAINAGE_M + index * element_size_m for index in range(count + 1)]


def _upper_wall_levels(base_z_m: float, element_size_m: float) -> list[float]:
    transition_top_z_m = base_z_m + WEDGE_SIZE_M
    levels = [round(transition_top_z_m, 9)]
    first_global_index = math.floor(transition_top_z_m / element_size_m) + 1
    levels.extend(
        index * element_size_m
        for index in range(first_global_index, 1)
        if index * element_size_m < CREST_Z_M - 1.0e-9
    )
    levels.append(CREST_Z_M)
    return list(dict.fromkeys(levels))


def build_left_wedged_wall(root: Path) -> LeftWedgedWallMesh:
    element_size_m = load_global_element_size(root / "Data" / "Computational_Grid_Controls.json")
    contours = load_contours(root / "Data" / "plinth.json")
    config = json.loads((root / "config.json").read_text())
    centerline_radius_m = float(config["wall_centerline_radius_m"])
    chainages_m = _chainages(element_size_m)
    base_levels_m = [_monotone_values(contours, "plinth_z_m", chainage_m) for chainage_m in chainages_m]
    wall_radii_m = [
        DOWNSTREAM_WALL_RADIUS_M + index * element_size_m
        for index in range(round((UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M) / element_size_m) + 1)
    ]
    wedge_divisions = round(WEDGE_SIZE_M / element_size_m)

    nodes: list[tuple[float, float, float]] = []
    coordinate_nodes: dict[tuple[float, float, float], int] = {}

    def node_id(station_index: int, radius_m: float, z_m: float) -> int:
        angle = chainages_m[station_index] / centerline_radius_m
        coordinate = (radius_m * math.sin(angle), radius_m * math.cos(angle), z_m)
        key = tuple(round(value, 9) for value in coordinate)
        if key not in coordinate_nodes:
            coordinate_nodes[key] = len(nodes) + 1
            nodes.append(coordinate)
        return coordinate_nodes[key]

    cells: list[tuple[int, tuple[int, ...]]] = []
    for station_index in range(len(chainages_m) - 1):
        for layer_index in range(wedge_divisions):
            lower_start = base_levels_m[station_index] + layer_index * element_size_m
            upper_start = lower_start + element_size_m
            lower_end = base_levels_m[station_index + 1] + layer_index * element_size_m
            upper_end = lower_end + element_size_m
            for radial_index in range(len(wall_radii_m) - 1):
                inner_radius_m = wall_radii_m[radial_index]
                outer_radius_m = wall_radii_m[radial_index + 1]
                cells.append((5, (
                    node_id(station_index, inner_radius_m, lower_start),
                    node_id(station_index + 1, inner_radius_m, lower_end),
                    node_id(station_index + 1, outer_radius_m, lower_end),
                    node_id(station_index, outer_radius_m, lower_start),
                    node_id(station_index, inner_radius_m, upper_start),
                    node_id(station_index + 1, inner_radius_m, upper_end),
                    node_id(station_index + 1, outer_radius_m, upper_end),
                    node_id(station_index, outer_radius_m, upper_start),
                )))

        upper_start_levels = _upper_wall_levels(base_levels_m[station_index], element_size_m)
        upper_end_levels = _upper_wall_levels(base_levels_m[station_index + 1], element_size_m)
        for face in _section_faces(upper_start_levels, upper_end_levels):
            for radial_index in range(len(wall_radii_m) - 1):
                inner = tuple(node_id(station_index + side, wall_radii_m[radial_index], (upper_start_levels, upper_end_levels)[side][level_index]) for side, level_index in face)
                outer = tuple(node_id(station_index + side, wall_radii_m[radial_index + 1], (upper_start_levels, upper_end_levels)[side][level_index]) for side, level_index in face)
                if len(face) == 3:
                    cells.append((6, inner + outer))
                else:
                    cells.append((5, (
                        inner[0], inner[1], outer[1], outer[0],
                        inner[3], inner[2], outer[2], outer[3],
                    )))

        for radial_index in range(wedge_divisions):
            for vertical_index in range(radial_index + 1):
                triangles = [
                    ((radial_index, vertical_index), (radial_index + 1, vertical_index), (radial_index + 1, vertical_index + 1)),
                ]
                if vertical_index < radial_index:
                    triangles.append(((radial_index, vertical_index), (radial_index + 1, vertical_index + 1), (radial_index, vertical_index + 1)))
                for triangle in triangles:
                    start_face = tuple(
                        node_id(station_index, WEDGE_TOE_RADIUS_M + radial * element_size_m, base_levels_m[station_index] + vertical * element_size_m)
                        for radial, vertical in triangle
                    )
                    end_face = tuple(
                        node_id(station_index + 1, WEDGE_TOE_RADIUS_M + radial * element_size_m, base_levels_m[station_index + 1] + vertical * element_size_m)
                        for radial, vertical in triangle
                    )
                    cells.append((6, start_face + end_face))

    face_patterns = {
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    }
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces = {}
    for element_type, cell in cells:
        for pattern in face_patterns[element_type]:
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
        coordinates = [nodes[node_id_value - 1] for node_id_value in face]
        radii_m = [math.hypot(x_m, y_m) for x_m, y_m, _ in coordinates]
        stations_m = [centerline_radius_m * math.atan2(x_m, y_m) for x_m, y_m, _ in coordinates]
        z_values_m = [z_m for _, _, z_m in coordinates]
        if max(abs(radius_m - UPSTREAM_RADIUS_M) for radius_m in radii_m) < tolerance:
            boundary_id = UPSTREAM_BOUNDARY_ID
        elif max(abs(z_m - CREST_Z_M) for z_m in z_values_m) < tolerance:
            boundary_id = CREST_BOUNDARY_ID
        elif max(abs(station_m - START_CHAINAGE_M) for station_m in stations_m) < tolerance:
            boundary_id = LEFT_END_BOUNDARY_ID
        elif max(abs(station_m - END_CHAINAGE_M) for station_m in stations_m) < tolerance:
            boundary_id = RIGHT_END_BOUNDARY_ID
        elif all(abs(z_m - _monotone_values(contours, "plinth_z_m", station_m)) < tolerance for station_m, z_m in zip(stations_m, z_values_m)):
            boundary_id = BASE_BOUNDARY_ID
        else:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return LeftWedgedWallMesh(nodes, cells, boundaries, element_size_m)


def audit_left_wedged_wall(mesh: LeftWedgedWallMesh) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    boundary_counts = Counter(boundary_id for boundary_id, _ in mesh.boundaries)
    if set(boundary_counts) != set(range(1, 7)):
        raise ValueError(f"Missing left-wedged-wall boundaries: found {sorted(boundary_counts)}")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "boundary_faces": dict(sorted(boundary_counts.items())),
        "chainage_m": [START_CHAINAGE_M, END_CHAINAGE_M],
        "plinth_z_m": [-17.03, -28.5],
        "wall_radius_m": [DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M],
        "wedge_radius_m": [WEDGE_TOE_RADIUS_M, DOWNSTREAM_WALL_RADIUS_M],
        "wedge_size_m": WEDGE_SIZE_M,
        "element_size_m": mesh.element_size_m,
    }


__all__ = ["audit_left_wedged_wall", "build_left_wedged_wall", "write_gmsh", "write_vtu"]
