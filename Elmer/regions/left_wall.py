"""Generate the standalone uniform left wall from the abutment to the transition."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

from regions.center_wall import write_gmsh, write_vtu
from regions.left_wedged_wall import _upper_wall_levels
from regions.left_wedged_wall_transition import (
    DOWNSTREAM_WALL_RADIUS_M,
    FULL_WEDGE_CHAINAGE_M,
    FULL_WEDGE_HEIGHT_M,
    UPSTREAM_RADIUS_M,
    _intersection_chainage,
)
from regions.plinth import (
    _monotone_values,
    _prism_orientation,
    _section_faces,
    load_contours,
    load_global_element_size,
)


CREST_Z_M = 0.0
BODY_ID = 1
BASE_BOUNDARY_ID = 1
UPSTREAM_BOUNDARY_ID = 2
DOWNSTREAM_BOUNDARY_ID = 3
CREST_BOUNDARY_ID = 4
LEFT_END_BOUNDARY_ID = 5
RIGHT_END_BOUNDARY_ID = 6


@dataclass
class LeftWallMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    chainages_m: list[float]
    transition_chainage_m: float
    element_size_m: float


def _chainages(start_m: float, end_m: float, contour_stations_m: list[float], target_size_m: float) -> list[float]:
    anchors_m = [start_m]
    anchors_m.extend(value for value in contour_stations_m if start_m < value < end_m)
    anchors_m.append(end_m)
    chainages_m = [start_m]
    for interval_start_m, interval_end_m in zip(anchors_m, anchors_m[1:]):
        segment_count = round((interval_end_m - interval_start_m) / target_size_m)
        chainages_m.extend(
            interval_start_m + (interval_end_m - interval_start_m) * index / segment_count
            for index in range(1, segment_count + 1)
        )
    return chainages_m


def build_left_wall(root: Path) -> LeftWallMesh:
    element_size_m = load_global_element_size(root / "Data" / "Computational_Grid_Controls.json")
    contours = load_contours(root / "Data" / "plinth.json")
    config = json.loads((root / "config.json").read_text())
    centerline_radius_m = float(config["wall_centerline_radius_m"])
    start_chainage_m = contours[0].chainage_m
    full_wedge_base_z_m = _monotone_values(contours, "plinth_z_m", FULL_WEDGE_CHAINAGE_M)
    reference_top_z_m = full_wedge_base_z_m + FULL_WEDGE_HEIGHT_M
    transition_chainage_m = _intersection_chainage(contours, reference_top_z_m)
    chainages_m = _chainages(
        start_chainage_m,
        transition_chainage_m,
        [point.chainage_m for point in contours],
        element_size_m,
    )
    base_levels_m = [_monotone_values(contours, "plinth_z_m", value) for value in chainages_m]
    radial_divisions = round((UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M) / element_size_m)
    wall_radii_m = [
        DOWNSTREAM_WALL_RADIUS_M + index * element_size_m
        for index in range(radial_divisions + 1)
    ]
    lower_layer_count = round(FULL_WEDGE_HEIGHT_M / element_size_m)
    fixed_levels_m = [
        full_wedge_base_z_m + index * element_size_m
        for index in range(lower_layer_count)
    ] + _upper_wall_levels(full_wedge_base_z_m, element_size_m)
    section_levels_m = [
        [base_z_m] + [level_m for level_m in fixed_levels_m if level_m > base_z_m + 1.0e-9]
        for base_z_m in base_levels_m
    ]

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
        levels = (section_levels_m[station_index], section_levels_m[station_index + 1])
        for face in _section_faces(*levels):
            for radial_index in range(radial_divisions):
                inner = tuple(
                    node_id(station_index + side, wall_radii_m[radial_index], levels[side][level_index])
                    for side, level_index in face
                )
                outer = tuple(
                    node_id(station_index + side, wall_radii_m[radial_index + 1], levels[side][level_index])
                    for side, level_index in face
                )
                if len(face) == 3:
                    cell = inner + outer
                    if _prism_orientation(nodes, cell) < 0.0:
                        cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
                    cells.append((6, cell))
                else:
                    cells.append((5, (
                        inner[0], inner[1], outer[1], outer[0],
                        inner[3], inner[2], outer[2], outer[3],
                    )))

    face_patterns = {
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    }
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
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
        coordinates = [nodes[value - 1] for value in face]
        radii_m = [math.hypot(x_m, y_m) for x_m, y_m, _ in coordinates]
        stations_m = [centerline_radius_m * math.atan2(x_m, y_m) for x_m, y_m, _ in coordinates]
        z_values_m = [z_m for _, _, z_m in coordinates]
        if max(abs(radius_m - UPSTREAM_RADIUS_M) for radius_m in radii_m) < tolerance:
            boundary_id = UPSTREAM_BOUNDARY_ID
        elif max(abs(z_m - CREST_Z_M) for z_m in z_values_m) < tolerance:
            boundary_id = CREST_BOUNDARY_ID
        elif max(abs(station_m - start_chainage_m) for station_m in stations_m) < tolerance:
            boundary_id = LEFT_END_BOUNDARY_ID
        elif max(abs(station_m - transition_chainage_m) for station_m in stations_m) < tolerance:
            boundary_id = RIGHT_END_BOUNDARY_ID
        elif all(
            abs(z_m - _monotone_values(contours, "plinth_z_m", station_m)) < tolerance
            for station_m, z_m in zip(stations_m, z_values_m)
        ):
            boundary_id = BASE_BOUNDARY_ID
        else:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return LeftWallMesh(nodes, cells, boundaries, chainages_m, transition_chainage_m, element_size_m)


def audit_left_wall(mesh: LeftWallMesh) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    boundary_counts = Counter(boundary_id for boundary_id, _ in mesh.boundaries)
    expected_boundaries = {BASE_BOUNDARY_ID, UPSTREAM_BOUNDARY_ID, DOWNSTREAM_BOUNDARY_ID, CREST_BOUNDARY_ID, RIGHT_END_BOUNDARY_ID}
    if set(boundary_counts) != expected_boundaries:
        raise ValueError(f"Missing left-wall boundaries: found {sorted(boundary_counts)}")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "boundary_faces": dict(sorted(boundary_counts.items())),
        "chainage_m": [mesh.chainages_m[0], mesh.transition_chainage_m],
        "plinth_z_m": [CREST_Z_M, -15.03],
        "wall_radius_m": [DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M],
        "wall_thickness_m": UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M,
        "element_size_m": mesh.element_size_m,
    }


__all__ = ["audit_left_wall", "build_left_wall", "write_gmsh", "write_vtu"]
