"""Generate the left transition into the full-height left wedged wall."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

from regions.center_wall import write_gmsh, write_vtu
from regions.left_wedged_wall import _upper_wall_levels
from regions.plinth import (
    _monotone_values,
    _section_faces,
    load_contours,
    load_global_element_size,
)


FULL_WEDGE_CHAINAGE_M = 62.0
DOWNSTREAM_WALL_RADIUS_M = 76.0
UPSTREAM_RADIUS_M = 80.0
FULL_WEDGE_HEIGHT_M = 2.0
CREST_Z_M = 0.0
BODY_ID = 1
BASE_BOUNDARY_ID = 1
UPSTREAM_BOUNDARY_ID = 2
DOWNSTREAM_BOUNDARY_ID = 3
CREST_BOUNDARY_ID = 4
LEFT_END_BOUNDARY_ID = 5
RIGHT_END_BOUNDARY_ID = 6


@dataclass
class LeftWedgedWallTransitionMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    chainages_m: list[float]
    intersection_chainage_m: float
    wedge_top_z_m: float
    element_size_m: float


def _intersection_chainage(contours, target_z_m: float) -> float:
    lower_chainage_m = next(
        point.chainage_m
        for point, following in zip(contours, contours[1:])
        if point.plinth_z_m >= target_z_m >= following.plinth_z_m
    )
    upper_chainage_m = next(
        following.chainage_m
        for point, following in zip(contours, contours[1:])
        if point.plinth_z_m >= target_z_m >= following.plinth_z_m
    )
    for _ in range(80):
        midpoint_m = 0.5 * (lower_chainage_m + upper_chainage_m)
        if _monotone_values(contours, "plinth_z_m", midpoint_m) > target_z_m:
            lower_chainage_m = midpoint_m
        else:
            upper_chainage_m = midpoint_m
    return 0.5 * (lower_chainage_m + upper_chainage_m)


def _chainages(start_m: float, end_m: float, target_size_m: float) -> list[float]:
    segment_count = round((end_m - start_m) / target_size_m)
    return [start_m + (end_m - start_m) * index / segment_count for index in range(segment_count + 1)]


def _chainage_at_elevation(contours, target_z_m: float, start_m: float, end_m: float) -> float:
    lower_m, upper_m = sorted((start_m, end_m))
    lower_delta = _monotone_values(contours, "plinth_z_m", lower_m) - target_z_m
    upper_delta = _monotone_values(contours, "plinth_z_m", upper_m) - target_z_m
    if lower_delta * upper_delta > 0.0:
        raise ValueError(f"Elevation {target_z_m:g} is not bracketed by the transition")
    for _ in range(80):
        midpoint_m = 0.5 * (lower_m + upper_m)
        midpoint_delta = _monotone_values(contours, "plinth_z_m", midpoint_m) - target_z_m
        if lower_delta * midpoint_delta <= 0.0:
            upper_m = midpoint_m
        else:
            lower_m = midpoint_m
            lower_delta = midpoint_delta
    return 0.5 * (lower_m + upper_m)


def build_wedged_wall_transition(
    root: Path,
    full_wedge_chainage_m: float,
    intersection_chainage_m: float,
) -> LeftWedgedWallTransitionMesh:
    element_size_m = load_global_element_size(root / "Data" / "Computational_Grid_Controls.json")
    contours = load_contours(root / "Data" / "plinth.json")
    config = json.loads((root / "config.json").read_text())
    centerline_radius_m = float(config["wall_centerline_radius_m"])
    full_wedge_base_z_m = _monotone_values(contours, "plinth_z_m", full_wedge_chainage_m)
    wedge_top_z_m = full_wedge_base_z_m + FULL_WEDGE_HEIGHT_M
    taper_end_chainage_m = _chainage_at_elevation(
        contours,
        wedge_top_z_m - element_size_m,
        full_wedge_chainage_m,
        intersection_chainage_m,
    )
    anchors_m = sorted((intersection_chainage_m, taper_end_chainage_m, full_wedge_chainage_m))
    chainages_m = []
    for start_m, end_m in zip(anchors_m, anchors_m[1:]):
        values = _chainages(start_m, end_m, element_size_m)
        chainages_m.extend(values if not chainages_m else values[1:])
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
    wedge_heights_m = [wedge_top_z_m - base_z_m for base_z_m in base_levels_m]
    wedge_divisions = round(FULL_WEDGE_HEIGHT_M / element_size_m)
    section_levels_m = [
        (
            [base_z_m + wedge_height_m * index / wedge_divisions for index in range(wedge_divisions)]
            + [level_m for level_m in fixed_levels_m if level_m > base_z_m + wedge_height_m - 1.0e-9]
            if wedge_height_m >= element_size_m - 1.0e-9
            else [base_z_m] + [level_m for level_m in fixed_levels_m if level_m > base_z_m + 1.0e-9]
        )
        for base_z_m, wedge_height_m in zip(base_levels_m, wedge_heights_m)
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
        for face in _section_faces(section_levels_m[station_index], section_levels_m[station_index + 1]):
            for radial_index in range(radial_divisions):
                levels = (section_levels_m[station_index], section_levels_m[station_index + 1])
                inner = tuple(
                    node_id(station_index + side, wall_radii_m[radial_index], levels[side][level_index])
                    for side, level_index in face
                )
                outer = tuple(
                    node_id(station_index + side, wall_radii_m[radial_index + 1], levels[side][level_index])
                    for side, level_index in face
                )
                if len(face) == 3:
                    cells.append((6, inner + outer))
                else:
                    cells.append((5, (
                        inner[0], inner[1], outer[1], outer[0],
                        inner[3], inner[2], outer[2], outer[3],
                    )))

        if min(wedge_heights_m[station_index:station_index + 2]) < element_size_m - 1.0e-9:
            continue
        for radial_index in range(wedge_divisions):
            for vertical_index in range(radial_index + 1):
                triangles = [
                    ((radial_index, vertical_index), (radial_index + 1, vertical_index), (radial_index + 1, vertical_index + 1)),
                ]
                if vertical_index < radial_index:
                    triangles.append(((radial_index, vertical_index), (radial_index + 1, vertical_index + 1), (radial_index, vertical_index + 1)))
                for triangle in triangles:
                    section_faces = []
                    for side in range(2):
                        height_m = wedge_heights_m[station_index + side]
                        base_z_m = base_levels_m[station_index + side]
                        section_faces.append(tuple(
                            node_id(
                                station_index + side,
                                DOWNSTREAM_WALL_RADIUS_M - height_m + radial * height_m / wedge_divisions,
                                base_z_m + vertical * height_m / wedge_divisions,
                            )
                            for radial, vertical in triangle
                        ))
                    cells.append((6, section_faces[0] + section_faces[1]))

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
        elif max(abs(station_m - chainages_m[0]) for station_m in stations_m) < tolerance:
            boundary_id = LEFT_END_BOUNDARY_ID
        elif max(abs(station_m - chainages_m[-1]) for station_m in stations_m) < tolerance:
            boundary_id = RIGHT_END_BOUNDARY_ID
        elif all(abs(z_m - _monotone_values(contours, "plinth_z_m", station_m)) < tolerance for station_m, z_m in zip(stations_m, z_values_m)):
            boundary_id = BASE_BOUNDARY_ID
        else:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return LeftWedgedWallTransitionMesh(
        nodes, cells, boundaries, chainages_m, intersection_chainage_m,
        wedge_top_z_m, element_size_m,
    )


def build_left_wedged_wall_transition(root: Path) -> LeftWedgedWallTransitionMesh:
    contours = load_contours(root / "Data" / "plinth.json")
    full_wedge_base_z_m = _monotone_values(contours, "plinth_z_m", FULL_WEDGE_CHAINAGE_M)
    intersection_chainage_m = _intersection_chainage(contours, full_wedge_base_z_m + FULL_WEDGE_HEIGHT_M)
    return build_wedged_wall_transition(root, FULL_WEDGE_CHAINAGE_M, intersection_chainage_m)


def audit_left_wedged_wall_transition(mesh: LeftWedgedWallTransitionMesh) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    boundary_counts = Counter(boundary_id for boundary_id, _ in mesh.boundaries)
    if set(boundary_counts) != set(range(1, 7)):
        raise ValueError(f"Missing transition boundaries: found {sorted(boundary_counts)}")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "boundary_faces": dict(sorted(boundary_counts.items())),
        "chainage_m": [mesh.intersection_chainage_m, FULL_WEDGE_CHAINAGE_M],
        "wall_radius_m": [DOWNSTREAM_WALL_RADIUS_M, UPSTREAM_RADIUS_M],
        "wall_thickness_m": UPSTREAM_RADIUS_M - DOWNSTREAM_WALL_RADIUS_M,
        "transition_length_m": FULL_WEDGE_CHAINAGE_M - mesh.intersection_chainage_m,
        "wedge_removed": True,
        "element_size_m": mesh.element_size_m,
    }


__all__ = [
    "audit_left_wedged_wall_transition",
    "build_left_wedged_wall_transition",
    "write_gmsh",
    "write_vtu",
]
