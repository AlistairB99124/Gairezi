"""Curved constant-thickness wall seated directly on the surveyed bedrock.

The wall runs from the rock surface up to the crest (z = 0) between the
chainages where the rock profile reaches the crest. Its underside follows the
piecewise-linear bedrock contour so that the rock mesh can conform to it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

from .bedrock_contours import CREST_Z_M, bedrock_elevation, load_global_element_size, load_model_contours
from .model_groups import FACE_GROUP_IDS
from ..model_config import load_config


DOWNSTREAM_WALL_RADIUS_M = float(load_config()["geometry"]["wall"]["downstream_radius_m"])
UPSTREAM_RADIUS_M = float(load_config()["geometry"]["wall"]["upstream_radius_m"])
BASE_BOUNDARY_ID = FACE_GROUP_IDS["FOUNDATION"]
UPSTREAM_BOUNDARY_ID = FACE_GROUP_IDS["UPSTREAM"]
DOWNSTREAM_BOUNDARY_ID = FACE_GROUP_IDS["DOWNSTREAM"]
CREST_BOUNDARY_ID = FACE_GROUP_IDS["CREST"]
LEFT_END_BOUNDARY_ID = FACE_GROUP_IDS["LEFT_ABUTMENT"]
RIGHT_END_BOUNDARY_ID = FACE_GROUP_IDS["RIGHT_ABUTMENT"]
OTHER_BOUNDARY_ID = FACE_GROUP_IDS["OTHER_EXTERIOR"]


@dataclass
class UniformWallMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    chainages_m: list[float]
    base_levels_m: list[float]
    element_size_m: float


def target_chainages(start_m: float, end_m: float, anchors_m: list[float], target_size_m: float) -> list[float]:
    anchors = [start_m] + [value for value in anchors_m if start_m < value < end_m] + [end_m]
    values = [start_m]
    for interval_start_m, interval_end_m in zip(anchors, anchors[1:]):
        segment_count = max(1, round((interval_end_m - interval_start_m) / target_size_m))
        values.extend(
            interval_start_m + (interval_end_m - interval_start_m) * index / segment_count
            for index in range(1, segment_count + 1)
        )
    return values


def vertical_levels(bottom_z_m: float, top_z_m: float, element_size_m: float) -> list[float]:
    """Seat, global grid courses at least half an element clear of both ends, and crest."""
    if top_z_m - bottom_z_m <= 1.0e-9:
        return [round(top_z_m, 9)]
    clearance_m = 0.5 * element_size_m
    levels = [bottom_z_m]
    first = math.ceil((bottom_z_m + clearance_m) / element_size_m)
    last = math.floor((top_z_m - clearance_m) / element_size_m)
    levels.extend(index * element_size_m for index in range(first, last + 1))
    levels.append(top_z_m)
    return sorted(set(round(level, 9) for level in levels))


def _triangulate_fan(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    return [
        (vertices[0], vertices[index], vertices[index + 1])
        for index in range(1, len(vertices) - 1)
        if len({side for side, _ in (vertices[0], vertices[index], vertices[index + 1])}) > 1
    ]


def top_aligned_section_faces(
    left_levels: list[float],
    right_levels: list[float],
) -> list[tuple[tuple[int, int], ...]]:
    """Quads between matching crest-down courses; triangles take up extra seat levels."""
    left_extra = max(len(left_levels) - len(right_levels), 0)
    right_extra = max(len(right_levels) - len(left_levels), 0)
    faces = _triangulate_fan(
        [(0, index) for index in range(left_extra + 1)]
        + [(1, index) for index in range(right_extra, -1, -1)]
    )
    faces.extend(
        (
            (0, left_extra + index),
            (1, right_extra + index),
            (1, right_extra + index + 1),
            (0, left_extra + index + 1),
        )
        for index in range(min(len(left_levels), len(right_levels)) - 1)
    )
    return faces


def _determinant(a, b, c) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _edge(nodes, cell, start: int, end: int) -> tuple[float, float, float]:
    first, second = nodes[cell[start] - 1], nodes[cell[end] - 1]
    return tuple(second[index] - first[index] for index in range(3))


def _prism_orientation(nodes: list[tuple[float, float, float]], cell: tuple[int, ...]) -> float:
    return _determinant(_edge(nodes, cell, 0, 1), _edge(nodes, cell, 0, 2), _edge(nodes, cell, 0, 3))


def _hex_orientation(nodes: list[tuple[float, float, float]], cell: tuple[int, ...]) -> float:
    return _determinant(_edge(nodes, cell, 0, 1), _edge(nodes, cell, 0, 3), _edge(nodes, cell, 0, 4))


def build_uniform_wall(
    root: Path,
    start_chainage_m: float,
    end_chainage_m: float,
    anchor_chainages_m: list[float],
    extra_radial_fractions: tuple[float, ...] = (),
) -> UniformWallMesh:
    element_size_m = load_global_element_size(root / "config.json")
    geometry = load_config(root / "config.json")["geometry"]
    bedrock_rows = geometry["bedrock"]["contours"]
    wall = geometry["wall"]
    centerline_radius_m = float(wall["centerline_radius_m"])
    upstream_radius_m = float(wall["upstream_radius_m"])
    downstream_radius_m = float(wall["downstream_radius_m"])
    chainages_m = target_chainages(start_chainage_m, end_chainage_m, anchor_chainages_m, element_size_m)
    base_levels_m = [min(bedrock_elevation(bedrock_rows, value), CREST_Z_M) for value in chainages_m]
    radial_divisions = math.ceil((upstream_radius_m - downstream_radius_m) / element_size_m)
    radial_fractions = sorted({*(index / radial_divisions for index in range(radial_divisions + 1)), *extra_radial_fractions})
    if radial_fractions[0] != 0.0 or radial_fractions[-1] != 1.0:
        raise ValueError("Extra radial fractions must lie within the wall thickness")
    radii_m = [downstream_radius_m + fraction * (upstream_radius_m - downstream_radius_m) for fraction in radial_fractions]
    section_levels_m = [vertical_levels(base_z_m, CREST_Z_M, element_size_m) for base_z_m in base_levels_m]

    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[int, int, int], int] = {}
    for station_index, (chainage_m, levels_m) in enumerate(zip(chainages_m, section_levels_m)):
        angle = chainage_m / centerline_radius_m
        for vertical_index, z_m in enumerate(levels_m):
            for radial_index, radius_m in enumerate(radii_m):
                node_ids[(station_index, vertical_index, radial_index)] = len(nodes) + 1
                nodes.append((radius_m * math.sin(angle), radius_m * math.cos(angle), z_m))

    cells: list[tuple[int, tuple[int, ...]]] = []
    for station_index in range(len(chainages_m) - 1):
        faces = top_aligned_section_faces(section_levels_m[station_index], section_levels_m[station_index + 1])
        for radial_index in range(len(radii_m) - 1):
            for face in faces:
                inner = tuple(node_ids[(station_index + side, level, radial_index)] for side, level in face)
                outer = tuple(node_ids[(station_index + side, level, radial_index + 1)] for side, level in face)
                cell = inner + outer
                if len(face) == 3:
                    if _prism_orientation(nodes, cell) < 0.0:
                        cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
                    cells.append((6, cell))
                else:
                    if _hex_orientation(nodes, cell) < 0.0:
                        cell = (cell[0], cell[3], cell[2], cell[1], cell[4], cell[7], cell[6], cell[5])
                    cells.append((5, cell))

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
        radii = [math.hypot(x_m, y_m) for x_m, y_m, _ in coordinates]
        stations_m = [centerline_radius_m * math.atan2(x_m, y_m) for x_m, y_m, _ in coordinates]
        z_values_m = [z_m for _, _, z_m in coordinates]
        if max(abs(radius_m - upstream_radius_m) for radius_m in radii) < tolerance:
            boundary_id = UPSTREAM_BOUNDARY_ID
        elif max(abs(radius_m - downstream_radius_m) for radius_m in radii) < tolerance:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        elif max(abs(z_m - CREST_Z_M) for z_m in z_values_m) < tolerance:
            boundary_id = CREST_BOUNDARY_ID
        elif all(
            abs(z_m - bedrock_elevation(bedrock_rows, station_m)) < tolerance
            for station_m, z_m in zip(stations_m, z_values_m)
        ):
            boundary_id = BASE_BOUNDARY_ID
        elif max(abs(station_m - start_chainage_m) for station_m in stations_m) < tolerance:
            boundary_id = LEFT_END_BOUNDARY_ID
        elif max(abs(station_m - end_chainage_m) for station_m in stations_m) < tolerance:
            boundary_id = RIGHT_END_BOUNDARY_ID
        else:
            boundary_id = OTHER_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return UniformWallMesh(nodes, cells, boundaries, chainages_m, base_levels_m, element_size_m)


def build_bedrock_seated_wall(root: Path, extra_radial_fractions: tuple[float, ...] = ()) -> UniformWallMesh:
    contours = load_model_contours(root)
    return build_uniform_wall(
        root,
        contours[0].chainage_m,
        contours[-1].chainage_m,
        [point.chainage_m for point in contours],
        extra_radial_fractions,
    )
