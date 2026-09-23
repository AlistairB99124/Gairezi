"""Shared construction for wall regions following the plinth contour."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

from regions.plinth import _monotone_values, load_contours, load_global_element_size


DOWNSTREAM_WALL_RADIUS_M = 76.0
UPSTREAM_RADIUS_M = 80.0
CREST_Z_M = 0.0
DOWNSTREAM_BATTER_HEIGHT_M = 4.0
DOWNSTREAM_BATTER_RUN_M = 2.0
DOWNSTREAM_BATTER_TOP_Z_M = -24.5
DOWNSTREAM_BATTER_BASE_Z_M = -28.5
BASE_BOUNDARY_ID = 1
UPSTREAM_BOUNDARY_ID = 2
DOWNSTREAM_BOUNDARY_ID = 3
CREST_BOUNDARY_ID = 4
LEFT_END_BOUNDARY_ID = 5
RIGHT_END_BOUNDARY_ID = 6


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


def section_faces(left_levels: list[float], right_levels: list[float]) -> list[tuple[tuple[int, int], ...]]:
    faces: list[tuple[tuple[int, int], ...]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left_levels) - 1 or right_index < len(right_levels) - 1:
        next_left = left_levels[left_index + 1] if left_index + 1 < len(left_levels) else math.inf
        next_right = right_levels[right_index + 1] if right_index + 1 < len(right_levels) else math.inf
        if abs(next_left - next_right) < 1.0e-9:
            faces.append(((0, left_index), (1, right_index), (1, right_index + 1), (0, left_index + 1)))
            left_index += 1
            right_index += 1
        elif next_left < next_right:
            faces.append(((0, left_index), (1, right_index), (0, left_index + 1)))
            left_index += 1
        else:
            faces.append(((0, left_index), (1, right_index), (1, right_index + 1)))
            right_index += 1
    return faces


def wall_levels(base_z_m: float, element_size_m: float) -> list[float]:
    if CREST_Z_M - base_z_m <= 1.0e-9:
        return [CREST_Z_M]
    first_global_index = math.floor(base_z_m / element_size_m) + 1
    levels = [base_z_m]
    levels.extend(
        index * element_size_m
        for index in range(first_global_index, 1)
        if base_z_m + 0.5 * element_size_m <= index * element_size_m < CREST_Z_M - 1.0e-9
    )
    levels.append(CREST_Z_M)
    return list(dict.fromkeys(levels))


def snapped_wall_base(base_z_m: float, element_size_m: float) -> float:
    return math.ceil((base_z_m - 1.0e-9) / element_size_m) * element_size_m


def downstream_radius(z_m: float) -> float:
    batter_fraction = min(max((DOWNSTREAM_BATTER_TOP_Z_M - z_m) / DOWNSTREAM_BATTER_HEIGHT_M, 0.0), 1.0)
    return DOWNSTREAM_WALL_RADIUS_M - DOWNSTREAM_BATTER_RUN_M * batter_fraction


def build_uniform_wall(
    root: Path,
    start_chainage_m: float,
    end_chainage_m: float,
    anchor_chainages_m: list[float],
) -> UniformWallMesh:
    element_size_m = load_global_element_size(root / "Data" / "Computational_Grid_Controls.json")
    contours = load_contours(root / "Data" / "plinth.json")
    config = json.loads((root / "config.json").read_text())
    centerline_radius_m = float(config["wall_centerline_radius_m"])
    chainages_m = target_chainages(start_chainage_m, end_chainage_m, anchor_chainages_m, element_size_m)
    base_levels_m = [
        snapped_wall_base(_monotone_values(contours, "plinth_z_m", value), element_size_m)
        for value in chainages_m
    ]
    maximum_wall_width_m = UPSTREAM_RADIUS_M - downstream_radius(DOWNSTREAM_BATTER_BASE_Z_M)
    radial_divisions = math.ceil(maximum_wall_width_m / element_size_m)
    section_levels_m = [wall_levels(base_z_m, element_size_m) for base_z_m in base_levels_m]

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

    def wall_radius(z_m: float, radial_index: int) -> float:
        fraction = radial_index / radial_divisions
        downstream_radius_m = downstream_radius(z_m)
        return downstream_radius_m + fraction * (UPSTREAM_RADIUS_M - downstream_radius_m)

    cells: list[tuple[int, tuple[int, ...]]] = []
    for station_index in range(len(chainages_m) - 1):
        levels = (section_levels_m[station_index], section_levels_m[station_index + 1])
        first_course_z_m = max(levels[0][0], levels[1][0])
        course_levels_m = [
            z_m for z_m in levels[0]
            if first_course_z_m - 1.0e-9 <= z_m <= CREST_Z_M
        ]
        if not course_levels_m or abs(course_levels_m[0] - first_course_z_m) > 1.0e-9:
            raise ValueError("Wall base staircase must begin on a global Z course")
        for lower_z_m, upper_z_m in zip(course_levels_m, course_levels_m[1:]):
            for radial_index in range(radial_divisions):
                lower_inner = tuple(
                    node_id(
                        station_index + side,
                        wall_radius(lower_z_m, radial_index),
                        lower_z_m,
                    )
                    for side in range(2)
                )
                lower_outer = tuple(
                    node_id(
                        station_index + side,
                        wall_radius(lower_z_m, radial_index + 1),
                        lower_z_m,
                    )
                    for side in range(2)
                )
                upper_inner = tuple(
                    node_id(
                        station_index + side,
                        wall_radius(upper_z_m, radial_index),
                        upper_z_m,
                    )
                    for side in range(2)
                )
                upper_outer = tuple(
                    node_id(
                        station_index + side,
                        wall_radius(upper_z_m, radial_index + 1),
                        upper_z_m,
                    )
                    for side in range(2)
                )
                cells.append((5, (
                    lower_inner[0], lower_inner[1], lower_outer[1], lower_outer[0],
                    upper_inner[0], upper_inner[1], upper_outer[1], upper_outer[0],
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
        elif max(abs(station_m - end_chainage_m) for station_m in stations_m) < tolerance:
            boundary_id = RIGHT_END_BOUNDARY_ID
        elif max(z_values_m) - min(z_values_m) < tolerance and max(z_values_m) < CREST_Z_M - tolerance:
            boundary_id = BASE_BOUNDARY_ID
        else:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return UniformWallMesh(nodes, cells, boundaries, chainages_m, base_levels_m, element_size_m)
