"""Build smooth contour-following reinforcement caps over the stepped plinth."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

from regions.plinth import _coordinate_key, _hex_orientation, _prism_orientation


BODY_ID = 3
BASE_BOUNDARY_ID = 1
UPSTREAM_BOUNDARY_ID = 2
DOWNSTREAM_BOUNDARY_ID = 3
CREST_BOUNDARY_ID = 4
LEFT_END_BOUNDARY_ID = 5
RIGHT_END_BOUNDARY_ID = 6
OTHER_BOUNDARY_ID = 7


@dataclass
class PlinthHaunchMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    element_size_m: float


def _face_edges(face: tuple[int, ...]) -> list[tuple[int, int]]:
    return [
        tuple(sorted((face[index], face[(index + 1) % len(face)])))
        for index in range(len(face))
    ]


def build_plinth_haunch(root: Path, wall, plinth) -> PlinthHaunchMesh:
    tolerance = 1.0e-8
    cap_height_m = wall.element_size_m
    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[float, float, float], int] = {}

    def node_id(point: tuple[float, float, float]) -> int:
        key = _coordinate_key(point)
        if key not in node_ids:
            node_ids[key] = len(nodes) + 1
            nodes.append(point)
        return node_ids[key]

    wall_base_edges = {
        tuple(sorted((_coordinate_key(wall.nodes[first - 1]), _coordinate_key(wall.nodes[second - 1]))))
        for boundary_id, face in wall.boundaries
        if boundary_id == BASE_BOUNDARY_ID
        for first, second in _face_edges(face)
    }
    station_levels: list[tuple[float, float]] = []
    for chainage_m, base_z_m in zip(wall.chainages_m, wall.base_levels_m):
        if station_levels and abs(chainage_m - station_levels[-1][0]) < tolerance:
            continue
        station_levels.append((chainage_m, base_z_m))
    roof_z_by_angle: dict[float, float] = {}
    platform_start = 0
    while platform_start < len(station_levels):
        platform_end = platform_start
        base_z_m = station_levels[platform_start][1]
        while (
            platform_end + 1 < len(station_levels)
            and abs(station_levels[platform_end + 1][1] - base_z_m) < tolerance
        ):
            platform_end += 1
        higher_left = (
            platform_start > 0
            and abs(station_levels[platform_start - 1][1] - (base_z_m + cap_height_m)) < tolerance
        )
        higher_right = (
            platform_end + 1 < len(station_levels)
            and abs(station_levels[platform_end + 1][1] - (base_z_m + cap_height_m)) < tolerance
        )
        if higher_left or higher_right:
            midpoint = 0.5 * (platform_start + platform_end)
            for index in range(platform_start, platform_end + 1):
                if higher_left and higher_right:
                    height_fraction = abs(index - midpoint) / max(midpoint - platform_start, 1.0)
                elif higher_left:
                    height_fraction = (platform_end - index) / max(platform_end - platform_start, 1)
                else:
                    height_fraction = (index - platform_start) / max(platform_end - platform_start, 1)
                chainage_m = station_levels[index][0]
                roof_z_by_angle[round(chainage_m / 78.0, 9)] = base_z_m + cap_height_m * height_fraction
        platform_start = platform_end + 1

    shelf_faces: dict[tuple[tuple[float, float, float], tuple[float, float, float]], list[tuple[float, float, float]]] = {}
    for _, face in plinth.boundaries:
        if len(face) != 4:
            continue
        points = [plinth.nodes[node_id_value - 1] for node_id_value in face]
        if max(point[2] for point in points) - min(point[2] for point in points) > tolerance:
            continue
        for first, second in zip(points, points[1:] + points[:1]):
            edge_key = tuple(sorted((_coordinate_key(first), _coordinate_key(second))))
            shelf_faces.setdefault(edge_key, []).append(points)

    cells: list[tuple[int, tuple[int, ...]]] = []
    wall_face_keys: set[tuple[int, ...]] = set()
    plinth_face_keys: set[tuple[int, ...]] = set()
    exposed_face_ids: dict[tuple[int, ...], int] = {}

    for wall_boundary_id, face in wall.boundaries:
        if wall_boundary_id not in (UPSTREAM_BOUNDARY_ID, DOWNSTREAM_BOUNDARY_ID) or len(face) != 4:
            continue
        wall_points = [wall.nodes[node_id_value - 1] for node_id_value in face]
        base_edge = next(
            (
                (first, second)
                for first, second in zip(wall_points, wall_points[1:] + wall_points[:1])
                if tuple(sorted((_coordinate_key(first), _coordinate_key(second)))) in wall_base_edges
            ),
            None,
        )
        if base_edge is None:
            continue
        base_start, base_end = base_edge
        base_z_m = base_start[2]
        if abs(base_end[2] - base_z_m) > tolerance:
            continue
        edge_key = tuple(sorted((_coordinate_key(base_start), _coordinate_key(base_end))))
        candidates = shelf_faces.get(edge_key, [])
        if not candidates:
            continue

        for shelf_points in candidates:
            outer_points = [
                point for point in shelf_points
                if _coordinate_key(point) not in edge_key
            ]
            if len(outer_points) != 2:
                continue
            radii = [math.hypot(point[0], point[1]) for point in (base_start, base_end)]
            outer_radii = [math.hypot(point[0], point[1]) for point in outer_points]
            if wall_boundary_id == DOWNSTREAM_BOUNDARY_ID:
                if not max(outer_radii) < min(radii) - tolerance:
                    continue
            elif not min(outer_radii) > max(radii) + tolerance:
                continue

            outer_by_chainage = {}
            for point in outer_points:
                angle = math.atan2(point[0], point[1])
                outer_by_chainage[round(angle, 9)] = point
            base_by_chainage = {
                round(math.atan2(point[0], point[1]), 9): point
                for point in (base_start, base_end)
            }
            if set(base_by_chainage) != set(outer_by_chainage):
                continue
            ordered_angles = sorted(base_by_chainage)
            inner_start, inner_end = (base_by_chainage[angle] for angle in ordered_angles)
            outer_start, outer_end = (outer_by_chainage[angle] for angle in ordered_angles)
            roof_start_z_m = roof_z_by_angle.get(ordered_angles[0])
            roof_end_z_m = roof_z_by_angle.get(ordered_angles[1])
            if roof_start_z_m is None or roof_end_z_m is None:
                continue
            roof_start = (
                inner_start[0],
                inner_start[1],
                roof_start_z_m,
            )
            roof_end = (
                inner_end[0],
                inner_end[1],
                roof_end_z_m,
            )
            roof_outer_start = (outer_start[0], outer_start[1], roof_start[2])
            roof_outer_end = (outer_end[0], outer_end[1], roof_end[2])
            start_collapsed = abs(roof_start_z_m - base_z_m) < tolerance
            end_collapsed = abs(roof_end_z_m - base_z_m) < tolerance
            if start_collapsed and end_collapsed:
                continue
            if start_collapsed:
                cell = (
                    node_id(inner_start), node_id(inner_end), node_id(roof_end),
                    node_id(outer_start), node_id(outer_end), node_id(roof_outer_end),
                )
                if _prism_orientation(nodes, cell) < 0.0:
                    cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
                cells.append((6, cell))
                plinth_face_keys.add(tuple(sorted((cell[0], cell[3], cell[4], cell[1]))))
                exposed_face_ids[tuple(sorted((cell[1], cell[4], cell[5], cell[2])))] = wall_boundary_id
            elif end_collapsed:
                cell = (
                    node_id(inner_end), node_id(inner_start), node_id(roof_start),
                    node_id(outer_end), node_id(outer_start), node_id(roof_outer_start),
                )
                if _prism_orientation(nodes, cell) < 0.0:
                    cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
                cells.append((6, cell))
                plinth_face_keys.add(tuple(sorted((cell[0], cell[3], cell[4], cell[1]))))
                exposed_face_ids[tuple(sorted((cell[1], cell[4], cell[5], cell[2])))] = wall_boundary_id
            else:
                cell = (
                    node_id(inner_start), node_id(inner_end), node_id(outer_end), node_id(outer_start),
                    node_id(roof_start), node_id(roof_end), node_id(roof_outer_end), node_id(roof_outer_start),
                )
                if _hex_orientation(nodes, cell) < 0.0:
                    cell = (cell[0], cell[3], cell[2], cell[1], cell[4], cell[7], cell[6], cell[5])
                cells.append((5, cell))
                plinth_face_keys.add(tuple(sorted(cell[:4])))
                exposed_face_ids[tuple(sorted((cell[4], cell[7], cell[6], cell[5])))] = wall_boundary_id
                exposed_face_ids[tuple(sorted((cell[2], cell[6], cell[7], cell[3])))] = wall_boundary_id

    if not cells:
        raise ValueError("No plinth haunch prisms could be matched to wall and plinth faces")

    face_patterns = {
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    }
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for element_type, cell in cells:
        for pattern in face_patterns[element_type]:
            face = tuple(cell[index] for index in pattern)
            face_key = tuple(sorted(face))
            face_counts[face_key] += 1
            oriented_faces[face_key] = face

    boundaries = []
    for face_key, count in face_counts.items():
        if count != 1:
            continue
        face = oriented_faces[face_key]
        if face_key in wall_face_keys or face_key in plinth_face_keys:
            boundary_id = OTHER_BOUNDARY_ID
        elif face_key in exposed_face_ids:
            boundary_id = exposed_face_ids[face_key]
        else:
            boundary_id = OTHER_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return PlinthHaunchMesh(nodes, cells, boundaries, wall.element_size_m)


def audit_plinth_haunch(mesh: PlinthHaunchMesh) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    if not set(element_counts) <= {5, 6}:
        raise ValueError(f"Haunch contains unsupported cells: {dict(element_counts)}")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "hexahedra": element_counts[5],
        "triangular_prisms": element_counts[6],
        "boundary_faces": dict(sorted(Counter(boundary_id for boundary_id, _ in mesh.boundaries).items())),
    }
