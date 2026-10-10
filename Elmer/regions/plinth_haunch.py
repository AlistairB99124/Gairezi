"""Build smooth contour-following reinforcement caps over the stepped plinth."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

from .plinth import _coordinate_key, _determinant, _hex_orientation, _prism_orientation
from .model_groups import FACE_GROUP_IDS, REGION_GROUP_IDS
from .uniform_wall import (
    CENTER_STEP_END_CHAINAGE_M,
    CENTER_STEP_START_CHAINAGE_M,
    CENTER_STEP_RADIUS_M,
    CENTER_STEP_TOP_Z_M,
    DOWNSTREAM_BATTER_BASE_Z_M,
    DOWNSTREAM_BATTER_TOP_Z_M,
    DOWNSTREAM_WALL_RADIUS_M,
    UPSTREAM_RADIUS_M,
)


BODY_ID = REGION_GROUP_IDS["HAUNCH"]
BASE_BOUNDARY_ID = FACE_GROUP_IDS["FOUNDATION"]
UPSTREAM_BOUNDARY_ID = FACE_GROUP_IDS["UPSTREAM"]
DOWNSTREAM_BOUNDARY_ID = FACE_GROUP_IDS["DOWNSTREAM"]
CREST_BOUNDARY_ID = FACE_GROUP_IDS["CREST"]
LEFT_END_BOUNDARY_ID = FACE_GROUP_IDS["LEFT_ABUTMENT"]
RIGHT_END_BOUNDARY_ID = FACE_GROUP_IDS["RIGHT_ABUTMENT"]
OTHER_BOUNDARY_ID = FACE_GROUP_IDS["OTHER_EXTERIOR"]
HAUNCH_WALL_MORTAR_THICKNESS_M = 0.1


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
        if not station_levels or abs(chainage_m - station_levels[-1][0]) >= tolerance:
            station_levels.append((chainage_m, base_z_m))
    roof_z_by_angle: dict[float, float] = {}
    platform_start = 0
    while platform_start < len(station_levels):
        platform_end = platform_start
        base_z_m = station_levels[platform_start][1]
        while platform_end + 1 < len(station_levels) and abs(station_levels[platform_end + 1][1] - base_z_m) < tolerance:
            platform_end += 1
        higher_left = platform_start > 0 and abs(station_levels[platform_start - 1][1] - (base_z_m + cap_height_m)) < tolerance
        higher_right = platform_end + 1 < len(station_levels) and abs(station_levels[platform_end + 1][1] - (base_z_m + cap_height_m)) < tolerance
        if higher_left or higher_right:
            midpoint = 0.5 * (platform_start + platform_end)
            for index in range(platform_start, platform_end + 1):
                if higher_left and higher_right:
                    fraction = abs(index - midpoint) / max(midpoint - platform_start, 1.0)
                elif higher_left:
                    fraction = (platform_end - index) / max(platform_end - platform_start, 1)
                else:
                    fraction = (index - platform_start) / max(platform_end - platform_start, 1)
                roof_z_by_angle[round(station_levels[index][0] / 78.0, 9)] = base_z_m + cap_height_m * fraction
        platform_start = platform_end + 1

    shelf_faces: dict[tuple[tuple[float, float, float], tuple[float, float, float]], list[tuple[float, float, float]]] = {}
    for _, face in plinth.boundaries:
        if len(face) != 4:
            continue
        points = [plinth.nodes[node_id_value - 1] for node_id_value in face]
        if max(point[2] for point in points) - min(point[2] for point in points) > tolerance:
            continue
        for first, second in zip(points, points[1:] + points[:1]):
            shelf_faces.setdefault(tuple(sorted((_coordinate_key(first), _coordinate_key(second)))), []).append(points)

    cells: list[tuple[int, tuple[int, ...]]] = []
    plinth_face_keys: set[tuple[int, ...]] = set()
    exposed_face_ids: dict[tuple[int, ...], int] = {}
    for wall_boundary_id, face in wall.boundaries:
        if wall_boundary_id not in (UPSTREAM_BOUNDARY_ID, DOWNSTREAM_BOUNDARY_ID) or len(face) != 4:
            continue
        wall_points = [wall.nodes[node_id_value - 1] for node_id_value in face]
        base_edge = next(((first, second) for first, second in zip(wall_points, wall_points[1:] + wall_points[:1]) if tuple(sorted((_coordinate_key(first), _coordinate_key(second)))) in wall_base_edges), None)
        if base_edge is None or abs(base_edge[0][2] - base_edge[1][2]) > tolerance:
            continue
        base_start, base_end = base_edge
        edge_key = tuple(sorted((_coordinate_key(base_start), _coordinate_key(base_end))))
        for shelf_points in shelf_faces.get(edge_key, []):
            outer_points = [point for point in shelf_points if _coordinate_key(point) not in edge_key]
            if len(outer_points) != 2:
                continue
            radii = [math.hypot(point[0], point[1]) for point in (base_start, base_end)]
            outer_radii = [math.hypot(point[0], point[1]) for point in outer_points]
            if (wall_boundary_id == DOWNSTREAM_BOUNDARY_ID and not max(outer_radii) < min(radii) - tolerance) or (wall_boundary_id == UPSTREAM_BOUNDARY_ID and not min(outer_radii) > max(radii) + tolerance):
                continue
            outer_by_angle = {round(math.atan2(point[0], point[1]), 9): point for point in outer_points}
            base_by_angle = {round(math.atan2(point[0], point[1]), 9): point for point in (base_start, base_end)}
            if set(base_by_angle) != set(outer_by_angle):
                continue
            start_angle, end_angle = sorted(base_by_angle)
            inner_start, inner_end = base_by_angle[start_angle], base_by_angle[end_angle]
            outer_start, outer_end = outer_by_angle[start_angle], outer_by_angle[end_angle]
            roof_start_z_m, roof_end_z_m = roof_z_by_angle.get(start_angle), roof_z_by_angle.get(end_angle)
            if roof_start_z_m is None or roof_end_z_m is None:
                continue
            roof_start = (inner_start[0], inner_start[1], roof_start_z_m)
            roof_end = (inner_end[0], inner_end[1], roof_end_z_m)
            roof_outer_start, roof_outer_end = (outer_start[0], outer_start[1], roof_start_z_m), (outer_end[0], outer_end[1], roof_end_z_m)
            if abs(roof_start_z_m - base_start[2]) < tolerance and abs(roof_end_z_m - base_start[2]) < tolerance:
                continue
            if abs(roof_start_z_m - base_start[2]) < tolerance:
                cell = (node_id(inner_start), node_id(inner_end), node_id(roof_end), node_id(outer_start), node_id(outer_end), node_id(roof_outer_end))
                if _prism_orientation(nodes, cell) < 0.0:
                    cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
                cells.append((6, cell))
                plinth_face_keys.add(tuple(sorted((cell[0], cell[3], cell[4], cell[1]))))
                exposed_face_ids[tuple(sorted((cell[1], cell[4], cell[5], cell[2])))] = wall_boundary_id
            elif abs(roof_end_z_m - base_start[2]) < tolerance:
                cell = (node_id(inner_end), node_id(inner_start), node_id(roof_start), node_id(outer_end), node_id(outer_start), node_id(roof_outer_start))
                if _prism_orientation(nodes, cell) < 0.0:
                    cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
                cells.append((6, cell))
                plinth_face_keys.add(tuple(sorted((cell[0], cell[3], cell[4], cell[1]))))
                exposed_face_ids[tuple(sorted((cell[1], cell[4], cell[5], cell[2])))] = wall_boundary_id
            else:
                cell = (node_id(inner_start), node_id(inner_end), node_id(outer_end), node_id(outer_start), node_id(roof_start), node_id(roof_end), node_id(roof_outer_end), node_id(roof_outer_start))
                if _hex_orientation(nodes, cell) < 0.0:
                    cell = (cell[0], cell[3], cell[2], cell[1], cell[4], cell[7], cell[6], cell[5])
                cells.append((5, cell))
                plinth_face_keys.add(tuple(sorted(cell[:4])))
                exposed_face_ids[tuple(sorted((cell[4], cell[7], cell[6], cell[5])))] = wall_boundary_id
                exposed_face_ids[tuple(sorted((cell[2], cell[6], cell[7], cell[3])))] = wall_boundary_id
    if not cells:
        raise ValueError("No plinth haunch prisms could be matched to wall and plinth faces")
    return PlinthHaunchMesh(nodes, cells, [], wall.element_size_m)


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


def build_center_landing_haunches(root: Path, wall) -> PlinthHaunchMesh:
    """Fill the centre downstream landing with 0.5 m hexahedral courses."""
    centerline_radius_m = 78.0
    chainages_m = [
        chainage_m
        for chainage_m in wall.chainages_m
        if CENTER_STEP_START_CHAINAGE_M - 1.0e-9 <= chainage_m <= CENTER_STEP_END_CHAINAGE_M + 1.0e-9
    ]
    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[float, float, float], int] = {}

    def node_id(chainage_m: float, radius_m: float, z_m: float) -> int:
        angle = chainage_m / centerline_radius_m
        point = (radius_m * math.sin(angle), radius_m * math.cos(angle), z_m)
        key = _coordinate_key(point)
        if key not in node_ids:
            node_ids[key] = len(nodes) + 1
            nodes.append(point)
        return node_ids[key]

    cells: list[tuple[int, tuple[int, ...]]] = []
    lower_z_m = DOWNSTREAM_BATTER_BASE_Z_M
    upper_z_m = CENTER_STEP_TOP_Z_M
    for start_m, end_m in zip(chainages_m, chainages_m[1:]):
        for course_lower_z_m, course_upper_z_m in (
            (lower_z_m, lower_z_m + 0.5),
            (lower_z_m + 0.5, upper_z_m),
        ):
            cell = (
                node_id(start_m, 74.0, course_lower_z_m),
                node_id(end_m, 74.0, course_lower_z_m),
                node_id(end_m, CENTER_STEP_RADIUS_M, course_lower_z_m),
                node_id(start_m, CENTER_STEP_RADIUS_M, course_lower_z_m),
                node_id(start_m, 74.0, course_upper_z_m),
                node_id(end_m, 74.0, course_upper_z_m),
                node_id(end_m, CENTER_STEP_RADIUS_M, course_upper_z_m),
                node_id(start_m, CENTER_STEP_RADIUS_M, course_upper_z_m),
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
    for key, count in face_counts.items():
        if count != 1:
            continue
        face = oriented_faces[key]
        points = [nodes[node_id_value - 1] for node_id_value in face]
        radii_m = [math.hypot(x_m, y_m) for x_m, y_m, _ in points]
        if max(abs(radius_m - CENTER_STEP_RADIUS_M) for radius_m in radii_m) < 1.0e-6:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        else:
            boundary_id = OTHER_BOUNDARY_ID
        boundaries.append((boundary_id, face))
    return PlinthHaunchMesh(nodes, cells, boundaries, wall.element_size_m)


def build_center_haunch_mortar(root: Path, wall) -> PlinthHaunchMesh:
    """Fill the vertical wall-centre-haunch joint without separating its plinth base."""
    centerline_radius_m = 78.0
    chainages_m = [
        chainage_m
        for chainage_m in wall.chainages_m
        if CENTER_STEP_START_CHAINAGE_M - 1.0e-9 <= chainage_m <= CENTER_STEP_END_CHAINAGE_M + 1.0e-9
    ]
    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[float, float, float], int] = {}

    def node_id(chainage_m: float, radius_m: float, z_m: float) -> int:
        angle = chainage_m / centerline_radius_m
        point = (radius_m * math.sin(angle), radius_m * math.cos(angle), z_m)
        key = _coordinate_key(point)
        if key not in node_ids:
            node_ids[key] = len(nodes) + 1
            nodes.append(point)
        return node_ids[key]

    cells: list[tuple[int, tuple[int, ...]]] = []
    lower_z_m = DOWNSTREAM_BATTER_BASE_Z_M
    middle_z_m = lower_z_m + 0.5
    upper_z_m = CENTER_STEP_TOP_Z_M
    mortar_radius_m = CENTER_STEP_RADIUS_M - HAUNCH_WALL_MORTAR_THICKNESS_M
    for start_m, end_m in zip(chainages_m, chainages_m[1:]):
        wall_lower_start = node_id(start_m, CENTER_STEP_RADIUS_M, lower_z_m)
        wall_lower_end = node_id(end_m, CENTER_STEP_RADIUS_M, lower_z_m)
        wall_middle_start = node_id(start_m, CENTER_STEP_RADIUS_M, middle_z_m)
        wall_middle_end = node_id(end_m, CENTER_STEP_RADIUS_M, middle_z_m)
        wall_upper_start = node_id(start_m, CENTER_STEP_RADIUS_M, upper_z_m)
        wall_upper_end = node_id(end_m, CENTER_STEP_RADIUS_M, upper_z_m)
        haunch_middle_start = node_id(start_m, mortar_radius_m, middle_z_m)
        haunch_middle_end = node_id(end_m, mortar_radius_m, middle_z_m)
        haunch_upper_start = node_id(start_m, mortar_radius_m, upper_z_m)
        haunch_upper_end = node_id(end_m, mortar_radius_m, upper_z_m)

        lower_cell = (
            wall_lower_start, wall_middle_start, haunch_middle_start,
            wall_lower_end, wall_middle_end, haunch_middle_end,
        )
        if _prism_orientation(nodes, lower_cell) < 0.0:
            lower_cell = (lower_cell[0], lower_cell[2], lower_cell[1], lower_cell[3], lower_cell[5], lower_cell[4])
        cells.append((6, lower_cell))

        upper_cell = (
            wall_middle_start, wall_middle_end, wall_upper_end, wall_upper_start,
            haunch_middle_start, haunch_middle_end, haunch_upper_end, haunch_upper_start,
        )
        if _hex_orientation(nodes, upper_cell) < 0.0:
            upper_cell = (upper_cell[0], upper_cell[3], upper_cell[2], upper_cell[1], upper_cell[4], upper_cell[7], upper_cell[6], upper_cell[5])
        cells.append((5, upper_cell))

    face_patterns = {
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    }
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for cell_type, cell in cells:
        for pattern in face_patterns[cell_type]:
            face = tuple(cell[index] for index in pattern)
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face
    boundaries = [
        (DOWNSTREAM_BOUNDARY_ID, oriented_faces[key])
        for key, count in face_counts.items()
        if count == 1
    ]
    return PlinthHaunchMesh(nodes, cells, boundaries, wall.element_size_m)


def build_landing_haunch_mortar(haunch: PlinthHaunchMesh, plinth) -> PlinthHaunchMesh:
    """Insert mortar at each generic haunch-wall face without moving plinth nodes."""
    face_patterns = {
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    }
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for cell_type, cell in haunch.cells:
        for pattern in face_patterns[cell_type]:
            face = tuple(cell[index] for index in pattern)
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face

    plinth_node_keys = {_coordinate_key(point) for point in plinth.nodes}
    mortar_faces: list[tuple[int, ...]] = []
    movable_node_ids: set[int] = set()
    for key, count in face_counts.items():
        if count != 1:
            continue
        face = oriented_faces[key]
        points = [haunch.nodes[node_id - 1] for node_id in face]
        radii_m = [math.hypot(x_m, y_m) for x_m, y_m, _ in points]
        if not (
            max(abs(radius_m - DOWNSTREAM_WALL_RADIUS_M) for radius_m in radii_m) < 1.0e-7
            or max(abs(radius_m - UPSTREAM_RADIUS_M) for radius_m in radii_m) < 1.0e-7
        ):
            continue
        free_node_ids = [
            node_id
            for node_id, point in zip(face, points)
            if _coordinate_key(point) not in plinth_node_keys
        ]
        if len(free_node_ids) != 1:
            continue
        mortar_faces.append(face)
        movable_node_ids.add(free_node_ids[0])

    original_points = {node_id: haunch.nodes[node_id - 1] for node_id in movable_node_ids}
    for node_id, (x_m, y_m, z_m) in original_points.items():
        radius_m = math.hypot(x_m, y_m)
        target_radius_m = (
            radius_m - HAUNCH_WALL_MORTAR_THICKNESS_M
            if abs(radius_m - DOWNSTREAM_WALL_RADIUS_M) < 1.0e-7
            else radius_m + HAUNCH_WALL_MORTAR_THICKNESS_M
        )
        scale = target_radius_m / radius_m
        haunch.nodes[node_id - 1] = (x_m * scale, y_m * scale, z_m)

    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[float, float, float], int] = {}

    def node_id(point: tuple[float, float, float]) -> int:
        key = _coordinate_key(point)
        if key not in node_ids:
            node_ids[key] = len(nodes) + 1
            nodes.append(point)
        return node_ids[key]

    def append_tetrahedron(cell: tuple[int, int, int, int]) -> None:
        points = [nodes[node_id_value - 1] for node_id_value in cell]
        volume_sign = _determinant(
            tuple(points[1][axis] - points[0][axis] for axis in range(3)),
            tuple(points[2][axis] - points[0][axis] for axis in range(3)),
            tuple(points[3][axis] - points[0][axis] for axis in range(3)),
        )
        if abs(volume_sign) < 1.0e-12:
            raise ValueError("Zero-volume tetrahedron in haunch-wall mortar")
        if volume_sign < 0.0:
            cell = (cell[0], cell[2], cell[1], cell[3])
        cells.append((4, cell))

    def append_pyramid(base: tuple[int, int, int, int], apex: int) -> None:
        points = [nodes[node_id_value - 1] for node_id_value in base + (apex,)]
        volume_sign = _determinant(
            tuple(points[1][axis] - points[0][axis] for axis in range(3)),
            tuple(points[2][axis] - points[0][axis] for axis in range(3)),
            tuple(points[4][axis] - points[0][axis] for axis in range(3)),
        ) + _determinant(
            tuple(points[2][axis] - points[0][axis] for axis in range(3)),
            tuple(points[3][axis] - points[0][axis] for axis in range(3)),
            tuple(points[4][axis] - points[0][axis] for axis in range(3)),
        )
        if abs(volume_sign) < 1.0e-12:
            raise ValueError("Zero-volume pyramid in haunch-wall mortar")
        if volume_sign < 0.0:
            base = (base[0], base[3], base[2], base[1])
        cells.append((7, base + (apex,)))

    cells: list[tuple[int, tuple[int, ...]]] = []
    for face in mortar_faces:
        outer_points = [original_points.get(node_id_value, haunch.nodes[node_id_value - 1]) for node_id_value in face]
        inner_points = [haunch.nodes[node_id_value - 1] for node_id_value in face]
        movable_index = next(index for index, node_id_value in enumerate(face) if node_id_value in movable_node_ids)
        outer_node_ids = [node_id(point) for point in outer_points]
        inner_node_ids = [node_id(point) for point in inner_points]
        if len(face) == 3:
            append_tetrahedron((
                outer_node_ids[(movable_index - 1) % 3],
                outer_node_ids[movable_index],
                outer_node_ids[(movable_index + 1) % 3],
                inner_node_ids[movable_index],
            ))
        else:
            append_pyramid(tuple(inner_node_ids), outer_node_ids[movable_index])

    return PlinthHaunchMesh(nodes, cells, [], haunch.element_size_m)


def _approach_landing_segments(wall) -> tuple[tuple[float, float, float], ...]:
    """Return low downstream landing spans on either side of the centre detail."""
    segments = []
    for index, (start_m, end_m) in enumerate(zip(wall.chainages_m, wall.chainages_m[1:])):
        base_z_m = wall.base_levels_m[index]
        if abs(base_z_m - wall.base_levels_m[index + 1]) > 1.0e-9:
            continue
        if base_z_m >= DOWNSTREAM_BATTER_TOP_Z_M - 1.0e-9:
            continue
        if end_m <= CENTER_STEP_START_CHAINAGE_M or start_m >= CENTER_STEP_END_CHAINAGE_M:
            segments.append((start_m, end_m, base_z_m))
    return tuple(segments)


def _downstream_wall_points(wall) -> dict[tuple[float, float], tuple[float, float, float]]:
    centerline_radius_m = 78.0
    points: dict[tuple[float, float], tuple[float, float, float]] = {}
    for boundary_id, face in wall.boundaries:
        if boundary_id != DOWNSTREAM_BOUNDARY_ID:
            continue
        for node_id in face:
            point = wall.nodes[node_id - 1]
            chainage_m = centerline_radius_m * math.atan2(point[0], point[1])
            key = (round(chainage_m, 9), round(point[2], 9))
            if key not in points or math.hypot(point[0], point[1]) < math.hypot(points[key][0], points[key][1]):
                points[key] = point
    return points


def build_approach_landing_haunches(wall) -> PlinthHaunchMesh:
    """Build downstream prisms across the stepped plinth landings beside the centre."""
    centerline_radius_m = 78.0
    wall_points = _downstream_wall_points(wall)
    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[float, float, float], int] = {}

    def node_id(point: tuple[float, float, float]) -> int:
        key = _coordinate_key(point)
        if key not in node_ids:
            node_ids[key] = len(nodes) + 1
            nodes.append(point)
        return node_ids[key]

    def outer_point(chainage_m: float, z_m: float) -> tuple[float, float, float]:
        angle = chainage_m / centerline_radius_m
        return (74.0 * math.sin(angle), 74.0 * math.cos(angle), z_m)

    cells: list[tuple[int, tuple[int, ...]]] = []
    for start_m, end_m, base_z_m in _approach_landing_segments(wall):
        lower_start = wall_points[(round(start_m, 9), round(base_z_m, 9))]
        lower_end = wall_points[(round(end_m, 9), round(base_z_m, 9))]
        upper_start = wall_points[(round(start_m, 9), round(base_z_m + 0.5, 9))]
        upper_end = wall_points[(round(end_m, 9), round(base_z_m + 0.5, 9))]
        cell = (
            node_id(outer_point(start_m, base_z_m)), node_id(lower_start), node_id(upper_start),
            node_id(outer_point(end_m, base_z_m)), node_id(lower_end), node_id(upper_end),
        )
        if _prism_orientation(nodes, cell) < 0.0:
            cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
        cells.append((6, cell))

    face_patterns = ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0))
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for _, cell in cells:
        for pattern in face_patterns:
            face = tuple(cell[index] for index in pattern)
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face
    boundaries = [
        (DOWNSTREAM_BOUNDARY_ID, oriented_faces[key])
        for key, count in face_counts.items()
        if count == 1
    ]
    return PlinthHaunchMesh(nodes, cells, boundaries, wall.element_size_m)


def build_approach_landing_mortar(wall) -> PlinthHaunchMesh:
    """Build horizontal mortar beds below the downstream approach prisms."""
    centerline_radius_m = 78.0
    wall_points = _downstream_wall_points(wall)
    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[float, float, float], int] = {}

    def node_id(point: tuple[float, float, float]) -> int:
        key = _coordinate_key(point)
        if key not in node_ids:
            node_ids[key] = len(nodes) + 1
            nodes.append(point)
        return node_ids[key]

    def outer_point(chainage_m: float, z_m: float) -> tuple[float, float, float]:
        angle = chainage_m / centerline_radius_m
        return (74.0 * math.sin(angle), 74.0 * math.cos(angle), z_m)

    cells: list[tuple[int, tuple[int, ...]]] = []
    foundation_faces: set[tuple[int, ...]] = set()
    for start_m, end_m, base_z_m in _approach_landing_segments(wall):
        lower_start = wall_points[(round(start_m, 9), round(base_z_m, 9))]
        lower_end = wall_points[(round(end_m, 9), round(base_z_m, 9))]
        lower_outer_start = outer_point(start_m, base_z_m - MORTAR_THICKNESS_M)
        lower_outer_end = outer_point(end_m, base_z_m - MORTAR_THICKNESS_M)
        lower_inner_start = (*lower_start[:2], base_z_m - MORTAR_THICKNESS_M)
        lower_inner_end = (*lower_end[:2], base_z_m - MORTAR_THICKNESS_M)
        upper_outer_start = outer_point(start_m, base_z_m)
        upper_outer_end = outer_point(end_m, base_z_m)
        cell = (
            node_id(lower_outer_start), node_id(lower_outer_end), node_id(lower_inner_end), node_id(lower_inner_start),
            node_id(upper_outer_start), node_id(upper_outer_end), node_id(lower_end), node_id(lower_start),
        )
        if _hex_orientation(nodes, cell) < 0.0:
            cell = (cell[0], cell[3], cell[2], cell[1], cell[4], cell[7], cell[6], cell[5])
        cells.append((5, cell))
        foundation_faces.add(tuple(sorted(cell[:4])))

    face_patterns = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for _, cell in cells:
        for pattern in face_patterns:
            face = tuple(cell[index] for index in pattern)
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face
    boundaries = [
        (BASE_BOUNDARY_ID if key in foundation_faces else DOWNSTREAM_BOUNDARY_ID, oriented_faces[key])
        for key, count in face_counts.items()
        if count == 1
    ]
    return PlinthHaunchMesh(nodes, cells, boundaries, wall.element_size_m)


def build_landing_haunches(wall, plinth) -> PlinthHaunchMesh:
    """Build radial triangular prisms that slope from each plinth step onto its landing."""
    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[float, float, float], int] = {}
    cells: list[tuple[int, tuple[int, ...]]] = []
    centerline_radius_m = 78.0

    def exterior_riser_strips(
        chainage_m: float,
        nominal_lower_z_m: float,
        nominal_upper_z_m: float,
    ) -> tuple[tuple[float, float, float, float], ...]:
        strips = []
        for _, face in plinth.boundaries:
            points = [plinth.nodes[node_id - 1] for node_id in face]
            face_chainages_m = [centerline_radius_m * math.atan2(x_m, y_m) for x_m, y_m, _ in points]
            radii_m = [math.hypot(x_m, y_m) for x_m, y_m, _ in points]
            elevations_m = [z_m for _, _, z_m in points]
            if max(face_chainages_m) - min(face_chainages_m) > 1.0e-7:
                continue
            if abs(face_chainages_m[0] - chainage_m) > 1.0e-7:
                continue
            if max(elevations_m) - min(elevations_m) < wall.element_size_m - 1.0e-9:
                continue
            if abs(max(elevations_m) - nominal_upper_z_m) > 1.0e-7:
                continue
            if abs(min(elevations_m) - nominal_lower_z_m) > 1.0e-7:
                continue
            radial_start_m = min(radii_m)
            radial_end_m = max(radii_m)
            is_upstream = radial_start_m >= UPSTREAM_RADIUS_M - 1.0e-7
            is_downstream = radial_end_m <= DOWNSTREAM_WALL_RADIUS_M + 1.0e-7
            if not (is_upstream or is_downstream):
                continue
            strips.append((radial_start_m, radial_end_m, min(elevations_m), max(elevations_m)))
        return tuple(sorted(set(strips)))

    def node_id(chainage_m: float, radius_m: float, z_m: float) -> int:
        angle = chainage_m / centerline_radius_m
        point = (radius_m * math.sin(angle), radius_m * math.cos(angle), z_m)
        key = _coordinate_key(point)
        if key not in node_ids:
            node_ids[key] = len(nodes) + 1
            nodes.append(point)
        return node_ids[key]

    def append_hex(
        radial_start_m: float,
        radial_end_m: float,
        cross_section: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> None:
        cell = tuple(node_id(chainage_m, radial_start_m, z_m) for chainage_m, z_m in cross_section) + tuple(
            node_id(chainage_m, radial_end_m, z_m) for chainage_m, z_m in cross_section
        )
        if _hex_orientation(nodes, cell) < 0.0:
            cell = (cell[0], cell[3], cell[2], cell[1], cell[4], cell[7], cell[6], cell[5])
        cells.append((5, cell))

    def append_prism(
        radial_start_m: float,
        radial_end_m: float,
        cross_section: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> None:
        cell = tuple(node_id(chainage_m, radial_start_m, z_m) for chainage_m, z_m in cross_section) + tuple(
            node_id(chainage_m, radial_end_m, z_m) for chainage_m, z_m in cross_section
        )
        if _prism_orientation(nodes, cell) < 0.0:
            cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
        cells.append((6, cell))

    def append_landing_slope(
        riser_chainage_m: float,
        landing_chainages_m: tuple[float, ...],
        radial_intervals_m: tuple[tuple[float, float, float, float], ...],
    ) -> None:
        for segment_index, (start_m, end_m) in enumerate(zip(landing_chainages_m, landing_chainages_m[1:])):
            start_fraction = abs(start_m - riser_chainage_m) / abs(landing_chainages_m[-1] - riser_chainage_m)
            end_fraction = abs(end_m - riser_chainage_m) / abs(landing_chainages_m[-1] - riser_chainage_m)
            for radial_start_m, radial_end_m, riser_lower_z_m, upper_z_m in radial_intervals_m:
                slope_z_m = upper_z_m - (upper_z_m - riser_lower_z_m) * start_fraction
                end_slope_z_m = upper_z_m - (upper_z_m - riser_lower_z_m) * end_fraction
                cross_section = (
                    (start_m, riser_lower_z_m),
                    (end_m, riser_lower_z_m),
                    (end_m, end_slope_z_m),
                    (start_m, slope_z_m),
                )
                if segment_index == len(landing_chainages_m) - 2:
                    append_prism(
                        radial_start_m,
                        radial_end_m,
                        (cross_section[0], cross_section[1], cross_section[3]),
                    )
                else:
                    append_hex(radial_start_m, radial_end_m, cross_section)

    for index, (start_m, end_m) in enumerate(zip(wall.chainages_m, wall.chainages_m[1:])):
        start_base_z_m = wall.base_levels_m[index]
        end_base_z_m = wall.base_levels_m[index + 1]
        if abs(start_base_z_m - end_base_z_m) < wall.element_size_m - 1.0e-9:
            continue
        if abs(abs(end_base_z_m - start_base_z_m) - wall.element_size_m) > 1.0e-9:
            continue
        if max(start_base_z_m, end_base_z_m) + wall.element_size_m > 1.0e-9:
            continue
        if start_m < CENTER_STEP_END_CHAINAGE_M and end_m > CENTER_STEP_START_CHAINAGE_M:
            continue

        if end_base_z_m < start_base_z_m:
            riser_index = index + 1
            landing_end_index = riser_index
            while (
                landing_end_index + 1 < len(wall.base_levels_m)
                and abs(wall.base_levels_m[landing_end_index + 1] - end_base_z_m) < 1.0e-9
            ):
                landing_end_index += 1
            if landing_end_index + 1 >= len(wall.chainages_m):
                continue
            landing_indices = range(riser_index, landing_end_index + 2)
        elif end_base_z_m > start_base_z_m:
            riser_index = index
            landing_end_index = riser_index
            while (
                landing_end_index > 0
                and abs(wall.base_levels_m[landing_end_index - 1] - start_base_z_m) < 1.0e-9
            ):
                landing_end_index -= 1
            if landing_end_index == 0:
                continue
            landing_indices = range(riser_index, landing_end_index - 2, -1)
        else:
            continue

        nominal_lower_z_m = min(start_base_z_m, end_base_z_m)
        nominal_upper_z_m = max(start_base_z_m, end_base_z_m)
        radial_intervals_m = exterior_riser_strips(
            wall.chainages_m[riser_index],
            nominal_lower_z_m,
            nominal_upper_z_m,
        )
        if not radial_intervals_m:
            continue
        landing_chainages_m = tuple(
            wall.chainages_m[station]
            for station in landing_indices
        )
        if landing_chainages_m[0] < CENTER_STEP_END_CHAINAGE_M and landing_chainages_m[-1] > CENTER_STEP_START_CHAINAGE_M:
            continue

        # Keep ordinary details to two existing, plinth-bonded landing spans.
        two_span_landing_chainages_m = landing_chainages_m[:3]
        append_landing_slope(
            wall.chainages_m[riser_index],
            two_span_landing_chainages_m,
            tuple(interval for interval in radial_intervals_m if interval[0] >= UPSTREAM_RADIUS_M - 1.0e-7),
        )

        downstream_landing_chainages_m = two_span_landing_chainages_m
        # Low approach details are one local 0.5 m triangular prism, fixed to the plinth riser.
        if min(start_base_z_m, end_base_z_m) < DOWNSTREAM_BATTER_TOP_Z_M - 1.0e-9:
            downstream_landing_chainages_m = landing_chainages_m[:2]
        append_landing_slope(
            wall.chainages_m[riser_index],
            downstream_landing_chainages_m,
            tuple(interval for interval in radial_intervals_m if interval[1] <= DOWNSTREAM_WALL_RADIUS_M + 1.0e-7),
        )

    face_patterns_by_cell_type = {
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    }
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for cell_type, cell in cells:
        for pattern in face_patterns_by_cell_type[cell_type]:
            face = tuple(cell[position] for position in pattern)
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face

    boundaries = []
    for key, count in face_counts.items():
        if count != 1:
            continue
        face = oriented_faces[key]
        radii_m = [math.hypot(nodes[node_id - 1][0], nodes[node_id - 1][1]) for node_id in face]
        if min(radii_m) >= UPSTREAM_RADIUS_M - 1.0e-6:
            boundary_id = UPSTREAM_BOUNDARY_ID
        elif max(radii_m) <= DOWNSTREAM_WALL_RADIUS_M + 1.0e-6:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        else:
            boundary_id = OTHER_BOUNDARY_ID
        boundaries.append((boundary_id, face))
    return PlinthHaunchMesh(nodes, cells, boundaries, wall.element_size_m)
