"""Generate the standalone curved plinth from the client contour data."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct


UPSTREAM_RADIUS_M = 81.0
DOWNSTREAM_RADIUS_M = 74.0
BODY_ID = 1
FOUNDATION_BOUNDARY_ID = 1
UPSTREAM_BOUNDARY_ID = 2
DOWNSTREAM_BOUNDARY_ID = 3
TOP_BOUNDARY_ID = 4
LEFT_END_BOUNDARY_ID = 5
RIGHT_END_BOUNDARY_ID = 6
OTHER_BOUNDARY_ID = 7


@dataclass(frozen=True)
class ContourPoint:
    chainage_m: float
    bedrock_z_m: float
    plinth_z_m: float


@dataclass
class PlinthMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]
    chainages_m: list[float]
    element_size_m: float


def _coordinate_key(point: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(round(value, 9) for value in point)


def load_contours(path: Path) -> list[ContourPoint]:
    rows = json.loads(path.read_text())
    contours = [
        ContourPoint(float(row["chainage"]), float(row["bedrock"]), float(row["plinth"]))
        for row in rows
    ]
    if len(contours) < 2 or any(a.chainage_m >= b.chainage_m for a, b in zip(contours, contours[1:])):
        raise ValueError("Plinth contour chainages must be strictly increasing")
    if any(point.bedrock_z_m > point.plinth_z_m for point in contours):
        raise ValueError("Bedrock must not be above the plinth contour")
    return contours


def load_global_element_size(path: Path) -> float:
    controls = json.loads(path.read_text())
    matches = [float(row["Client Value"]) for row in controls if row["Mesh Control"] == "Global Element Size"]
    if len(matches) != 1 or matches[0] <= 0.0:
        raise ValueError("Expected one positive Global Element Size")
    return matches[0]


def _monotone_values(contours: list[ContourPoint], attribute: str, chainage_m: float) -> float:
    stations = [point.chainage_m for point in contours]
    values = [getattr(point, attribute) for point in contours]
    if chainage_m <= stations[0]:
        return values[0]
    if chainage_m >= stations[-1]:
        return values[-1]
    secants = [
        (end_value - start_value) / (end_station - start_station)
        for start_station, end_station, start_value, end_value
        in zip(stations, stations[1:], values, values[1:])
    ]
    slopes = [secants[0]]
    for left, right in zip(secants, secants[1:]):
        slopes.append(0.0 if left * right <= 0.0 else 2.0 * left * right / (left + right))
    slopes.append(secants[-1])
    for index, (start_station, end_station) in enumerate(zip(stations, stations[1:])):
        if start_station <= chainage_m <= end_station:
            span = end_station - start_station
            fraction = (chainage_m - start_station) / span
            fraction_2 = fraction * fraction
            fraction_3 = fraction_2 * fraction
            return (
                (2.0 * fraction_3 - 3.0 * fraction_2 + 1.0) * values[index]
                + (fraction_3 - 2.0 * fraction_2 + fraction) * span * slopes[index]
                + (-2.0 * fraction_3 + 3.0 * fraction_2) * values[index + 1]
                + (fraction_3 - fraction_2) * span * slopes[index + 1]
            )
    raise ValueError(f"Chainage {chainage_m:g} is outside the plinth contour")


def _chainages(contours: list[ContourPoint], element_size_m: float) -> list[float]:
    start = contours[0].chainage_m
    end = contours[-1].chainage_m
    values = {point.chainage_m for point in contours}
    step = 0
    while start + step * element_size_m < end:
        values.add(round(start + step * element_size_m, 9))
        step += 1
    values.add(end)
    return sorted(values)


def _vertical_levels(bottom_z_m: float, top_z_m: float, element_size_m: float) -> list[float]:
    boundary_clearance_m = 0.5 * element_size_m
    levels = [bottom_z_m]
    first_layer_index = math.ceil((bottom_z_m + boundary_clearance_m) / element_size_m)
    last_layer_index = math.floor((top_z_m - boundary_clearance_m) / element_size_m)
    levels.extend(
        layer_index * element_size_m
        for layer_index in range(first_layer_index, last_layer_index + 1)
    )
    levels.append(top_z_m)
    return sorted(set(round(level, 9) for level in levels))


def _triangulate_convex_polygon(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    triangles = []
    for index in range(1, len(vertices) - 1):
        triangle = (vertices[0], vertices[index], vertices[index + 1])
        if len({side for side, _ in triangle}) > 1:
            triangles.append(triangle)
    return triangles


def _section_faces(left_levels: list[float], right_levels: list[float]) -> list[tuple[tuple[int, int], ...]]:
    left_by_z = {z_m: index for index, z_m in enumerate(left_levels)}
    right_by_z = {z_m: index for index, z_m in enumerate(right_levels)}
    common = sorted(set(left_by_z) & set(right_by_z))
    faces: list[tuple[tuple[int, int], ...]] = []

    if not common:
        polygon = (
            [(0, index) for index in range(len(left_levels))]
            + [(1, index) for index in range(len(right_levels) - 1, -1, -1)]
        )
        return _triangulate_convex_polygon(polygon)

    lower_z_m = common[0]
    lower_polygon = (
        [(0, index) for index, z_m in enumerate(left_levels) if z_m <= lower_z_m]
        + [(1, index) for index in range(len(right_levels) - 1, -1, -1) if right_levels[index] <= lower_z_m]
    )
    faces.extend(_triangulate_convex_polygon(lower_polygon))

    for lower_level_m, upper_level_m in zip(common, common[1:]):
        faces.append((
            (0, left_by_z[lower_level_m]),
            (1, right_by_z[lower_level_m]),
            (1, right_by_z[upper_level_m]),
            (0, left_by_z[upper_level_m]),
        ))

    upper_z_m = common[-1]
    upper_polygon = (
        [(0, index) for index, z_m in enumerate(left_levels) if z_m >= upper_z_m]
        + [(1, index) for index in range(len(right_levels) - 1, -1, -1) if right_levels[index] >= upper_z_m]
    )
    faces.extend(_triangulate_convex_polygon(upper_polygon))
    return faces


def _top_aligned_section_faces(
    left_levels: list[float],
    right_levels: list[float],
) -> list[tuple[tuple[int, int], ...]]:
    left_extra = max(len(left_levels) - len(right_levels), 0)
    right_extra = max(len(right_levels) - len(left_levels), 0)
    faces = _triangulate_convex_polygon(
        [(0, index) for index in range(left_extra + 1)]
        + [(1, index) for index in range(right_extra, -1, -1)]
    )
    common_level_count = min(len(left_levels), len(right_levels))
    faces.extend(
        (
            (0, left_extra + index),
            (1, right_extra + index),
            (1, right_extra + index + 1),
            (0, left_extra + index + 1),
        )
        for index in range(common_level_count - 1)
    )
    return faces


def _determinant(a, b, c) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _prism_orientation(nodes: list[tuple[float, float, float]], cell: tuple[int, ...]) -> float:
    points = [nodes[node_id - 1] for node_id in cell]
    d_chainage = tuple(points[1][index] - points[0][index] for index in range(3))
    d_z = tuple(points[2][index] - points[0][index] for index in range(3))
    d_radius = tuple(points[3][index] - points[0][index] for index in range(3))
    return _determinant(d_chainage, d_z, d_radius)


def _hex_orientation(nodes: list[tuple[float, float, float]], cell: tuple[int, ...]) -> float:
    points = [nodes[node_id - 1] for node_id in cell]
    d_chainage = tuple(points[1][index] - points[0][index] for index in range(3))
    d_z = tuple(points[3][index] - points[0][index] for index in range(3))
    d_radius = tuple(points[4][index] - points[0][index] for index in range(3))
    return _determinant(d_chainage, d_z, d_radius)


def build_plinth(root: Path) -> PlinthMesh:
    contours = load_contours(root / "Data" / "plinth.json")
    element_size_m = load_global_element_size(root / "Data" / "Computational_Grid_Controls.json")
    config = json.loads((root / "config.json").read_text())
    centerline_radius_m = float(config["wall_centerline_radius_m"])
    chainages_m = _chainages(contours, element_size_m)
    radial_count = round((UPSTREAM_RADIUS_M - DOWNSTREAM_RADIUS_M) / element_size_m)
    radial_levels_m = [DOWNSTREAM_RADIUS_M + index * element_size_m for index in range(radial_count + 1)]

    section_levels = []
    for chainage_m in chainages_m:
        bedrock_z_m = _monotone_values(contours, "bedrock_z_m", chainage_m)
        plinth_z_m = _monotone_values(contours, "plinth_z_m", chainage_m)
        if bedrock_z_m > plinth_z_m:
            raise ValueError(f"Plinth intersects immutable bedrock at chainage {chainage_m:g}")
        section_levels.append(_vertical_levels(bedrock_z_m, plinth_z_m, element_size_m))

    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple[int, int, int], int] = {}
    for station_index, (chainage_m, levels_m) in enumerate(zip(chainages_m, section_levels)):
        angle = chainage_m / centerline_radius_m
        for vertical_index, z_m in enumerate(levels_m):
            for radial_index, radius_m in enumerate(radial_levels_m):
                node_ids[(station_index, vertical_index, radial_index)] = len(nodes) + 1
                nodes.append((radius_m * math.sin(angle), radius_m * math.cos(angle), z_m))

    cells: list[tuple[int, tuple[int, ...]]] = []
    for station_index in range(len(chainages_m) - 1):
        faces = _top_aligned_section_faces(section_levels[station_index], section_levels[station_index + 1])
        for radial_index in range(radial_count):
            for face in faces:
                inner = tuple(node_ids[(station_index + side, vertical_index, radial_index)] for side, vertical_index in face)
                outer = tuple(node_ids[(station_index + side, vertical_index, radial_index + 1)] for side, vertical_index in face)
                if len(face) == 3:
                    cell = inner + outer
                    if _prism_orientation(nodes, cell) < 0.0:
                        cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
                    cells.append((6, cell))
                else:
                    cell = inner + outer
                    if _hex_orientation(nodes, cell) < 0.0:
                        cell = (cell[0], cell[3], cell[2], cell[1], cell[4], cell[7], cell[6], cell[5])
                    cells.append((5, cell))

    face_patterns = {
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    }
    face_counts: Counter[tuple[int, ...]] = Counter()
    oriented_faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    face_owners: dict[tuple[int, ...], int] = {}
    parents = list(range(len(cells)))

    def find(cell_index: int) -> int:
        while parents[cell_index] != cell_index:
            parents[cell_index] = parents[parents[cell_index]]
            cell_index = parents[cell_index]
        return cell_index

    for cell_index, (element_type, cell) in enumerate(cells):
        for pattern in face_patterns[element_type]:
            face = tuple(cell[index] for index in pattern)
            key = tuple(sorted(face))
            face_counts[key] += 1
            oriented_faces[key] = face
            if key in face_owners:
                first_root = find(cell_index)
                second_root = find(face_owners[key])
                if first_root != second_root:
                    parents[second_root] = first_root
            else:
                face_owners[key] = cell_index

    def node_chainage(node_id: int) -> float:
        x_m, y_m, _ = nodes[node_id - 1]
        return centerline_radius_m * math.atan2(x_m, y_m)

    boundaries = []
    tolerance = 1.0e-6
    for key, count in face_counts.items():
        if count != 1:
            continue
        face = oriented_faces[key]
        coordinates = [nodes[node_id - 1] for node_id in face]
        radii_m = [math.hypot(x_m, y_m) for x_m, y_m, _ in coordinates]
        stations_m = [node_chainage(node_id) for node_id in face]
        if max(abs(radius_m - UPSTREAM_RADIUS_M) for radius_m in radii_m) < tolerance:
            boundary_id = UPSTREAM_BOUNDARY_ID
        elif max(abs(radius_m - DOWNSTREAM_RADIUS_M) for radius_m in radii_m) < tolerance:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        elif max(abs(station_m - chainages_m[0]) for station_m in stations_m) < tolerance:
            boundary_id = LEFT_END_BOUNDARY_ID
        elif max(abs(station_m - chainages_m[-1]) for station_m in stations_m) < tolerance:
            boundary_id = RIGHT_END_BOUNDARY_ID
        else:
            on_bedrock = all(
                abs(z_m - _monotone_values(contours, "bedrock_z_m", station_m)) < tolerance
                for station_m, (_, _, z_m) in zip(stations_m, coordinates)
            )
            on_plinth = all(
                abs(z_m - _monotone_values(contours, "plinth_z_m", station_m)) < tolerance
                for station_m, (_, _, z_m) in zip(stations_m, coordinates)
            )
            if on_bedrock:
                boundary_id = FOUNDATION_BOUNDARY_ID
            elif on_plinth:
                boundary_id = TOP_BOUNDARY_ID
            else:
                boundary_id = OTHER_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return PlinthMesh(nodes, cells, boundaries, chainages_m, element_size_m)


def build_conforming_plinth(root: Path, interface_meshes: list[object]) -> PlinthMesh:
    contours = load_contours(root / "Data" / "plinth.json")
    element_size_m = load_global_element_size(root / "Data" / "Computational_Grid_Controls.json")
    config = json.loads((root / "config.json").read_text())
    centerline_radius_m = float(config["wall_centerline_radius_m"])

    top_faces: dict[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]] = {}
    segment_nodes: dict[tuple[float, float], dict[float, list[tuple[float, float, float]]]] = {}
    for mesh in interface_meshes:
        for boundary_id, face in mesh.boundaries:
            if boundary_id != 1:
                continue
            coordinates = tuple(mesh.nodes[node_id - 1] for node_id in face)
            key = tuple(sorted(_coordinate_key(point) for point in coordinates))
            top_faces[key] = coordinates
            chainages = sorted({round(centerline_radius_m * math.atan2(x_m, y_m), 9) for x_m, y_m, _ in coordinates})
            if len(chainages) != 2:
                raise ValueError("A structural base face must span exactly two chainage stations")
            stations = segment_nodes.setdefault((chainages[0], chainages[1]), {})
            for point in coordinates:
                chainage_m = round(centerline_radius_m * math.atan2(point[0], point[1]), 9)
                stations.setdefault(chainage_m, []).append(point)

    for (start_m, end_m), stations in segment_nodes.items():
        if set(stations) != {start_m, end_m}:
            raise ValueError("Incomplete structural base segment")
        start_points = stations[start_m]
        end_points = stations[end_m]
        start_min = min(math.hypot(x_m, y_m) for x_m, y_m, _ in start_points)
        end_min = min(math.hypot(x_m, y_m) for x_m, y_m, _ in end_points)
        start_max = max(math.hypot(x_m, y_m) for x_m, y_m, _ in start_points)
        end_max = max(math.hypot(x_m, y_m) for x_m, y_m, _ in end_points)
        tread_z_m = max(point[2] for point in start_points + end_points)

        def top_point(chainage_m: float, radius_m: float) -> tuple[float, float, float]:
            angle = chainage_m / centerline_radius_m
            return (
                radius_m * math.sin(angle),
                radius_m * math.cos(angle),
                tread_z_m,
            )

        strips = []
        if min(start_min, end_min) > DOWNSTREAM_RADIUS_M + 1.0e-9:
            strips.append((
                top_point(start_m, DOWNSTREAM_RADIUS_M),
                top_point(end_m, DOWNSTREAM_RADIUS_M),
                top_point(end_m, end_min),
                top_point(start_m, start_min),
            ))
        if max(start_max, end_max) < UPSTREAM_RADIUS_M - 1.0e-9:
            strips.append((
                top_point(start_m, start_max),
                top_point(end_m, end_max),
                top_point(end_m, UPSTREAM_RADIUS_M),
                top_point(start_m, UPSTREAM_RADIUS_M),
            ))
        for face in strips:
            key = tuple(sorted(_coordinate_key(point) for point in face))
            top_faces[key] = face

    chainages_m = sorted({
        round(centerline_radius_m * math.atan2(x_m, y_m), 9)
        for face in top_faces.values()
        for x_m, y_m, _ in face
    })
    highest_top_z_m = max(point[2] for face in top_faces.values() for point in face)
    lowest_bedrock_z_m = min(
        _monotone_values(contours, "bedrock_z_m", chainage_m)
        for chainage_m in chainages_m
    )
    highest_layer_index = math.ceil(highest_top_z_m / element_size_m)
    lowest_layer_index = math.floor(lowest_bedrock_z_m / element_size_m)
    global_levels_m = [
        layer_index * element_size_m
        for layer_index in range(highest_layer_index, lowest_layer_index - 1, -1)
    ]

    nodes: list[tuple[float, float, float]] = []
    coordinate_nodes: dict[tuple[float, float, float], int] = {}
    bedrock_by_top_node = {
        _coordinate_key(point): _monotone_values(
            contours,
            "bedrock_z_m",
            centerline_radius_m * math.atan2(point[0], point[1]),
        )
        for face in top_faces.values()
        for point in face
    }

    def node_id(point: tuple[float, float, float], target_z_m: float) -> int:
        x_m, y_m, top_z_m = point
        bedrock_z_m = bedrock_by_top_node[_coordinate_key(point)]
        boundary_clearance_m = 0.5 * element_size_m
        if target_z_m >= top_z_m - boundary_clearance_m:
            z_m = top_z_m
        elif target_z_m <= bedrock_z_m + boundary_clearance_m:
            z_m = bedrock_z_m
        else:
            z_m = target_z_m
        coordinate = (x_m, y_m, z_m)
        key = _coordinate_key(coordinate)
        if key not in coordinate_nodes:
            coordinate_nodes[key] = len(nodes) + 1
            nodes.append(coordinate)
        return coordinate_nodes[key]

    cells: list[tuple[int, tuple[int, ...]]] = []

    def append_prism(first: tuple[int, int, int], second: tuple[int, int, int]) -> None:
        cell = first + second
        if _prism_orientation(nodes, cell) < 0.0:
            cell = (cell[0], cell[2], cell[1], cell[3], cell[5], cell[4])
        cells.append((6, cell))

    def append_tetrahedron(cell: tuple[int, int, int, int]) -> None:
        points = [nodes[value - 1] for value in cell]
        vectors = [
            tuple(points[index][axis] - points[0][axis] for axis in range(3))
            for index in range(1, 4)
        ]
        determinant = _determinant(*vectors)
        if abs(determinant) < 1.0e-12:
            raise ValueError("Zero-volume tetrahedron in conforming plinth")
        if determinant < 0.0:
            cell = (cell[0], cell[2], cell[1], cell[3])
        cells.append((4, cell))

    def append_pyramid(base: tuple[int, int, int, int], apex: int) -> None:
        points = [nodes[value - 1] for value in base + (apex,)]
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
            raise ValueError("Zero-volume pyramid in conforming plinth")
        if volume_sign < 0.0:
            base = (base[0], base[3], base[2], base[1])
        cells.append((7, base + (apex,)))

    for face in top_faces.values():
        for upper_z_m, lower_z_m in zip(global_levels_m, global_levels_m[1:]):
            upper = tuple(node_id(point, upper_z_m) for point in face)
            lower = tuple(node_id(point, lower_z_m) for point in face)
            unique_nodes = len(set(upper + lower))
            if len(face) == 3 and unique_nodes == 6:
                append_prism(upper, lower)
            elif len(face) == 3 and unique_nodes == 5:
                collapsed_index = next(
                    index for index, (upper_id, lower_id) in enumerate(zip(upper, lower))
                    if upper_id == lower_id
                )
                active_indices = [index for index in range(3) if index != collapsed_index]
                base = (
                    upper[active_indices[0]],
                    upper[active_indices[1]],
                    lower[active_indices[1]],
                    lower[active_indices[0]],
                )
                append_pyramid(base, upper[collapsed_index])
            elif len(face) == 3 and unique_nodes == 4:
                collapsed_indices = [
                    index for index, (upper_id, lower_id) in enumerate(zip(upper, lower))
                    if upper_id == lower_id
                ]
                active_index = next(index for index in range(3) if index not in collapsed_indices)
                append_tetrahedron((
                    upper[collapsed_indices[0]],
                    upper[collapsed_indices[1]],
                    upper[active_index],
                    lower[active_index],
                ))
            elif len(face) == 3 and unique_nodes == 3:
                continue
            elif len(face) == 4 and unique_nodes == 8:
                cell = upper + lower
                if _hex_orientation(nodes, cell) < 0.0:
                    cell = (cell[0], cell[3], cell[2], cell[1], cell[4], cell[7], cell[6], cell[5])
                cells.append((5, cell))
            elif len(face) == 4 and unique_nodes == 6:
                collapsed = [index for index, (upper_id, lower_id) in enumerate(zip(upper, lower)) if upper_id == lower_id]
                if len(collapsed) != 2:
                    raise ValueError("Unsupported collapsed plinth quadrilateral")
                triangles = []
                for collapsed_index in collapsed:
                    adjacent = [
                        (collapsed_index - 1) % 4,
                        (collapsed_index + 1) % 4,
                    ]
                    active_index = next(index for index in adjacent if upper[index] != lower[index])
                    triangles.append((upper[collapsed_index], upper[active_index], lower[active_index]))
                append_prism(triangles[0], triangles[1])
            elif len(face) == 4 and unique_nodes == 4:
                continue
            else:
                raise ValueError(
                    f"Unsupported conforming plinth extrusion: {len(face)}-node face, {unique_nodes} unique nodes"
                )

    face_patterns = {
        4: ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        5: ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
        6: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
        7: ((0, 1, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)),
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
        if max(abs(radius_m - UPSTREAM_RADIUS_M) for radius_m in radii_m) < tolerance:
            boundary_id = UPSTREAM_BOUNDARY_ID
        elif max(abs(radius_m - DOWNSTREAM_RADIUS_M) for radius_m in radii_m) < tolerance:
            boundary_id = DOWNSTREAM_BOUNDARY_ID
        elif max(abs(station_m - chainages_m[0]) for station_m in stations_m) < tolerance:
            boundary_id = LEFT_END_BOUNDARY_ID
        elif max(abs(station_m - chainages_m[-1]) for station_m in stations_m) < tolerance:
            boundary_id = RIGHT_END_BOUNDARY_ID
        elif all(
            abs(z_m - _monotone_values(contours, "bedrock_z_m", station_m)) < tolerance
            for station_m, (_, _, z_m) in zip(stations_m, coordinates)
        ):
            boundary_id = FOUNDATION_BOUNDARY_ID
        elif all(
            abs(z_m - _monotone_values(contours, "plinth_z_m", station_m)) < tolerance
            for station_m, (_, _, z_m) in zip(stations_m, coordinates)
        ):
            boundary_id = TOP_BOUNDARY_ID
        else:
            boundary_id = OTHER_BOUNDARY_ID
        boundaries.append((boundary_id, face))

    return PlinthMesh(nodes, cells, boundaries, chainages_m, element_size_m)


def write_gmsh(mesh: PlinthMesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
        stream.write(f"{len(mesh.nodes)}\n")
        for node_id, (x_m, y_m, z_m) in enumerate(mesh.nodes, start=1):
            stream.write(f"{node_id} {x_m:.12g} {y_m:.12g} {z_m:.12g}\n")
        stream.write("$EndNodes\n$Elements\n")
        stream.write(f"{len(mesh.cells) + len(mesh.boundaries)}\n")
        element_id = 1
        for element_type, cell in mesh.cells:
            stream.write(f"{element_id} {element_type} 2 {BODY_ID} {BODY_ID} {' '.join(map(str, cell))}\n")
            element_id += 1
        for boundary_id, face in mesh.boundaries:
            element_type = 2 if len(face) == 3 else 3
            stream.write(f"{element_id} {element_type} 2 {boundary_id} {boundary_id} {' '.join(map(str, face))}\n")
            element_id += 1
        stream.write("$EndElements\n")


def write_vtu(mesh: PlinthMesh, path: Path) -> None:
    """Write the plinth volume cells in ParaView's native unstructured-grid format."""
    connectivity = [node_id - 1 for _, cell in mesh.cells for node_id in cell]
    offsets = []
    offset = 0
    for _, cell in mesh.cells:
        offset += len(cell)
        offsets.append(offset)
    vtk_cell_types = {4: 10, 5: 12, 6: 13, 7: 14}
    cell_types = [vtk_cell_types[element_type] for element_type, _ in mesh.cells]
    points = [coordinate for point in mesh.nodes for coordinate in point]
    chunks = [
        struct.pack(f"<I{len(points)}d", 8 * len(points), *points),
        struct.pack(f"<I{len(connectivity)}i", 4 * len(connectivity), *connectivity),
        struct.pack(f"<I{len(offsets)}i", 4 * len(offsets), *offsets),
        struct.pack(f"<I{len(cell_types)}B", len(cell_types), *cell_types),
    ]
    chunk_offsets = []
    byte_offset = 0
    for chunk in chunks:
        chunk_offsets.append(byte_offset)
        byte_offset += len(chunk)
    header = "\n".join((
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{len(mesh.nodes)}" NumberOfCells="{len(mesh.cells)}">',
        "      <PointData/>",
        "      <CellData/>",
        "      <Points>",
        f'        <DataArray type="Float64" NumberOfComponents="3" format="appended" offset="{chunk_offsets[0]}"/>',
        "      </Points>",
        "      <Cells>",
        f'        <DataArray type="Int32" Name="connectivity" format="appended" offset="{chunk_offsets[1]}"/>',
        f'        <DataArray type="Int32" Name="offsets" format="appended" offset="{chunk_offsets[2]}"/>',
        f'        <DataArray type="UInt8" Name="types" format="appended" offset="{chunk_offsets[3]}"/>',
        "      </Cells>",
        "    </Piece>",
        "  </UnstructuredGrid>",
        '  <AppendedData encoding="raw">',
    )).encode("ascii")
    path.write_bytes(header + b"\n_" + b"".join(chunks) + b"\n  </AppendedData>\n</VTKFile>\n")


def audit_plinth(mesh: PlinthMesh) -> dict[str, object]:
    element_counts = Counter(element_type for element_type, _ in mesh.cells)
    boundary_counts = Counter(boundary_id for boundary_id, _ in mesh.boundaries)
    off_grid_nodes = {
        node_id
        for node_id, (_, _, z_m) in enumerate(mesh.nodes, start=1)
        if abs(z_m / mesh.element_size_m - round(z_m / mesh.element_size_m)) > 1.0e-8
    }
    contour_nodes = {
        node_id
        for boundary_id, face in mesh.boundaries
        if boundary_id in (FOUNDATION_BOUNDARY_ID, TOP_BOUNDARY_ID)
        for node_id in face
    }
    unexpected_off_grid_nodes = off_grid_nodes - contour_nodes
    non_bedrock_transition_cells = []
    for cell_index, (element_type, cell) in enumerate(mesh.cells, start=1):
        if element_type == 5:
            continue
        touches_bedrock = any(
            abs(
                mesh.nodes[node_id - 1][2]
                - _monotone_values(
                    load_contours(Path(__file__).resolve().parents[2] / "Data" / "plinth.json"),
                    "bedrock_z_m",
                    math.atan2(mesh.nodes[node_id - 1][0], mesh.nodes[node_id - 1][1])
                    * json.loads((Path(__file__).resolve().parents[2] / "config.json").read_text())["wall_centerline_radius_m"],
                )
            ) < 1.0e-6
            for node_id in cell
        )
        if not touches_bedrock:
            non_bedrock_transition_cells.append(cell_index)
    expected_radial_cells = round((UPSTREAM_RADIUS_M - DOWNSTREAM_RADIUS_M) / mesh.element_size_m)
    if not {FOUNDATION_BOUNDARY_ID, UPSTREAM_BOUNDARY_ID, DOWNSTREAM_BOUNDARY_ID, TOP_BOUNDARY_ID} <= set(boundary_counts):
        raise ValueError(f"Missing plinth boundaries: found {sorted(boundary_counts)}")
    if unexpected_off_grid_nodes:
        raise ValueError(f"Plinth has {len(unexpected_off_grid_nodes)} off-grid nodes outside contour closures")
    if non_bedrock_transition_cells:
        raise ValueError(
            f"Plinth has {len(non_bedrock_transition_cells)} non-hexahedral cells outside the bedrock closure"
        )
    component_count = len({find(cell_index) for cell_index in range(len(cells))})
    if component_count != 1:
        raise ValueError(f"Plinth has {component_count} face-connected components")
    return {
        "nodes": len(mesh.nodes),
        "cells": len(mesh.cells),
        "tetrahedra": element_counts[4],
        "hexahedra": element_counts[5],
        "boundary_prisms": element_counts[6],
        "pyramids": element_counts[7],
        "boundary_faces": dict(sorted(boundary_counts.items())),
        "chainage_segments": len(mesh.chainages_m) - 1,
        "radial_cells": expected_radial_cells,
        "exact_contour_nodes_off_grid": len(off_grid_nodes),
        "interior_nodes_off_grid": len(unexpected_off_grid_nodes),
        "face_connected_components": component_count,
    }
