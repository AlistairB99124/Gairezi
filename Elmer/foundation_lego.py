"""LEGO-column foundation and near-seat shell contract."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from .foundation_trench import FoundationMesh, tet_volumes
from .graded_bedrock import audit_tetrahedral_boundary
from .regions.tetrahedral import _CELL_FACES, split_cell, split_face


def build_lego_foundation(config: dict, concrete) -> FoundationMesh:
    from .model_config import wall_thickness_steps

    geometry = config["geometry"]
    settings, wall = geometry["foundation"], geometry["wall"]
    depth, margin = float(settings["depth_m"]), float(settings["side_margin_m"])
    size = float(geometry["global_element_size_m"])
    upstream = float(wall["upstream_radius_m"])
    width = max(value for _, value in wall_thickness_steps(wall)) + 2 * margin
    inner, outer = upstream + margin - width, upstream + margin
    coordinates = np.asarray(concrete.nodes, dtype=float)
    wall_source = [tuple(face) for tag, face in concrete.boundaries if tag == 1]
    wall_ids = sorted({node for face in wall_source for node in face})
    if len(wall_ids) < 4 or len(wall_source) < 2:
        raise ValueError("LEGO foundation requires a complete wall seat")

    points: list[tuple[float, float, float]] = []
    lookup: dict[tuple[float, float, float], int] = {}

    def node(point):
        key = tuple(np.round(point, 8))
        if key not in lookup:
            lookup[key] = len(points)
            points.append(tuple(point))
        return lookup[key]

    for wall_node in wall_ids:
        node(coordinates[wall_node])

    source_faces = [tuple(node(coordinates[n]) for n in face) for face in wall_source]
    source_triangles = [
        tuple(triangle)
        for face in source_faces
        for triangle in split_face(face)
    ]
    source_triangle_keys = {tuple(sorted(face)) for face in source_triangles}
    source_polygons: dict[tuple[float, float, float], list[np.ndarray]] = {}
    for face in source_faces:
        polygon = np.asarray(points)[list(face), :]
        if np.ptp(polygon[:, 2]) > 1e-7:
            continue
        angles = sorted({
            round(math.atan2(x, y), 10)
            for x, y, _ in polygon
        })
        if len(angles) != 2:
            continue
        key = (angles[0], angles[1], round(float(polygon[0, 2]), 8))
        source_polygons.setdefault(key, []).append(polygon)

    def on_source_surface(triangle: tuple[int, int, int]) -> bool:
        candidate = np.asarray(points)[list(triangle), :]
        if np.ptp(candidate[:, 2]) > 1e-7:
            return False
        angles = sorted({
            round(math.atan2(x, y), 10)
            for x, y, _ in candidate
        })
        if len(angles) != 2:
            return False
        key = (angles[0], angles[1], round(float(candidate[0, 2]), 8))
        for polygon in source_polygons.get(key, ()):
            p0, p1, p2 = polygon[:3]
            normal = np.cross(p1 - p0, p2 - p0)
            magnitude = np.linalg.norm(normal)
            if magnitude <= 1e-14:
                continue
            normal /= magnitude
            if np.max(np.abs((candidate - p0) @ normal)) > 1e-7:
                continue
            drop = int(np.argmax(np.abs(normal)))
            axes = [axis for axis in range(3) if axis != drop]
            polygon_2d = polygon[:, axes]
            points_2d = candidate[:, axes]
            inside = True
            for point in points_2d:
                crossings = 0
                for a, b in zip(polygon_2d, np.roll(polygon_2d, -1, axis=0)):
                    if (a[1] > point[1]) == (b[1] > point[1]):
                        continue
                    x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                    if x >= point[0] - 1e-9:
                        crossings += 1
                if crossings % 2 == 0 and min(
                    np.linalg.norm(point - vertex) for vertex in polygon_2d
                ) > 1e-9:
                    inside = False
                    break
            if inside:
                return True
        return False

    seat_by_interval: dict[tuple[float, float], float] = {}
    for face in source_faces:
        xyz = np.asarray(points)[list(face)]
        if np.ptp(xyz[:, 2]) > 1e-7:
            continue
        angles = sorted({round(math.atan2(x, y), 10) for x, y, _ in xyz})
        if len(angles) != 2:
            continue
        key = tuple(angles)
        elevation = float(xyz[0, 2])
        previous = seat_by_interval.setdefault(key, elevation)
        if not math.isclose(previous, elevation, abs_tol=1e-7):
            raise ValueError("One LEGO wall interval has inconsistent bottom courses")
    if not seat_by_interval:
        raise ValueError("Wall seat has no horizontal LEGO bottom faces")

    station_angles = sorted({angle for pair in seat_by_interval for angle in pair})
    extension = max(size, depth) / inner
    angles = [station_angles[0] - extension, *station_angles,
              station_angles[-1] + extension]
    interval_tops = []
    for left, right in zip(angles, angles[1:]):
        overlapping = [
            top for (a, b), top in seat_by_interval.items()
            if max(left, a) < min(right, b) + 1e-10
        ]
        if overlapping:
            interval_tops.append(min(overlapping))
        elif left < station_angles[0]:
            interval_tops.append(min(seat_by_interval.values(),
                                      key=lambda top: top))
        else:
            interval_tops.append(min(seat_by_interval.values(),
                                      key=lambda top: top))

    source_radii = {round(math.hypot(x, y), 8) for x, y, _ in coordinates[wall_ids]}
    radii = sorted(source_radii | {round(inner, 8), round(outer, 8)})
    complete = [radii[0]]
    for left, right in zip(radii, radii[1:]):
        divisions = max(1, math.ceil((right - left) / size - 1e-9))
        complete.extend(np.linspace(left, right, divisions + 1)[1:])
    radii = list(map(float, complete))

    xy = {
        (round(math.atan2(x, y), 10), round(math.hypot(x, y), 8)): (x, y)
        for x, y, _ in coordinates[wall_ids]
    }

    def grid(angle, radius, elevation):
        key = (angle, round(radius, 8))
        x, y = xy.get(key, (radius * math.sin(angle), radius * math.cos(angle)))
        return node((x, y, elevation))

    global_levels = sorted(set(
        [round(top, 9) for top in interval_tops]
        + [round(top - depth, 9) for top in interval_tops]
        + [
            round(min(interval_tops[index:index + 2]) - depth, 9)
            for index in range(len(interval_tops) - 1)
        ]
    ))
    interval_bottoms = [
        min(interval_tops) - depth
        for _ in interval_tops
    ]
    hexes: list[tuple[int, ...]] = []
    for column, top in enumerate(interval_tops):
        bottom = interval_bottoms[column]
        levels = [bottom, *(level for level in global_levels if bottom < level < top), top]
        if len(levels) < 2:
            raise ValueError("Foundation LEGO column has no vertical thickness")
        for lower, upper in zip(levels, levels[1:]):
            if upper - lower <= 1e-8:
                continue
            for radial in range(len(radii) - 1):
                hexes.append((
                    grid(angles[column], radii[radial], lower),
                    grid(angles[column + 1], radii[radial], lower),
                    grid(angles[column + 1], radii[radial + 1], lower),
                    grid(angles[column], radii[radial + 1], lower),
                    grid(angles[column], radii[radial], upper),
                    grid(angles[column + 1], radii[radial], upper),
                    grid(angles[column + 1], radii[radial + 1], upper),
                    grid(angles[column], radii[radial + 1], upper),
                ))

    owner_count: Counter[tuple[int, ...]] = Counter()
    oriented: dict[tuple[int, ...], tuple[int, ...]] = {}
    for cell in hexes:
        for pattern in _CELL_FACES[5]:
            face = tuple(cell[index] for index in pattern)
            key = tuple(sorted(face))
            owner_count[key] += 1
            oriented[key] = face

    roof, wall_faces, contact, terrain_risers = [], [], [], []
    for key, count in owner_count.items():
        if count != 1:
            continue
        face = oriented[key]
        triangles = split_face(face)
        matches = all(
            tuple(sorted(triangle)) in source_triangle_keys
            or on_source_surface(tuple(triangle))
            for triangle in triangles
        )
        if matches:
            wall_faces.extend(triangles)
            roof.extend(triangles)
            continue
        face_points = np.asarray(points)[list(face)]
        face_z = face_points[:, 2]
        face_angles = {round(math.atan2(x, y), 10) for x, y, _ in face_points}
        is_internal_riser = (
            len(face_angles) == 1
            and next(iter(face_angles)) in set(station_angles[1:-1])
            and np.ptp(face_z) > 1e-7
            and np.ptp(np.hypot(face_points[:, 0], face_points[:, 1])) > 1e-7
        )
        is_top_surface = (
            np.ptp(face_z) <= 1e-7
            and any(
                math.isclose(float(face_z[0]), top, abs_tol=1e-7)
                for top in interval_tops
            )
        )
        is_top_riser = is_internal_riser
        if is_top_surface or is_top_riser:
            roof.extend(triangles)
            if is_top_riser:
                terrain_risers.extend(triangles)
        else:
            contact.extend(triangles)
    if not wall_faces:
        raise ValueError("LEGO foundation has no exact wall-seat subtriangles")
    if not contact:
        raise ValueError("LEGO foundation has no bedrock contact")

    point_array = np.asarray(points, dtype=float)
    tetrahedra = np.asarray(
        [tet for cell in hexes for tet in split_cell(point_array, 5, cell)],
        dtype=np.int64,
    )
    audit_tetrahedral_boundary(
        tetrahedra, np.asarray(roof + contact, dtype=np.int64).reshape(-1, 3))

    snap = np.asarray([
        np.linalg.norm(point_array[node(coordinates[n])] - coordinates[n])
        for n in wall_ids
    ])
    report = {
        "present": True,
        "material": "concrete",
        "region_id": 3,
        "depth_m": depth,
        "side_margin_m": margin,
        "width_m": width,
        "radial_limits_m": [inner, outer],
        "seat_definition": "lego_interval_course",
        "depth_definition": "whole horizontal vertical courses",
        "construction_cell_type": "hexahedron",
        "construction_hexahedra": len(hexes),
        "solver_cell_type": "tetrahedron",
        "element_size_m": size,
        "minimum_vertical_thickness_m": depth,
        "minimum_normal_thickness_m": depth,
        "chamfer_count": 0,
        "volume_m3": float(tet_volumes(point_array, tetrahedra).sum()),
        "open_excavation_volume_m3": 0.0,
        "wall_seat_node_snap": {
            "wall_nodes": len(wall_ids),
            "matched_nodes": int((snap <= 1e-6).sum()),
            "unmatched_nodes": int((snap > 1e-6).sum()),
            "maximum_coincidence_distance_m": float(snap.max()),
            "tolerance_m": 1e-6,
            "matched_faces": len(wall_faces),
        },
    }
    return FoundationMesh(
        point_array, tetrahedra, np.asarray(roof, dtype=np.int64),
        np.asarray(wall_faces, dtype=np.int64),
        np.asarray(contact, dtype=np.int64), report,
        np.asarray(hexes, dtype=np.int64),
        np.asarray(roof, dtype=np.int64),
        np.asarray(terrain_risers, dtype=np.int64),
    )
