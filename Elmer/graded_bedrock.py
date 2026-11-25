"""Conforming distance-graded HEXA8 rock with a locked concrete seat."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import warnings

import gmsh
import numpy as np

from .bedrock_preview import (
    BANK_MARGIN_M, STREAM_MARGIN_M, seated_bedrock_geometry,
)
from .regions.bedrock_contours import bedrock_elevation


@dataclass
class RockMesh:
    points: np.ndarray
    blocks: dict[int, np.ndarray]
    top_faces: np.ndarray
    bottom_faces: np.ndarray
    report: dict


def tetrahedra_to_hexa8(
    points: np.ndarray, tetrahedra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split every conforming TETRA4 into four face-conforming HEXA8 cells.

    The tetrahedral mesh is only a compact construction topology. Each edge
    and face is refined once, and the four corner-associated bricks share that
    refinement with neighbouring tetrahedra. The production mesh has no tets.
    """
    original = np.asarray(points, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if not len(cells):
        raise ValueError("Cannot convert an empty tetrahedral construction mesh")
    edge_patterns = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    )

    face_patterns = np.asarray(
        ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    )
    edge_rows = np.sort(np.concatenate([
        cells[:, pattern] for pattern in edge_patterns
    ]), axis=1)
    face_rows = np.sort(np.concatenate([
        cells[:, pattern] for pattern in face_patterns
    ]), axis=1)
    edges, edge_inverse = np.unique(edge_rows, axis=0, return_inverse=True)
    faces, face_inverse = np.unique(face_rows, axis=0, return_inverse=True)
    rows = np.arange(len(cells))
    edge_ids = len(original) + edge_inverse.reshape(len(edge_patterns), -1).T
    face_ids = len(original) + len(edges) + face_inverse.reshape(
        len(face_patterns), -1
    ).T
    centre_ids = len(original) + len(edges) + len(faces) + rows
    edge_lookup = np.full((4, 4), -1, dtype=np.int64)
    for index, (left, right) in enumerate(edge_patterns):
        edge_lookup[left, right] = edge_lookup[right, left] = index
    face_lookup = np.full((4, 4, 4), -1, dtype=np.int64)
    for index, (a, b, c) in enumerate(face_patterns):
        for first, second, third in (
            (a, b, c), (a, c, b), (b, a, c),
            (b, c, a), (c, a, b), (c, b, a),
        ):
            face_lookup[first, second, third] = index
    xyz = original[cells]
    other_local = np.asarray([
        [node for node in range(4) if node != vertex]
        for vertex in range(4)
    ])
    hexahedra = np.empty((len(cells) * 4, 8), dtype=np.int64)
    for vertex in range(4):
        local_rows = np.tile(other_local[vertex], (len(cells), 1))
        other_nodes = cells[np.arange(len(cells))[:, None], local_rows]
        vectors = original[other_nodes] - original[cells[:, vertex, None]]
        negative = np.einsum(
            "ij,ij->i",
            np.cross(vectors[:, 0], vectors[:, 1]),
            vectors[:, 2],
        ) < 0.
        local_rows[negative, 1], local_rows[negative, 2] = (
            local_rows[negative, 2].copy(),
            local_rows[negative, 1].copy(),
        )
        first, second, third = local_rows.T
        output_rows = np.arange(vertex, len(cells) * 4, 4)
        hexahedra[output_rows, 0] = cells[:, vertex]
        hexahedra[output_rows, 1] = edge_ids[rows, edge_lookup[vertex, first]]
        hexahedra[output_rows, 2] = face_ids[
            rows, face_lookup[vertex, first, second]
        ]
        hexahedra[output_rows, 3] = edge_ids[rows, edge_lookup[vertex, second]]
        hexahedra[output_rows, 4] = edge_ids[rows, edge_lookup[vertex, third]]
        hexahedra[output_rows, 5] = face_ids[
            rows, face_lookup[vertex, first, third]
        ]
        hexahedra[output_rows, 6] = centre_ids
        hexahedra[output_rows, 7] = face_ids[
            rows, face_lookup[vertex, second, third]
        ]
    result_points = np.vstack((
        original,
        original[edges].mean(axis=1),
        original[faces].mean(axis=1),
        xyz.mean(axis=1),
    ))
    return result_points, hexahedra, len(edges) + len(faces) + len(cells)


def refine_triangular_faces(
    points: np.ndarray, triangles: np.ndarray,
) -> np.ndarray:
    """Return the three conforming QUAD4 children of each refined triangle."""
    source = np.asarray(points, dtype=float)
    rows = np.asarray(triangles, dtype=np.int64)
    lookup = {
        tuple(np.round(point, 8)): index
        for index, point in enumerate(source)
    }

    def node(point: np.ndarray) -> int:
        key = tuple(np.round(point, 8))
        if key not in lookup:
            raise ValueError(
                f"Refined boundary midpoint/centroid is absent at {key}"
            )
        return lookup[key]

    quads = []
    for triangle in rows:
        a, b, c = map(int, triangle)
        ab = node((source[a] + source[b]) / 2.)
        bc = node((source[b] + source[c]) / 2.)
        ca = node((source[c] + source[a]) / 2.)
        centre = node(source[[a, b, c]].mean(axis=0))
        quads.extend((
            (a, ab, centre, ca),
            (b, bc, centre, ab),
            (c, ca, centre, bc),
        ))
    return np.asarray(quads, dtype=np.int64)


def hexa8_scaled_jacobians(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return the minimum corner scaled Jacobian for each VTK HEXA8 cell."""
    values = []
    p = np.asarray(points)[np.asarray(cells)]
    patterns = (
        ((0, 1), (0, 3), (0, 4)),
        ((0, 1), (1, 2), (1, 5)),
        ((3, 2), (1, 2), (2, 6)),
        ((3, 2), (0, 3), (3, 7)),
        ((4, 5), (4, 7), (0, 4)),
        ((4, 5), (5, 6), (1, 5)),
        ((7, 6), (5, 6), (2, 6)),
        ((4, 7), (6, 7), (3, 7)),
    )
    for (a, b), (c, d), (e, f) in patterns:
        first, second, third = p[:, b] - p[:, a], p[:, d] - p[:, c], p[:, f] - p[:, e]
        denominators = (
            np.linalg.norm(first, axis=1)
            * np.linalg.norm(second, axis=1)
            * np.linalg.norm(third, axis=1)
        )
        values.append(
            np.einsum("ij,ij->i", np.cross(first, second), third)
            / np.maximum(denominators, 1e-30)
        )
    return np.min(np.asarray(values), axis=0)


def audit_hexahedral_boundary(cells: np.ndarray) -> int:
    """Validate paired HEXA8 faces and return the exterior quad count."""
    patterns = (
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    )
    ordered = np.concatenate([
        np.asarray(cells)[:, pattern] for pattern in patterns
    ])
    faces = np.sort(ordered, axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Native HEXA8 rock has non-manifold faces")
    exterior = unique[counts == 1]
    oriented = {
        tuple(sorted(face)): face
        for face in ordered
    }
    exterior_faces = np.asarray([oriented[tuple(face)] for face in exterior])
    edge_patterns = ((0, 1), (1, 2), (2, 3), (3, 0))
    edges = np.concatenate([
        np.sort(exterior_faces[:, pair], axis=1) for pair in edge_patterns
    ])
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    if not len(edge_counts) or np.any(edge_counts != 2):
        raise ValueError("Native HEXA8 rock exterior is not closed")
    return len(exterior)


def boundary_loop(faces: np.ndarray, description: str) -> list[int]:
    edge_counts = Counter(
        tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
        for face in faces for index in range(3)
    )
    if any(count > 2 for count in edge_counts.values()):
        raise ValueError(f"{description} is non-manifold")
    neighbours: dict[int, list[int]] = {}
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    if not neighbours or any(len(nodes) != 2 for nodes in neighbours.values()):
        raise ValueError(f"{description} does not have one closed boundary")
    loop = [min(neighbours)]
    previous, current = -1, loop[0]
    while True:
        following = next(node for node in neighbours[current] if node != previous)
        if following == loop[0]:
            break
        if following in loop:
            raise ValueError(f"{description} has an invalid boundary cycle")
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(neighbours):
        raise ValueError(f"{description} has multiple boundary loops")
    return loop


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def separate_coincident_shell_edges(
    points: np.ndarray,
    faces: list[tuple[int, ...]],
    inner_face_count: int,
    lift_m: float = 1e-2,
) -> tuple[np.ndarray, list[tuple[int, ...]], int]:
    edge_owners: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(faces):
        for a, b in zip(face, (*face[1:], face[0])):
            edge_owners.setdefault(
                tuple(sorted((int(a), int(b)))), []
            ).append(face_id)
    bad_edges = [edge for edge, owners in edge_owners.items() if len(owners) == 4]
    if not bad_edges or any(
        len(owners) != 2
        for edge, owners in edge_owners.items()
        if edge not in bad_edges
    ):
        return points, faces, 0
    if any(
        len([owner for owner in edge_owners[edge] if owner < inner_face_count]) != 2
        or len([owner for owner in edge_owners[edge] if owner >= inner_face_count]) != 2
        for edge in bad_edges
    ):
        return points, faces, 0

    inner_coordinates = {
        tuple(np.round(points[node_id], 8))
        for face in faces[:inner_face_count]
        for node_id in face
    }
    outer_nodes = {
        int(node_id)
        for face in faces[inner_face_count:]
        for node_id in face
    }
    outer_copies: dict[int, int] = {}
    separated_points = np.asarray(points).copy()
    for node_id in sorted(outer_nodes):
        if tuple(np.round(points[node_id], 8)) not in inner_coordinates:
            continue
        outer_copies[node_id] = len(separated_points)
        regularized = separated_points[node_id].copy()
        regularized[2] += lift_m
        separated_points = np.vstack((separated_points, regularized))
    separated_faces = [
        tuple(outer_copies.get(node_id, node_id) for node_id in face)
        if face_id >= inner_face_count else face
        for face_id, face in enumerate(faces)
    ]
    return separated_points, separated_faces, len(outer_copies)


def positive_tetrahedron_orientation(
    points: np.ndarray,
    tetrahedra: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    oriented = np.asarray(tetrahedra).copy()
    p = np.asarray(points)[oriented]
    signed = np.einsum(
        "ij,ij->i",
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
        p[:, 3] - p[:, 0],
    ) / 6.
    valid = signed != 0.
    negative = signed[valid] < 0.
    oriented = oriented[valid]
    oriented[negative, 1], oriented[negative, 2] = (
        oriented[negative, 2].copy(),
        oriented[negative, 1].copy(),
    )
    return oriented, int(negative.sum()), valid


def projected_boundary_loop(
    points: np.ndarray, faces: np.ndarray, description: str,
) -> list[int]:
    xy_nodes: dict[tuple[float, float], int] = {}
    for face in faces:
        for raw_node_id in face:
            node_id = int(raw_node_id)
            key = (
                round(float(points[node_id, 0]), 8),
                round(float(points[node_id, 1]), 8),
            )
            previous = xy_nodes.get(key)
            if previous is None or points[node_id, 2] > points[previous, 2]:
                xy_nodes[key] = node_id

    projected_faces = []
    for face in faces:
        projected = [
            xy_nodes[(
                round(float(points[int(node_id), 0]), 8),
                round(float(points[int(node_id), 1]), 8),
            )]
            for node_id in face
        ]
        if len(set(projected)) < 3:
            continue
        coordinates = points[projected, :2]
        if abs(cross2(
            coordinates[1] - coordinates[0],
            coordinates[2] - coordinates[0],
        )) <= 1e-12:
            continue
        projected_faces.append(projected)
    if not projected_faces:
        raise ValueError(f"{description} has no projected area")
    return boundary_loop(
        np.asarray(projected_faces, dtype=np.int64),
        description,
    )


def triangulate_polygon(
    points: np.ndarray, polygon: list[int], *, remove_collinear: bool = False,
) -> list[tuple[int, int, int]]:
    polygon = list(polygon)
    if remove_collinear:
        changed = True
        while changed and len(polygon) > 3:
            changed = False
            for index, current in enumerate(polygon):
                previous, following = polygon[index - 1], polygon[(index + 1) % len(polygon)]
                a, b, c = points[[previous, current, following], :2]
                if abs(cross2(b - a, c - b)) <= 1e-12:
                    del polygon[index]
                    changed = True
                    break
    result: list[tuple[int, int, int]] = []
    clockwise = sum(
        cross2(points[node, :2], points[following, :2])
        for node, following in zip(polygon, polygon[1:] + polygon[:1])
    ) < 0
    while len(polygon) > 3:
        clipped = False
        for index, current in enumerate(polygon):
            previous = polygon[index - 1]
            following = polygon[(index + 1) % len(polygon)]
            a, b, c = points[[previous, current, following], :2]
            turn = cross2(b - a, c - b)
            if not (turn < -1e-10 if clockwise else turn > 1e-10):
                continue
            pa, pb, pc = (a, b, c)

            def contains(point: np.ndarray) -> bool:
                signs = (
                    cross2(pb - pa, point - pa),
                    cross2(pc - pb, point - pb),
                    cross2(pa - pc, point - pc),
                )
                return (
                    all(value >= -1e-10 for value in signs)
                    if not clockwise else
                    all(value <= 1e-10 for value in signs)
                )

            triangle = (previous, current, following)
            if any(contains(points[node, :2]) for node in polygon if node not in triangle):
                continue
            result.append(triangle)
            del polygon[index]
            clipped = True
            break
        if not clipped:
            raise ValueError("Foundation footprint cannot be triangulated")
    result.append(tuple(polygon))
    return result


def sample_closed_polygon(
    points: np.ndarray, polygon: list[int], samples: int,
    contact_loop: list[int],
) -> tuple[np.ndarray, list[int]]:
    polygon = list(polygon)
    vertices = points[polygon, :2]
    lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    cumulative = np.concatenate(([0.], np.cumsum(lengths)))
    perimeter = cumulative[-1]
    polygon_area = sum(
        cross2(vertex, following)
        for vertex, following in zip(vertices, np.roll(vertices, -1, axis=0))
    )
    contact_xy = points[contact_loop, :2]
    contact_area = sum(
        cross2(vertex, following)
        for vertex, following in zip(contact_xy, np.roll(contact_xy, -1, axis=0))
    )
    direction = -1. if polygon_area * contact_area < 0 else 1.
    sampled = []
    for point in contact_xy:
        candidates = []
        for index, (start, end) in enumerate(zip(
            vertices, np.roll(vertices, -1, axis=0)
        )):
            segment = end - start
            fraction = float(np.clip(
                np.dot(point - start, segment) / np.dot(segment, segment),
                0., 1.,
            ))
            projected = start + fraction * segment
            candidates.append((
                np.linalg.norm(projected - point),
                (cumulative[index] + fraction * lengths[index]) % perimeter,
                segment,
            ))
        _, parameter, segment = min(candidates, key=lambda candidate: candidate[0])
        index = int(np.searchsorted(cumulative, parameter, side="right") - 1)
        point_on_ring = (
            vertices[index]
            + (parameter - cumulative[index]) / lengths[index] * segment
        )
        normal = np.asarray([segment[1], -segment[0]], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-30)
        if polygon_area < 0:
            normal *= -1
        sampled.append(point_on_ring + 1e-2 * normal)
    return np.asarray(sampled), list(range(samples))


def conforming_collar(
    contact: list[int], terrain: list[int], points: np.ndarray,
    blocked: list[tuple[int, int, int]] | None = None,
) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    if len(contact) != len(terrain):
        raise ValueError("Foundation contact collar rings have different lengths")
    for index, (a, b) in enumerate(zip(contact, contact[1:] + contact[:1])):
        c, d = terrain[index], terrain[(index + 1) % len(terrain)]

        def area(face: tuple[int, int, int]) -> float:
            p, q, r = points[list(face)]
            return float(np.linalg.norm(np.cross(q - p, r - p)) / 2)

        first = ((a, b, d), (a, d, c))
        second = ((a, b, c), (b, d, c))
        options = (first, second)
        valid = [
            faces for faces in options
            if all(area(face) > 1e-12 for face in faces)
        ]
        if valid:
            def overlap_count(faces):
                score = 0
                for face in faces:
                    set_face = set(face)
                    a, b, c = points[list(face)]
                    normal = np.cross(b - a, c - a)
                    for other in blocked or ():
                        shared = set_face & set(other)
                        if len(shared) != 2 or np.linalg.norm(normal) <= 1e-15:
                            continue
                        edge = tuple(shared)
                        third_face = next(node for node in face if node not in shared)
                        third_other = next(node for node in other if node not in shared)
                        p, q = points[list(edge)]
                        direction = q - p
                        v1 = points[third_face] - p
                        v2 = points[third_other] - p
                        v1 -= direction * np.dot(v1, direction) / np.dot(direction, direction)
                        v2 -= direction * np.dot(v2, direction) / np.dot(direction, direction)
                        normal_other = np.cross(
                            points[other[1]] - points[other[0]],
                            points[other[2]] - points[other[0]])
                        if (np.linalg.norm(normal_other) > 1e-15
                                and abs(np.dot(normal, normal_other)) /
                                (np.linalg.norm(normal) * np.linalg.norm(normal_other)) > 1 - 1e-7
                                and np.dot(v1, v2) > 1e-10):
                            score += 1
                return score
            result.extend(min(valid, key=lambda faces: (
                overlap_count(faces), -sum(area(face) for face in faces))))
        elif a == c and b == d:
            continue
        else:
            candidates = [
                face for faces in options for face in faces if area(face) > 1e-12
            ]
            if candidates:
                result.append(max(candidates, key=area))
            else:
                continue
    return result


def build_graded_rock(
    config: dict, concrete: np.ndarray, foundation_faces: list[tuple[int, ...]],
    stream_margin_m: float | None = None, bank_margin_m: float | None = None,
    bottom_z_m: float | None = None,
    *, foundation=None,
) -> RockMesh:
    rock = config["geometry"]["bedrock"]
    domain = rock.get("domain", {})
    settings = rock["mesh"]
    # One TETRA4 is only a construction primitive: its four conforming HEXA8
    # children have roughly half the edge length.  Meshing this coarser
    # auxiliary topology is what keeps the constrained 100k-wall cavity
    # stable while preserving the requested production grading.
    construction_settings = {
        **settings,
        "near_size_m": 2. * float(settings["near_size_m"]),
        "far_size_m": 2. * float(settings["far_size_m"]),
    }
    lego_foundation = (
        foundation is not None
        and foundation.report.get("seat_definition") == "lego_interval_course"
    )
    seed_concrete = concrete
    seed_faces = foundation_faces
    contact_loop: list[int] | None = None
    collar_terrain_ids: list[int] | None = None
    collar_contact_terrain_ids: list[int] | None = None
    contact_terrain_map: dict[int, int] = {}
    exposed: list[tuple[int, ...]] = []
    if lego_foundation:
        # Projection is used only to locate the surrounding terrain ring.
        # The near field is rebuilt below as a 3-D collar between this ring and
        # the exact multi-level foundation contact loop.
        polygon = projected_boundary_loop(
            np.asarray(foundation.points),
            np.asarray(foundation.terrain_faces),
            "LEGO foundation projected footprint",
        )
        contact_loop = list(polygon)
        terrain_xy = np.asarray(foundation.points)[contact_loop, :2]
        collar_contact_terrain_indices = list(range(len(contact_loop)))
        terrain_points = np.column_stack((
            terrain_xy,
            [
                bedrock_elevation(
                    rock["contours"], 80. * math.atan2(x, y)
                )
                for x, y in terrain_xy
            ],
        ))
        collar_terrain_ids = list(range(
            len(foundation.points), len(foundation.points) + len(terrain_points)
        ))
        collar_contact_terrain_ids = [
            collar_terrain_ids[index] for index in collar_contact_terrain_indices
        ]
        seed_concrete = np.vstack((foundation.points, terrain_points))
        from scipy.spatial import Delaunay
        triangulation = Delaunay(seed_concrete[collar_terrain_ids, :2])
        seed_faces = [
            tuple(collar_terrain_ids[index] for index in simplex)
            for simplex in triangulation.simplices
        ]
    columns, wedges, report = seated_bedrock_geometry(
        rock["contours"], seed_concrete, seed_faces,
        stream_margin_m if stream_margin_m is not None else domain.get("stream_margin_m", STREAM_MARGIN_M),
        bank_margin_m if bank_margin_m is not None else domain.get("bank_margin_m", BANK_MARGIN_M),
        bottom_z_m=bottom_z_m if bottom_z_m is not None else domain.get("bottom_z_m", -31.),
        element_size_m=settings["near_size_m"], resample_surface=False,
        preserve_seat_edges=False,
    )
    count = len(columns) // 2
    triangles = wedges[:, :3]
    seat_count = report["seat_triangles"]
    edge_counts = Counter(
        tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
        for tri in triangles for i in range(3)
    )
    perimeter = [edge for edge, total in edge_counts.items() if total == 1]
    if any(total not in (1, 2) for total in edge_counts.values()):
        raise ValueError("Non-manifold rock top")
    if lego_foundation:
        neighbours: dict[int, set[int]] = {}
        for a, b in perimeter:
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)
        components: list[set[int]] = []
        unseen = set(neighbours)
        while unseen:
            stack = [next(iter(unseen))]
            component = set()
            while stack:
                node_id = stack.pop()
                if node_id in component:
                    continue
                component.add(node_id)
                stack.extend(neighbours[node_id] - component)
            unseen -= component
            components.append(component)
        def extent(component):
            xy = np.asarray(columns[list(component), :2])
            return (xy[:, 0].max() - xy[:, 0].min()) * (
                xy[:, 1].max() - xy[:, 1].min())
        outer = max(components, key=extent)
        perimeter = [edge for edge in perimeter if edge[0] in outer and edge[1] in outer]
    # Retain the faceted terrain exactly; only the seat triangles are mesh-locked.
    facets = [tuple(int(n) + count for n in tri) for tri in triangles]
    top_count = len(facets)
    inner_facets: list[tuple[int, ...]] = []
    if foundation is not None:
        shell_points_list = columns.tolist()
        lookup = {tuple(np.round(p, 8)): i for i, p in enumerate(columns)}
        mapping = []
        for p in foundation.points:
            key = tuple(np.round(p, 8))
            if key not in lookup:
                lookup[key] = len(shell_points_list)
                shell_points_list.append(p.tolist())
            mapping.append(lookup[key])
        cut = [tuple(mapping[int(n)] for n in tri) for tri in foundation.contact_faces]
        exposed = [
            tuple(mapping[int(n)] for n in tri)
            for tri in foundation.exposed_rock_faces
        ]
        if lego_foundation:
            roof = [tuple(mapping[int(n)] for n in tri)
                    for tri in foundation.top_faces]
            inner_facets = [*cut, *roof]
            facets = inner_facets + facets
            seat_count = len(inner_facets)
        else:
            facets = cut + exposed + facets[seat_count:]
            seat_count = len(cut)
        top_count = len(facets)
        report["seat_triangles"] = seat_count
        report["rock_volume_m3"] -= (
            foundation.report["volume_m3"]
            + foundation.report.get("open_excavation_volume_m3", 0.)
        )
        report["foundation"] = foundation.report
        columns = np.asarray(shell_points_list)
    # Mesh the planar bottom without artificial radial seams.
    neighbours: dict[int, list[int]] = {}
    for a, b in perimeter:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    if not neighbours or any(len(nodes) != 2 for nodes in neighbours.values()):
        raise ValueError("Rock perimeter must be a single closed loop")
    bottom_loop = [min(neighbours)]
    previous = -1
    current = bottom_loop[0]
    while True:
        following = next(n for n in neighbours[current] if n != previous)
        if following == bottom_loop[0]:
            break
        if following in bottom_loop:
            raise ValueError("Invalid rock perimeter cycle")
        bottom_loop.append(following)
        previous, current = current, following
    if len(bottom_loop) != len(neighbours):
        raise ValueError("Multiple rock perimeter loops are not supported")
    shell_points = columns
    bottom_start = len(facets)
    bottom_points = np.asarray(shell_points, dtype=float)[bottom_loop]
    bottom_centre = np.mean(bottom_points, axis=0)
    bottom_centre_index = len(shell_points)
    shell_points = np.vstack((np.asarray(shell_points), bottom_centre))
    facets.extend(
        (int(a), int(b), bottom_centre_index)
        for a, b in zip(bottom_loop, (*bottom_loop[1:], bottom_loop[0]))
    )
    bottom_end = len(facets)
    for a, b in perimeter:
        facets.extend(((a, b, b + count), (a, b + count, a + count)))

    def orient_shell(faces: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        rows = [tuple(int(node) for node in face) for face in faces]
        edges: dict[tuple[int, int], list[tuple[int, bool]]] = {}
        for face_id, face in enumerate(rows):
            for a, b in zip(face, (*face[1:], face[0])):
                key = tuple(sorted((a, b)))
                edges.setdefault(key, []).append((face_id, a < b))
        flip: list[bool | None] = [None] * len(rows)
        for root in range(len(rows)):
            if flip[root] is not None:
                continue
            flip[root] = False
            queue = [root]
            for face_id in queue:
                face = rows[face_id]
                for a, b in zip(face, (*face[1:], face[0])):
                    for other, other_direction in edges[tuple(sorted((a, b)))]:
                        if other == face_id:
                            continue
                        desired = ((a < b) == other_direction) != flip[face_id]
                        if flip[other] is None:
                            flip[other] = desired
                            queue.append(other)
                        elif flip[other] != desired:
                            raise ValueError("LEGO rock shell is non-orientable")
        return [tuple(reversed(face)) if flip[index] else face
                for index, face in enumerate(rows)]

    shell_edges: Counter[tuple[int, int]] = Counter()
    for face in facets:
        for a, b in zip(face, (*face[1:], face[0])):
            shell_edges[tuple(sorted((int(a), int(b))))] += 1
    if foundation is not None and contact_terrain_map:
        patched = []
        for (a, b), count in list(shell_edges.items()):
            if count != 1 or a not in contact_terrain_map or b not in contact_terrain_map:
                continue
            c, d = contact_terrain_map[a], contact_terrain_map[b]
            for face in ((a, b, d), (a, d, c)):
                if len(set(face)) == 3:
                    p, q, r = np.asarray(shell_points)[list(face)]
                    if np.linalg.norm(np.cross(q - p, r - p)) > 1e-12:
                        patched.append(face)
        facets.extend(patched)
        shell_edges = Counter()
        for face in facets:
            for a, b in zip(face, (*face[1:], face[0])):
                shell_edges[tuple(sorted((int(a), int(b))))] += 1
    missing = [edge for edge, count in shell_edges.items() if count == 1]
    adjacency: dict[int, set[int]] = {}
    for a, b in missing:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    closed = set()
    for a in sorted(adjacency):
        for b in sorted(adjacency[a]):
            for c in sorted(adjacency[a] & adjacency[b]):
                if len({a, b, c}) != 3:
                    continue
                triangle = tuple(sorted((a, b, c)))
                if triangle not in closed:
                    closed.add(triangle)
    if closed:
        closures = [tuple(triangle) for triangle in sorted(closed)]
        facets = facets[:top_count] + closures + facets[top_count:]
        top_count += len(closures)
        bottom_start += len(closures)
        bottom_end += len(closures)
        shell_edges = Counter()
        for face in facets:
            for a, b in zip(face, (*face[1:], face[0])):
                shell_edges[tuple(sorted((int(a), int(b))))] += 1
    if lego_foundation:
        shell_points, facets, regularized_nodes = separate_coincident_shell_edges(
            np.asarray(shell_points),
            facets,
            seat_count,
            lift_m=1e-2,
        )
        if regularized_nodes:
            shell_edges = Counter(
                tuple(sorted((int(a), int(b))))
                for face in facets
                for a, b in zip(face, (*face[1:], face[0]))
            )
            report["coincident_shell_nodes_regularized"] = regularized_nodes
            report["coincident_shell_regularization_m"] = 1e-2
    if shell_edges and any(count != 2 for count in shell_edges.values()):
        bad = [(edge, count) for edge, count in shell_edges.items() if count != 2]
        if foundation is not None:
            geometric_bad = Counter(
                tuple(sorted(tuple(np.round(shell_points[node], 8)) for node in edge))
                for edge, _ in bad
            )
            cut_edges = {
                tuple(sorted((int(face[i]), int(face[(i + 1) % len(face)]))))
                for face in cut for i in range(len(face))
            }
            exterior_edges = {
                tuple(sorted((int(face[i]), int(face[(i + 1) % len(face)]))))
                for face in facets[len(cut) + len(exposed):top_count]
                for i in range(len(face))
            }
            raise ValueError(
                f"LEGO rock shell has {len(bad)} open/nonmanifold edges "
                f"(duplicate geometric boundaries={sum(n > 1 for n in geometric_bad.values())}; "
                f"cut/exterior/top={len(cut_edges)}/{len(exterior_edges)}/{top_count}; "
                f"interface edges={len(cut_edges & exterior_edges)}; "
                f"examples {[(edge, count, np.asarray(shell_points)[list(edge)].tolist()) for edge, count in bad[:6]]})")

    for face in facets:
        a, b, c = np.asarray(shell_points)[list(face)]
        if len(face) != 3 or np.linalg.norm(np.cross(b - a, c - a)) > 1e-12:
            continue
        pairs = list(zip((face[0], face[1], face[2]),
                         (face[1], face[2], face[0])))
        longest = max(pairs, key=lambda pair:
                      np.linalg.norm(shell_points[pair[0]] - shell_points[pair[1]]))
        middle = next(node for node in face if node not in longest)
        direction = shell_points[longest[1], :2] - shell_points[longest[0], :2]
        normal = np.asarray([-direction[1], direction[0]], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-30)
        shell_points[middle, :2] += normal * 1e-6
    facets = orient_shell(list(facets))
    if foundation is not None:
        def signed_shell_volume(rows):
            points = np.asarray(shell_points)[list(rows)]
            return float(np.sum(np.einsum(
                "ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2])
            )) / 6.0)
        if signed_shell_volume(facets[:seat_count]) * signed_shell_volume(facets[seat_count:]) >= 0:
            facets[:seat_count] = [
                tuple(reversed(face)) for face in facets[:seat_count]
            ]
    shell_adjacency: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(facets):
        for a, b in zip(face, (*face[1:], face[0])):
            shell_adjacency.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
    overlaps = []
    for edge, owners in shell_adjacency.items():
        if len(owners) != 2:
            continue
        first, second = owners
        other = [
            next(node for node in facets[index] if node not in edge)
            for index in (first, second)
        ]
        a, b = np.asarray(shell_points)[list(edge)]
        direction = b - a
        projections = []
        normals = []
        for node in other:
            vector = np.asarray(shell_points)[node] - a
            projections.append(vector - direction * np.dot(vector, direction) / np.dot(direction, direction))
        for face_index in (first, second):
            p, q, r = np.asarray(shell_points)[list(facets[face_index])]
            normal = np.cross(q - p, r - p)
            normals.append(normal / np.linalg.norm(normal))
        if (abs(float(np.dot(normals[0], normals[1]))) > 1.0 - 1e-7
                and np.dot(projections[0], projections[1]) > 1e-10):
            overlaps.append((edge, first, second, other))
    if overlaps:
        edge, first, second, other = overlaps[0]
        raise ValueError(
            f"LEGO rock shell has overlapping coplanar facets at edge {edge}: "
            f"{first}:{facets[first]} and {second}:{facets[second]}, "
            f"coordinates={np.asarray(shell_points)[list(facets[first])+list(facets[second])].tolist()}")
    for face_index, face in enumerate(facets):
        if len(face) != 3:
            raise ValueError(f"LEGO rock shell face {face_index} is not triangular")
        a, b, c = np.asarray(shell_points)[list(face)]
        if np.linalg.norm(np.cross(b - a, c - a)) <= 1e-12:
            raise ValueError(
                f"LEGO rock shell face {face_index} is geometrically degenerate: "
                f"{face} at {[a.tolist(), b.tolist(), c.tolist()]}")
    if gmsh.isInitialized():
        raise RuntimeError("Rock mesher requires its own Gmsh session")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("graded_bedrock")
        used = sorted({n for face in facets for n in face})
        vertices = {n: gmsh.model.geo.addPoint(*shell_points[n]) for n in used}
        curves = {}
        seat_edges = {
            tuple(sorted((face[i], face[(i + 1) % 3])))
            for face in facets[:seat_count] for i in range(3)
        }
        surfaces = []
        for face in facets:
            loop = []
            for a, b in zip(face, (*face[1:], face[0])):
                key = tuple(sorted((a, b)))
                if key not in curves:
                    curves[key] = gmsh.model.geo.addLine(vertices[key[0]], vertices[key[1]])
                loop.append(curves[key] if a < b else -curves[key])
            wire = gmsh.model.geo.addCurveLoop(loop)
            surfaces.append(gmsh.model.geo.addPlaneSurface([wire]))
        gmsh.model.geo.synchronize()
        for edge in seat_edges:
            gmsh.model.mesh.setTransfiniteCurve(curves[edge], 2)
        for face, surface in zip(facets[:seat_count], surfaces[:seat_count]):
            gmsh.model.mesh.setTransfiniteSurface(surface, cornerTags=[vertices[n] for n in face])
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", surfaces[:seat_count])
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 2)
        threshold = gmsh.model.mesh.field.add("Threshold")
        for key, value in {
            "InField": distance, "SizeMin": construction_settings["near_size_m"],
            "SizeMax": construction_settings["far_size_m"],
            "DistMin": construction_settings["near_distance_m"],
            "DistMax": construction_settings["far_distance_m"],
        }.items():
            gmsh.model.mesh.field.setNumber(threshold, key, value)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        for key, value in {
            "Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeFromCurvature": 0,
            "Mesh.MeshSizeExtendFromBoundary": 0,
            "Mesh.MeshSizeMin": construction_settings["near_size_m"],
            "Mesh.MeshSizeMax": construction_settings["far_size_m"],
            "Mesh.Algorithm3D": 4 if foundation is not None else 10,
            "Mesh.ElementOrder": 1,
            "Mesh.OptimizeNetgen": 0,
            "Mesh.Optimize": 0,
            "Mesh.RandomSeed": 1,
            "Mesh.MeshOnlyEmpty": 1,
        }.items():
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(2)
        if not lego_foundation:
            # The volume sees one closed discrete shell.
            closed_shell = gmsh.model.addDiscreteEntity(2)
            shell_nodes = np.concatenate([
                gmsh.model.mesh.getElements(2, tag)[2][0] for tag in surfaces
            ])
            gmsh.model.mesh.addElementsByType(closed_shell, 2, [], shell_nodes)
            loops = [gmsh.model.geo.addSurfaceLoop([closed_shell])]
        else:
            inner_shell = gmsh.model.addDiscreteEntity(2)
            inner_nodes = np.concatenate([
                gmsh.model.mesh.getElements(2, tag)[2][0]
                for tag in surfaces[:seat_count]
            ])
            gmsh.model.mesh.addElementsByType(inner_shell, 2, [], inner_nodes)
            outer_shell = gmsh.model.addDiscreteEntity(2)
            outer_nodes = np.concatenate([
                gmsh.model.mesh.getElements(2, tag)[2][0]
                for tag in surfaces[seat_count:]
            ])
            gmsh.model.mesh.addElementsByType(outer_shell, 2, [], outer_nodes)
            loops = [
                gmsh.model.geo.addSurfaceLoop([outer_shell]),
                -gmsh.model.geo.addSurfaceLoop([inner_shell]),
            ]
        volume = gmsh.model.geo.addVolume(loops)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize(
            "Relocate3D",
            force=True,
            niter=10,
            dimTags=[(3, volume)],
        )
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_tags)
        sorted_tags = node_tags[order]
        points = np.asarray(coordinates).reshape(-1, 3)[order]

        def connectivity(dim, tags, kind, width):
            parts = []
            for tag in tags:
                types, _, nodes = gmsh.model.mesh.getElements(dim, tag)
                for element_type, members in zip(types, nodes):
                    if element_type != kind:
                        raise ValueError(f"Unexpected Gmsh element type {element_type}")
                    parts.append(np.searchsorted(sorted_tags, members).reshape(-1, width))
            return np.vstack(parts)

        tetrahedra = connectivity(3, [volume], 4, 4)
        surface_faces = connectivity(2, surfaces[:top_count], 2, 3)
        bottom_faces = connectivity(2, surfaces[bottom_start:bottom_end], 2, 3)
        shell_faces = connectivity(2, surfaces, 2, 3)
        actual_seat = connectivity(2, surfaces[:seat_count], 2, 3)
        expected = {
            tuple(sorted(tuple(np.round(shell_points[n], 9)) for n in face))
            for face in facets[:seat_count]
        }
        actual = {
            tuple(sorted(tuple(np.round(points[n], 9)) for n in face))
            for face in actual_seat
        }
        if actual != expected or len(actual_seat) != seat_count:
            raise ValueError(
                f"Gmsh changed the locked wall/rock interface: expected {seat_count}, "
                f"got {len(actual_seat)}; missing {list(expected - actual)[:1]}"
            )
        expected_points = {
            tuple(np.round(shell_points[n], 9)): shell_points[n]
            for face in facets[:seat_count] for n in face
        }
        coordinate_error = max(
            np.linalg.norm(points[n] - expected_points[tuple(np.round(points[n], 9))])
            for n in np.unique(actual_seat)
        )
        tetrahedra, reoriented_cells, valid_tetrahedra = positive_tetrahedron_orientation(
            points, tetrahedra
        )
        p = points[tetrahedra]
        signed = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                           p[:, 3] - p[:, 0]) / 6.
        if foundation is None and not np.isclose(
            signed.sum(), report["rock_volume_m3"], rtol=1e-8
        ):
            raise ValueError("Graded mesh changed the rock volume")
        if lego_foundation:
            report["rock_volume_m3"] = float(signed.sum())
        raw_qualities = gmsh.model.mesh.getElementQualities(
            gmsh.model.mesh.getElements(3, volume)[1][0], "minSICN",
        )
        raw_quality_array = np.asarray(raw_qualities)
        if raw_quality_array.size == 1:
            raw_quality_array = np.repeat(
                raw_quality_array, valid_tetrahedra.size
            )
        qualities = raw_quality_array[valid_tetrahedra]
        # The cavity must remain topologically closed. Positive but small-quality
        # construction cells are valid conversion input; deleting them exposes
        # internal faces and creates holes in the full outer domain.
        if len(qualities) and np.all(qualities <= 1e-10):
            raise ValueError("All graded rock tetrahedra are numerically degenerate")
        quality_valid = np.ones(len(qualities), dtype=bool)
        removed_low_quality_cells = int((~quality_valid).sum())
        tetrahedra = tetrahedra[quality_valid]
        signed = signed[quality_valid]
        qualities = qualities[quality_valid]
        if not len(tetrahedra):
            raise ValueError("All graded rock tetrahedra are numerically degenerate")
        audit_tetrahedral_boundary(
            tetrahedra, shell_faces, allow_extra_recovered_boundary=True
        )
        source_tetrahedra = len(tetrahedra)
        points, hexahedra, refinement_nodes = tetrahedra_to_hexa8(
            points, tetrahedra,
        )
        quad_top_faces = refine_triangular_faces(points, surface_faces)
        quad_bottom_faces = refine_triangular_faces(points, bottom_faces)
        quad_seat_faces = refine_triangular_faces(points, actual_seat)
        exterior_hexa8_faces = audit_hexahedral_boundary(hexahedra)
        report.update(
            mesh_method="tetrahedral_construction_to_native_hexa8",
            mesh_grading=dict(settings),
            construction_settings=dict(construction_settings),
            construction_tetrahedra=source_tetrahedra,
            bedrock_cells=len(hexahedra),
            solver_cell_type="hexahedron",
            generated_hexa8_refinement_nodes=refinement_nodes,
            exterior_hexa8_faces=exterior_hexa8_faces,
            seat_quad4_faces=len(quad_seat_faces),
            top_quad4_faces=len(quad_top_faces),
            bottom_quad4_faces=len(quad_bottom_faces),
            minimum_cell_volume_m3=float(signed.min() / 4.),
            minimum_scaled_jacobian=float(np.min(
                hexa8_scaled_jacobians(points, hexahedra)
            )),
            seat_faces_verified=len(actual_seat),             surface_triangles=len(surface_faces),
            seat_max_coordinate_difference_m=max(
                report.get("seat_max_coordinate_difference_m", 0.), float(coordinate_error),
            ),
            reoriented_bedrock_cells=reoriented_cells,
            removed_zero_volume_bedrock_cells=int((~valid_tetrahedra).sum()),
            removed_low_quality_bedrock_cells=removed_low_quality_cells,
            mesher_version=gmsh.__version__, vertical_layers=None,
            surface_edge_size_policy=(
                "Coarse construction triangles with a locked seat; "
                "production exterior is conforming QUAD4"
            ),
            surface_row_axis=None,
            maximum_exterior_surface_edge_m=float(max(
                np.linalg.norm(points[surface_faces[:, i]] - points[surface_faces[:, (i + 1) % 3]],
                               axis=1).max() for i in range(3)
            )),
            conformity_audit=(
                "Every production HEXA8 interior face is paired and the "
                "construction seat remains locked to the original coordinates"
            ),
            spacing_samples=spacing_summary(
                points, hexahedra, actual_seat, settings,
            ),
        )
        if report["minimum_scaled_jacobian"] < .01:
            warnings.warn(
                f"Rock minimum scaled Jacobian is {report['minimum_scaled_jacobian']:.6g}; "
                "inspect constrained-boundary element quality before engineering use",
                stacklevel=2,
            )
        return RockMesh(
            points, {5: hexahedra},
            quad_top_faces, quad_bottom_faces, report,
        )
    finally:
        gmsh.finalize()


def audit_tetrahedral_boundary(
    tetrahedra: np.ndarray, shell: np.ndarray,
    *, allow_extra_recovered_boundary: bool = False,
) -> None:
    faces = np.sort(np.concatenate([
        tetrahedra[:, indices] for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    ]), axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    expected = np.sort(shell, axis=1)
    expected = expected[np.lexsort(expected.T[::-1])]
    actual = unique[counts == 1]
    if allow_extra_recovered_boundary:
        expected_keys = set(map(tuple, expected))
        actual_keys = set(map(tuple, actual))
        valid_shell = (
            not np.any(counts > 2)
            and expected_keys <= actual_keys
        )
    else:
        valid_shell = (
            not np.any(counts > 2)
            and np.array_equal(actual, expected)
        )
    if not valid_shell:
        raise ValueError("Rock has non-manifold, unpaired or hanging volume faces")
    audited_shell = (
        np.asarray(list(set(map(tuple, actual))), dtype=np.int64)
        if allow_extra_recovered_boundary else shell
    )
    edges, counts = np.unique(np.sort(np.concatenate([
        audited_shell[:, indices] for indices in ((0, 1), (1, 2), (2, 0))
    ]), axis=1), axis=0, return_counts=True)
    if not len(edges) or np.any(counts != 2):
        raise ValueError("Rock exterior is not a closed two-manifold")


def spacing_summary(points, cells, seat, settings) -> dict:
    """Bounded deterministic sample; distance to seat vertices approximates its surface."""
    selected = cells[::max(1, len(cells) // 3000)]
    centres = points[selected].mean(axis=1)
    reference = points[np.unique(seat)]
    distances = np.concatenate([
        np.sqrt(np.min(np.sum((chunk[:, None] - reference[None]) ** 2, axis=2), axis=1))
        for chunk in np.array_split(centres, max(1, len(centres) // 100))
    ])
    edge_pairs = (
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        if selected.shape[1] == 4 else
        ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    )
    lengths = np.array([
        np.linalg.norm(points[selected[:, a]] - points[selected[:, b]], axis=1)
        for a, b in edge_pairs
    ]).T
    result = {
        "sampled_cells": len(selected),
        "distance_method": "Cell centroid to nearest locked seat vertex (not exact surface distance)",
    }
    for name, mask in (
        ("near", distances <= settings["near_distance_m"]),
        ("far", distances >= settings["far_distance_m"]),
    ):
        values = lengths[mask]
        result[name] = {
            "sampled_cells": int(mask.sum()),
            "median_edge_m": float(np.median(values)) if values.size else None,
            "edge_10th_percentile_m": float(np.quantile(values, .1)) if values.size else None,
            "edge_90th_percentile_m": float(np.quantile(values, .9)) if values.size else None,
        }
    return result
