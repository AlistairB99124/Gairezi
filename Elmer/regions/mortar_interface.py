"""Build the thin elastic mortar layer between the plinth and wall."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from .model_groups import FACE_GROUP_IDS
from .plinth import _hex_orientation


MORTAR_THICKNESS_M = 0.01
MORTAR_WALL_BOUNDARY_ID = 8


@dataclass
class MortarMesh:
    nodes: list[tuple[float, float, float]]
    cells: list[tuple[int, tuple[int, ...]]]
    boundaries: list[tuple[int, tuple[int, ...]]]


def build_mortar_interface(wall) -> MortarMesh:
    nodes: list[tuple[float, float, float]] = []
    coordinate_nodes: dict[tuple[float, float, float], int] = {}

    def add_node(point: tuple[float, float, float]) -> int:
        key = tuple(round(value, 9) for value in point)
        if key not in coordinate_nodes:
            coordinate_nodes[key] = len(nodes) + 1
            nodes.append(point)
        return coordinate_nodes[key]

    cells: list[tuple[int, tuple[int, ...]]] = []
    wall_base_faces = [
        face
        for boundary_id, face in wall.boundaries
        if boundary_id == FACE_GROUP_IDS["FOUNDATION"]
    ]
    if not wall_base_faces:
        raise ValueError("Wall has no foundation faces for the mortar interface")
    for face in wall_base_faces:
        if len(face) != 4:
            raise ValueError("Mortar interface requires quadrilateral wall base faces")
        top = tuple(add_node(wall.nodes[node_id - 1]) for node_id in face)
        bottom = tuple(
            add_node((x_m, y_m, z_m - MORTAR_THICKNESS_M))
            for x_m, y_m, z_m in (wall.nodes[node_id - 1] for node_id in face)
        )
        cell = top + bottom
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

    top_face_keys = {tuple(sorted(cell[:4])) for _, cell in cells}
    bottom_face_keys = {tuple(sorted(cell[4:])) for _, cell in cells}
    boundaries = []
    tolerance = 1.0e-6
    for key, count in face_counts.items():
        if count != 1:
            continue
        face = oriented_faces[key]
        coordinates = [nodes[node_id - 1] for node_id in face]
        radii_m = [math.hypot(x_m, y_m) for x_m, y_m, _ in coordinates]
        chainages_m = [math.atan2(x_m, y_m) for x_m, y_m, _ in coordinates]
        if key in top_face_keys:
            boundary_id = MORTAR_WALL_BOUNDARY_ID
        elif key in bottom_face_keys:
            boundary_id = FACE_GROUP_IDS["FOUNDATION"]
        elif max(radii_m) - min(radii_m) < tolerance:
            boundary_id = FACE_GROUP_IDS["DOWNSTREAM"]
        elif max(chainages_m) - min(chainages_m) < tolerance:
            boundary_id = FACE_GROUP_IDS["LEFT_ABUTMENT"]
        else:
            boundary_id = FACE_GROUP_IDS["OTHER_EXTERIOR"]
        boundaries.append((boundary_id, face))
    return MortarMesh(nodes, cells, boundaries)