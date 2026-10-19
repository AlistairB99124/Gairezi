"""Split the combined hexahedral/prismatic structure into conforming linear tetrahedra.

No nodes are added. Each quadrilateral face is cut along the diagonal through its
lowest global node id, and each volume cell is coned from its lowest node id, so
neighbouring cells (and boundary faces) always agree on the shared diagonal.
"""
from __future__ import annotations

from dataclasses import replace

from Elmer.regions.combined_structure import CombinedStructure

GMSH_TETRAHEDRON = 4
GMSH_HEXAHEDRON = 5
GMSH_PRISM = 6

_CELL_FACES = {
    GMSH_TETRAHEDRON: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    GMSH_HEXAHEDRON: ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    GMSH_PRISM: ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
}


def split_face(face: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Return triangles for a face, cutting quads through the lowest node id and keeping orientation."""
    if len(face) == 3:
        return [tuple(face)]
    if len(face) != 4:
        raise ValueError(f"Unsupported face with {len(face)} nodes")
    start = face.index(min(face))
    a, b, c, d = (face[(start + offset) % 4] for offset in range(4))
    return [(a, b, c), (a, c, d)]


def _signed_volume(nodes, tetrahedron) -> float:
    p0, p1, p2, p3 = (nodes[node_id] for node_id in tetrahedron)
    u = [p1[i] - p0[i] for i in range(3)]
    v = [p2[i] - p0[i] for i in range(3)]
    w = [p3[i] - p0[i] for i in range(3)]
    return (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    ) / 6.0


def split_cell(nodes, element_type: int, cell: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
    if element_type == GMSH_TETRAHEDRON:
        return [tuple(cell)]
    if element_type not in _CELL_FACES:
        raise ValueError(f"Unsupported volume element type {element_type}")
    apex = min(cell)
    tetrahedra = []
    for pattern in _CELL_FACES[element_type]:
        face = tuple(cell[index] for index in pattern)
        if apex in face:
            continue
        for a, b, c in split_face(face):
            tetrahedron = (apex, a, b, c)
            volume = _signed_volume(nodes, tetrahedron)
            if volume == 0.0:
                raise ValueError(f"Degenerate tetrahedron produced from cell {cell}")
            if volume < 0.0:
                tetrahedron = (apex, a, c, b)
            tetrahedra.append(tetrahedron)
    return tetrahedra


def tetrahedralize_structure(mesh: CombinedStructure) -> CombinedStructure:
    cells: list[tuple[int, tuple[int, ...]]] = []
    region_ids: list[int] = []
    for (element_type, cell), region_id in zip(mesh.cells, mesh.region_ids):
        for tetrahedron in split_cell(mesh.nodes, element_type, cell):
            cells.append((GMSH_TETRAHEDRON, tetrahedron))
            region_ids.append(region_id)
    boundaries = [
        (boundary_id, triangle)
        for boundary_id, face in mesh.boundaries
        for triangle in split_face(tuple(face))
    ]
    return replace(mesh, cells=cells, boundaries=boundaries, region_ids=region_ids)
