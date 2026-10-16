from collections import Counter
from pathlib import Path

import pytest

from Elmer.regions.combined_structure import build_combined_structure
from Elmer.regions.tetrahedral import (
    GMSH_HEXAHEDRON,
    GMSH_PRISM,
    GMSH_TETRAHEDRON,
    _signed_volume,
    split_cell,
    split_face,
    tetrahedralize_structure,
)

ROOT = Path(__file__).resolve().parents[1]

CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def test_hexahedron_becomes_six_positive_tetrahedra_filling_its_volume():
    tetrahedra = split_cell(CUBE, GMSH_HEXAHEDRON, tuple(range(8)))
    volumes = [_signed_volume(CUBE, tet) for tet in tetrahedra]
    assert len(tetrahedra) == 6
    assert all(volume > 0 for volume in volumes)
    assert sum(volumes) == pytest.approx(1.0)


def test_prism_becomes_three_positive_tetrahedra_filling_its_volume():
    nodes = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)]
    tetrahedra = split_cell(nodes, GMSH_PRISM, (3, 4, 5, 0, 1, 2))
    volumes = [_signed_volume(nodes, tet) for tet in tetrahedra]
    assert len(tetrahedra) == 3
    assert all(volume > 0 for volume in volumes)
    assert sum(volumes) == pytest.approx(0.5)


def test_quad_split_uses_lowest_node_diagonal_and_keeps_orientation():
    assert split_face((7, 3, 9, 5)) == [(3, 9, 5), (3, 5, 7)]


def test_combined_structure_tetrahedralization_is_conforming():
    hex_mesh = build_combined_structure(ROOT)
    mesh = tetrahedralize_structure(hex_mesh)

    expected = sum(6 if t == GMSH_HEXAHEDRON else 3 for t, _ in hex_mesh.cells)
    assert len(mesh.cells) == expected == len(mesh.region_ids)
    assert {t for t, _ in mesh.cells} == {GMSH_TETRAHEDRON}
    assert len(mesh.nodes) == len(hex_mesh.nodes)
    assert Counter(mesh.region_ids) == Counter(
        r for (t, _), r in zip(hex_mesh.cells, hex_mesh.region_ids) for _ in range(6 if t == GMSH_HEXAHEDRON else 3)
    )

    face_count: Counter[tuple[int, ...]] = Counter()
    for _, tet in mesh.cells:
        for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            face_count[tuple(sorted(tet[i] for i in face))] += 1
    assert max(face_count.values()) == 2
    exterior = {face for face, count in face_count.items() if count == 1}
    assert all(tuple(sorted(tri)) in exterior for _, tri in mesh.boundaries)
