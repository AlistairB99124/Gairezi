"""Preserve topology, tags and volume under deterministic subdivision."""
from types import SimpleNamespace
import unittest

import numpy as np

from dam_mesh import FACES, import_structure, subdivide


class DamMeshTests(unittest.TestCase):
    def cube(self):
        points = np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                             (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], dtype=float)
        return SimpleNamespace(nodes=points, cells=[(5, tuple(range(8)))], region_ids=[2],
                               boundaries=[(i + 1, face) for i, face in enumerate(FACES[5])])

    def test_cube_import(self):
        imported = import_structure(self.cube(), 1e-12)
        self.assertEqual(len(imported.tetrahedra), 6)
        self.assertAlmostEqual(imported.volumes.sum(), 1)
        self.assertEqual(len(imported.facet_tags.indices), 12)
        self.assertEqual(set(imported.cell_tags.values), {2})

    def test_prism_import(self):
        structure = SimpleNamespace(
            nodes=np.asarray([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                              (0, 0, 1), (1, 0, 1), (0, 1, 1)], dtype=float),
            cells=[(6, tuple(range(6)))], region_ids=[1],
            boundaries=[(i + 1, face) for i, face in enumerate(FACES[6])]
        )
        imported = import_structure(structure, 1e-12)
        self.assertEqual(len(imported.tetrahedra), 3)
        self.assertAlmostEqual(imported.volumes.sum(), 0.5)

    def test_coincident_disconnected_cubes_remain_disconnected(self):
        structure = self.cube()
        structure.nodes = np.vstack((structure.nodes, structure.nodes))
        structure.cells.append((5, tuple(range(8, 16))))
        structure.region_ids.append(1)
        structure.boundaries += [(9, tuple(i + 8 for i in face)) for face in FACES[5]]
        imported = import_structure(structure, 1e-12)
        self.assertEqual(imported.report["imported_nodes"], 16)
        self.assertEqual(len(imported.facet_tags.indices), 24)

    def test_degenerate_cells_rejected(self):
        structure = self.cube()
        structure.nodes[:, 2] = 0
        with self.assertRaisesRegex(ValueError, "degenerate"):
            subdivide(structure.nodes, structure.cells, structure.region_ids, structure.boundaries)

    def test_adjacent_hexes_have_conforming_shared_face(self):
        structure = self.cube()
        structure.nodes = np.vstack((structure.nodes, [(2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1)]))
        second = (1, 8, 9, 2, 5, 10, 11, 6)
        structure.cells.append((5, second))
        structure.region_ids.append(2)
        structure.boundaries = [(tag, face) for tag, face in structure.boundaries if set(face) != {1, 2, 5, 6}]
        structure.boundaries += [
            (7, tuple(second[i] for i in pattern)) for pattern in FACES[5]
            if set(second[i] for i in pattern) != {1, 2, 5, 6}
        ]
        imported = import_structure(structure, 1e-12)
        self.assertEqual(len(imported.facet_tags.indices), 20)
        self.assertAlmostEqual(imported.volumes.sum(), 2)
        structure.nodes[6, 0] += 0.2
        refined = import_structure(structure, 1e-12)
        self.assertEqual(len(refined.facet_tags.indices), 20)
        self.assertGreater(refined.report["added_nodes"], 0)
        self.assertLess(refined.report["maximum_parent_cell_relative_volume_change"], 1e-12)

    def test_warped_prism_uses_extra_nodes_and_preserves_parent_volume(self):
        structure = SimpleNamespace(
            nodes=np.asarray([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                              (0.2, 0, 1), (1.2, 0.1, 2), (0, 1, 1)], dtype=float),
            cells=[(6, tuple(range(6)))], region_ids=[1],
            boundaries=[(i + 1, face) for i, face in enumerate(FACES[6])]
        )
        imported = import_structure(structure, 1e-12)
        self.assertGreater(imported.report["added_nodes"], 0)
        self.assertLess(imported.report["maximum_parent_cell_relative_volume_change"], 1e-12)
        np.testing.assert_array_equal(imported.points[:6], structure.nodes)


if __name__ == "__main__":
    unittest.main()
