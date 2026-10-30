"""Preserve topology, tags and volume under deterministic subdivision."""
from types import SimpleNamespace
import unittest

import numpy as np
from dolfinx import fem, mesh
import ufl

from dam_mesh import FACES, import_structure, selected_wall_cells, source_cell_tags, subdivide
from no_tension_fem import NoTensionMaterial


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

    def split_tetrahedra(self):
        points = np.asarray([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
        cells = [(4, tuple(range(start, start + 4))) for start in (0, 4, 8)]
        return SimpleNamespace(
            nodes=np.vstack((points + [2., 0., 0.], points, points)),
            cells=cells, region_ids=[2, 1, 1],
            boundaries=[(10 + row, tuple(cell[i] for i in face))
                        for row, (_, cell) in enumerate(cells) for face in FACES[4]],
        )

    def test_native_source_rows_preserve_reordering_and_split_identities(self):
        structure = self.split_tetrahedra()
        imported = import_structure(structure, 1e-12)
        tags = source_cell_tags(imported)
        selected = selected_wall_cells(tags, imported.cell_tags, [1])
        self.assertEqual(selected.sum(), 1)
        domain = imported.domain
        for cell, row in zip(tags.indices, tags.values):
            source_nodes = domain.geometry.input_global_indices[domain.geometry.dofmaps[0][cell]]
            np.testing.assert_array_equal(np.sort(source_nodes), structure.cells[row][1])
            self.assertEqual(selected[cell], row == 1)
        self.assertFalse(selected[tags.find(2)[0]])
        displacement = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
        displacement.interpolate(lambda x: .001 * x)
        material = NoTensionMaterial(displacement, 35e9, .2, selected)
        stress = material.stress.x.array.reshape(-1, 3, 3)[material.tensor_dofs]
        np.testing.assert_allclose(stress[selected], 0, atol=1e-7)
        self.assertTrue(np.all(np.trace(stress[~selected], axis1=1, axis2=2) > 0))

    def test_source_rows_survive_repeated_native_refinement(self):
        imported = import_structure(self.split_tetrahedra(), 1e-12)
        domain, regions = imported.domain, imported.cell_tags
        sources = source_cell_tags(imported)
        for _ in range(2):
            domain.topology.create_entities(1)
            domain, parent_cell, _ = mesh.refine(domain, option=mesh.RefinementOption.parent_cell)
            sources = mesh.transfer_meshtag(sources, domain, parent_cell)
            regions = mesh.transfer_meshtag(regions, domain, parent_cell)
            selected = selected_wall_cells(sources, regions, [1])
            self.assertGreater(selected.sum(), 1)
            self.assertTrue(np.all(regions.values[selected] == 1))
            dx = ufl.Measure("dx", domain=domain, subdomain_data=sources)
            for row in (0, 1, 2):
                self.assertAlmostEqual(fem.assemble_scalar(fem.form(1. * dx(row))), 1 / 6, places=12)

    def test_subdivision_transfers_one_source_row_to_all_children(self):
        imported = import_structure(self.cube(), 1e-12)
        sources = source_cell_tags(imported)
        np.testing.assert_array_equal(sources.values, 0)
        with self.assertRaisesRegex(ValueError, "wall concrete"):
            selected_wall_cells(sources, imported.cell_tags, [0])

    def test_invalid_or_foreign_source_selection_is_rejected(self):
        imported = import_structure(self.split_tetrahedra(), 1e-12)
        sources = source_cell_tags(imported)
        for selected in ([], [1, 1], [True], [1.], [-1], [3], [0, 1]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                selected_wall_cells(sources, imported.cell_tags, selected)
        other = import_structure(self.split_tetrahedra(), 1e-12)
        with self.assertRaisesRegex(ValueError, "same tetrahedral mesh"):
            selected_wall_cells(sources, other.cell_tags, [1])
        incomplete = mesh.meshtags(imported.domain, 3, sources.indices[:1], sources.values[:1])
        with self.assertRaisesRegex(ValueError, "Complete local"):
            selected_wall_cells(incomplete, imported.cell_tags, [1])
        imported.parent_ids[0] = 100
        with self.assertRaisesRegex(ValueError, "provenance"):
            source_cell_tags(imported)


if __name__ == "__main__":
    unittest.main()
