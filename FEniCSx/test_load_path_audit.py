"""Distinguish coordinate conformity from physical load transfer."""
import unittest

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
import ufl

from load_path_audit import match_checkpoint_geometry, region_connection
from run import assert_fracture_ready


class LoadPathAuditTests(unittest.TestCase):
    def test_real_bonded_runner_records_hold_before_importing_or_solving(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from run import LOAD_CASE, run_baseline, run_fracture

        for phase_baseline in (False, True):
            with self.subTest(phase_baseline=phase_baseline), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                config = output / "config.json"
                config.write_bytes(LOAD_CASE.read_bytes())
                with patch("dam_mesh.import_structure") as importer, self.assertRaisesRegex(
                    RuntimeError, "Cracking is on hold",
                ):
                    if phase_baseline:
                        run_baseline(config, output, {"test": True}, phase=True)
                    else:
                        run_fracture(config, output, {"test": True}, output / "unused.pvtu")
                importer.assert_not_called()
                manifest = json.loads(next(output.glob("dam_phase_field/*/manifest.json")).read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertIn("on hold", manifest["error"])
                self.assertFalse(list(output.glob("dam_phase_field/*/phase_field_report.json")))

    def test_bonded_production_fracture_is_explicitly_on_hold(self):
        from types import SimpleNamespace

        with self.assertRaisesRegex(RuntimeError, "on hold"):
            assert_fracture_ready(SimpleNamespace(boundaries=[(1, (0, 1, 2))]))
        assert_fracture_ready(SimpleNamespace(boundaries=[(8, (0, 1, 2))]))

    def test_bonded_vector_elasticity_transfers_wall_only_load_through_plinth(self):
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from dam_elasticity import ground_supports, solve
        from dam_mesh import FACES, import_structure
        import test_dam_mesh

        structure = test_dam_mesh.DamMeshTests().cube()
        structure.nodes = np.vstack((structure.nodes, structure.nodes[4:8] + [0, 0, 1]))
        structure.cells.append((5, (4, 5, 6, 7, 8, 9, 10, 11)))
        structure.region_ids = [1, 2]
        # Only the wall crown carries pressure; the lower block has no body or water load.
        face_owners = {}
        for _, cell in structure.cells:
            for pattern in FACES[5]:
                face = tuple(cell[i] for i in pattern)
                face_owners.setdefault(tuple(sorted(face)), []).append(face)
        structure.boundaries = []
        for owners in face_owners.values():
            if len(owners) == 1:
                face = owners[0]
                z = structure.nodes[list(face), 2]
                tag = 1 if np.allclose(z, 0) else 4 if np.allclose(z, 2) else 7
                structure.boundaries.append((tag, face))
        structure.material = SimpleNamespace(youngs_modulus_pa=1e7, poissons_ratio=0.2, density_kg_m3=0.0)
        structure.loads = SimpleNamespace(
            gravity_z_m_s2=-10.0, water_density_kg_m3=1000.0,
            tailwater_head_m=0.0, overflow_head_m=1.0,
        )
        structure.foundation_support = SimpleNamespace(
            spring_x_n_per_m3=1e6, spring_y_n_per_m3=1e6, spring_z_n_per_m3=1e6,
        )
        # A very large obsolete wall-ground stiffness must not change the bonded result.
        structure.plinth_support = SimpleNamespace(
            spring_x_n_per_m3=1e15, spring_y_n_per_m3=1e15, spring_z_n_per_m3=1e15,
        )
        imported = import_structure(structure, 1e-12)
        springs, connection = ground_supports(imported, structure)
        self.assertEqual(set(springs), {1})
        self.assertEqual(connection, "bonded_intact_provisional")
        with tempfile.TemporaryDirectory() as directory:
            displacement, sigma, report = solve(
                imported, structure, {"relative_tolerance": 1e-10, "maximum_iterations": 500}, Path(directory),
            )
        np.testing.assert_allclose(report["applied_force_n"], [0, 0, -10000], atol=1e-7)
        np.testing.assert_allclose(report["ground_spring_reactions_n"]["1"], [0, 0, 10000], atol=1e-5)
        self.assertEqual(set(report["ground_spring_reactions_n"]), {"1"})
        self.assertLess(report["relative_force_balance_error"], 1e-9)
        self.assertLess(report["relative_true_residual"], 1e-10)
        self.assertTrue(np.all(displacement[:8, 2] < 0))
        plinth = imported.cell_tags.find(1)
        dofs = sigma.function_space.dofmap.list[plinth, 0]
        stresses = sigma.x.array.reshape(-1, 3, 3)[dofs]
        np.testing.assert_allclose(stresses[:, 2, 2].mean(), -10000, rtol=0.03)

    def test_coincident_interface_without_shared_ids_is_not_bonded(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1],
                           [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        report = region_connection(points, [[0, 1, 2, 3], [4, 5, 6, 7]], [1, 2])
        self.assertEqual(report["shared_vertex_ids"], 0)
        self.assertEqual(report["coincident_coordinate_count_rounded_1e_9_m"], 3)
        self.assertFalse(report["direct_continuous_p1_displacement_transfer"])
        bonded = region_connection(points, [[0, 1, 2, 3], [0, 1, 2, 7]], [1, 2])
        self.assertEqual(bonded["shared_vertex_ids"], 3)
        self.assertTrue(bonded["direct_continuous_p1_displacement_transfer"])

    def test_missing_region_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly plinth"):
            region_connection(np.zeros((4, 3)), [[0, 1, 2, 3]], [2])

    def test_checkpoint_mapping_preserves_duplicate_interface_nodes_after_reordering(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1],
                           [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        permutation = np.array([7, 0, 4, 1, 5, 2, 6, 3])
        inverse = np.argsort(permutation)
        actual = match_checkpoint_geometry(
            points, cells, points[permutation], inverse[cells[::-1]], np.array([2, 1]),
        )
        np.testing.assert_array_equal(actual, permutation)
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            match_checkpoint_geometry(points, cells, points + 1, cells, np.array([1, 2]))

    def test_split_and_bonded_blocks_have_different_load_paths(self):
        """Scalar analogue; unit wall load must reach the correct supports."""
        import basix.ufl

        original = mesh.create_box(MPI.COMM_SELF, [[0, 0, -1], [1, 1, 1]], [1, 1, 2])
        for bonded in (False, True):
            with self.subTest(bonded=bonded):
                self.check_block_load_path(original, bonded, basix.ufl)

    def check_block_load_path(self, original, bonded, element_module):
        xyz = original.geometry.x.copy()
        cells = original.geometry.dofmaps[0].copy()
        if not bonded:
            wall = xyz[cells].mean(axis=1)[:, 2] > 0
            interface = np.flatnonzero(np.isclose(xyz[:, 2], 0))
            duplicates = np.arange(len(xyz), len(xyz) + len(interface))
            for old, new in zip(interface, duplicates):
                cells[wall] = np.where(cells[wall] == old, new, cells[wall])
            xyz = np.vstack((xyz, xyz[interface]))
        domain = mesh.create_mesh(
            MPI.COMM_SELF, cells.astype(np.int64),
            ufl.Mesh(element_module.element("Lagrange", "tetrahedron", 1, shape=(3,))), xyz,
        )
        domain.topology.create_connectivity(3, 3)
        domain.topology.create_connectivity(2, 3)
        count = domain.topology.index_map(3).size_local
        ids = np.arange(count, dtype=np.int32)
        regions = np.where(mesh.compute_midpoints(domain, 3, ids)[:, 2] > 0, 2, 1).astype(np.int32)
        tags = mesh.meshtags(domain, 3, ids, regions)
        boundary = mesh.locate_entities_boundary(domain, 2, lambda x: np.ones(x.shape[1], dtype=bool))
        boundary.sort()
        centers = mesh.compute_midpoints(domain, 2, boundary)
        values = np.full(len(boundary), 7, dtype=np.int32)
        values[np.isclose(centers[:, 2], -1)] = 1
        adjacency = domain.topology.connectivity(2, 3)
        for index in np.flatnonzero(np.isclose(centers[:, 2], 0)):
            if regions[adjacency.links(int(boundary[index]))[0]] == 2:
                values[index] = 8
        facets = mesh.meshtags(domain, 2, boundary, values)
        space = fem.functionspace(domain, ("Lagrange", 1))
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        dx = ufl.Measure("dx", domain=domain, subdomain_data=tags)
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)
        stiffness = 7.0
        problem = LinearProblem(
            ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
            + stiffness * trial * test * (ds(1) + ds(8)),
            test * dx(2), petsc_options_prefix="split_support_test_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        displacement = problem.solve()
        plinth = fem.locate_dofs_topological(space, 3, tags.find(1))
        wall_dofs = fem.locate_dofs_topological(space, 3, tags.find(2))
        self.assertGreater(np.max(np.abs(displacement.x.array[wall_dofs])), 0)
        plinth_reaction = fem.assemble_scalar(fem.form(stiffness * displacement * ds(1)))
        wall_reaction = fem.assemble_scalar(fem.form(stiffness * displacement * ds(8)))
        load = fem.assemble_scalar(fem.form(1.0 * dx(2)))
        self.assertAlmostEqual(plinth_reaction + wall_reaction, load, places=12)
        if bonded:
            self.assertGreater(len(np.intersect1d(plinth, wall_dofs)), 0)
            self.assertAlmostEqual(plinth_reaction, load, places=12)
            self.assertAlmostEqual(wall_reaction, 0, places=12)
        else:
            self.assertEqual(len(np.intersect1d(plinth, wall_dofs)), 0)
            np.testing.assert_allclose(displacement.x.array[plinth], 0, atol=1e-14)
            self.assertAlmostEqual(plinth_reaction, 0, places=12)
            self.assertAlmostEqual(wall_reaction, load, places=12)


if __name__ == "__main__":
    unittest.main()
