"""Loaded intact baseline, heterogeneous loss, unchanged loads and explicit failure."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import ufl

from tensile_loss_event import EventSolverOptions, solve_event, solve_source_event, solve_bound_event
from three_body_model import materials, support_form
from dam_mesh import import_structure
from interface_springs import InterfaceSprings
from test_three_body import column, foundation_column
from dam_elasticity import solve as solve_elastic_baseline
from fracture_faces import source_vector_values


class TensileLossEventTests(unittest.TestCase):
    def fixture(self, divisions=3, traction=1e6):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, divisions, divisions, divisions)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        domain.topology.create_connectivity(0, 3)
        vertices = np.arange(domain.topology.index_map(0).size_local, dtype=np.int32)
        left = mesh.locate_entities(domain, 0, lambda x: np.isclose(x[0], 0.))
        bcs = [fem.dirichletbc(0., fem.locate_dofs_topological(space.sub(component), 0, ids), space.sub(component))
               for component, ids in ((0, left), (1, vertices), (2, vertices))]
        right = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 1.))
        tags = mesh.meshtags(domain, 2, right, np.ones(len(right), dtype=np.int32))
        ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
        force = fem.Constant(domain, float(traction))
        cells = domain.topology.index_map(3)
        count = cells.size_local + cells.num_ghosts
        centres = mesh.compute_midpoints(domain, 3, np.arange(count, dtype=np.int32))
        selected = np.all((centres > .3) & (centres < .7), axis=1)
        return space, selected, bcs, lambda test: force * test[0] * ds(1), force

    def test_selected_interior_release_redistributes_unchanged_tensile_load(self):
        for divisions in (3, 4):
            with self.subTest(divisions=divisions):
                space, selected, bcs, load, force = self.fixture(divisions)
                self.assertGreater(selected.sum(), 0)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "event"
                    event = solve_event(
                        space, young_pa=35e9, poisson=0., selected_cells=selected,
                        boundary_conditions=bcs, external_load=load, options=EventSolverOptions(), output=output,
                    )
                    saved = json.loads((output / "event_report.json").read_text())
                    self.assertEqual(saved["status"], "passed_reduced_tensile_loss_event")
                    self.assertTrue((output / "displacement.pvd").is_file())
                    self.assertTrue((output / "post_event_material.pvd").is_file())
                    self.assertTrue((output / "intact_state.npz").is_file())
                    self.assertTrue((output / "post_event_state.npz").is_file())
                    with np.load(output / "native_inputs.npz", allow_pickle=False) as inputs:
                        np.testing.assert_array_equal(inputs["selected_local_cells"], selected)
                        np.testing.assert_array_equal(inputs["young_pa"], np.full(len(selected), 35e9))
                        np.testing.assert_array_equal(inputs["poisson"], np.zeros(len(selected)))
                        np.testing.assert_allclose(inputs["assembled_external_force_n"].reshape(-1, 3).sum(axis=0),
                                                   [1e6, 0., 0.], atol=1e-8)
                    series = ET.parse(output / "displacement.pvd").findall(".//DataSet")
                    self.assertEqual(len(series), 2)
                self.assertEqual(float(force.value), 1e6)
                expected = space.tabulate_dof_coordinates()[:, 0] * 1e6 / 35e9
                np.testing.assert_allclose(event.intact_displacement.x.array.reshape(-1, 3)[:, 0], expected, atol=1e-12)
                rows = event.report["stages"]
                self.assertEqual([row["stage"] for row in rows], ["intact_loaded", "post_event"])
                self.assertGreater(rows[0]["maximum_selected_principal_stress_pa"], 9e5)
                self.assertLessEqual(rows[1]["maximum_selected_principal_stress_pa"], .01)
                self.assertGreater(rows[1]["solver_iterations"], 0)
                for row in rows:
                    np.testing.assert_allclose(row["support_reaction_n"], [-1e6, 0., 0.], atol=.02)
                    self.assertLess(row["scaled_free_residual"], 1e-8)
                    self.assertLess(row["scaled_force_imbalance"], 1e-8)
                    np.testing.assert_allclose(row["applied_force_n"], [1e6, 0., 0.], atol=1e-8)
                self.assertGreater(len(event.report["post_event_iterations"]), 1)
                # The same load does more work after tensile capacity is lost.
                self.assertGreater(rows[1]["maximum_displacement_m"], rows[0]["maximum_displacement_m"])
                self.assertFalse(event.report["solver_ready"])

    def test_compressive_equilibrium_retains_original_compliance(self):
        space, selected, bcs, load, _ = self.fixture(2, traction=-1e6)
        selected[:] = True
        event = solve_event(space, young_pa=35e9, poisson=0., selected_cells=selected,
                            boundary_conditions=bcs, external_load=load, options=EventSolverOptions())
        np.testing.assert_allclose(event.intact_displacement.x.array,
                                   event.post_event_displacement.x.array, atol=1e-12)
        for row in event.report["stages"]:
            np.testing.assert_allclose(row["support_reaction_n"], [1e6, 0., 0.], atol=.01)

    def test_loss_of_all_tensile_load_paths_fails_with_persistent_diagnostics(self):
        space, selected, bcs, load, _ = self.fixture(2)
        selected[:] = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            with self.assertRaisesRegex(RuntimeError, "no equilibrium or opening verdict accepted"):
                solve_event(space, young_pa=35e9, poisson=0., selected_cells=selected,
                            boundary_conditions=bcs, external_load=load,
                            options=EventSolverOptions(maximum_iterations=5), output=output)
            report = json.loads((output / "event_report.json").read_text())
            self.assertEqual(report["status"], "failed_reduced_tensile_loss_event")
            self.assertEqual(report["stages"][0]["stage"], "intact_loaded")
            self.assertGreater(report["stages"][0]["solver_reason"], 0)
            self.assertLess(report["stages"][0]["scaled_free_residual"], 1e-8)
            self.assertLessEqual(report["stages"][-1]["solver_reason"], 0)
            self.assertTrue(report["post_event_iterations"])
            self.assertFalse((output / "displacement.pvd").exists())
            self.assertTrue((output / "intact_state.npz").is_file())
            self.assertFalse((output / "post_event_state.npz").exists())

    def test_full_vector_boundary_indices_preserve_reaction_checks(self):
        space, selected, _, load, _ = self.fixture(2, traction=-1e6)
        domain = space.mesh
        left = mesh.locate_entities(domain, 0, lambda x: np.isclose(x[0], 0.))
        bc = fem.dirichletbc(np.zeros(3), fem.locate_dofs_topological(space, 0, left), space)
        selected[:] = True
        event = solve_event(space, young_pa=35e9, poisson=0., selected_cells=selected,
                            boundary_conditions=[bc], external_load=load, options=EventSolverOptions())
        np.testing.assert_allclose(event.intact_displacement.x.array, event.post_event_displacement.x.array,
                                   atol=1e-12)
        for row in event.report["stages"]:
            self.assertLess(row["scaled_free_residual"], 1e-8)
            np.testing.assert_allclose(row["support_reaction_n"], [1e6, 0., 0.], atol=.01)

    def test_existing_three_body_material_fields_preserve_layered_compressive_response(self):
        space, selected, bcs, load, _ = self.fixture(4, traction=-1e6)
        domain = space.mesh
        indices = np.arange(len(selected), dtype=np.int32)
        centres = mesh.compute_midpoints(domain, 3, indices)
        regions = np.where(centres[:, 0] < .5, 1, 2).astype(np.int32)
        imported = SimpleNamespace(domain=domain, cell_tags=mesh.meshtags(domain, 3, indices, regions))
        # Synthetic verification materials, using the existing canonical helper.
        structure = SimpleNamespace(three_body_config={"material": {
            "concrete": {"density": 2400., "youngs_modulus": 35e9, "poisson_ratio": 0.},
            "bedrock": {"density": 2700., "youngs_modulus": 70e9, "poisson_ratio": 0.},
        }})
        coefficients = materials(imported, structure)
        selected = regions == 1
        event = solve_event(
            space, young_pa=coefficients["youngs_modulus"], poisson=coefficients["poisson_ratio"],
            selected_cells=selected, boundary_conditions=bcs, external_load=load, options=EventSolverOptions(),
        )
        x = space.tabulate_dof_coordinates()[:, 0]
        exact = -1e6 * (np.minimum(x, .5) / 35e9 + np.maximum(x - .5, 0.) / 70e9)
        np.testing.assert_allclose(event.intact_displacement.x.array.reshape(-1, 3)[:, 0], exact, atol=1e-12)
        np.testing.assert_allclose(event.post_event_displacement.x.array.reshape(-1, 3)[:, 0], exact, atol=1e-12)
        stress = event.material.stress.x.array.reshape(-1, 3, 3)[event.material.tensor_dofs]
        np.testing.assert_allclose(stress[:, 0, 0], -1e6, atol=.01)
        self.assertEqual(len(event.report["material_groups"]), 2)
        np.testing.assert_array_equal(event.material.selected, regions == 1)

    def test_heterogeneous_loss_preserves_other_region_elastic_tension(self):
        space, selected, bcs, load, _ = self.fixture(4)
        domain = space.mesh
        centres = mesh.compute_midpoints(domain, 3, np.arange(len(selected), dtype=np.int32))
        young = np.where(centres[:, 0] < .5, 35e9, 70e9)
        selected &= centres[:, 0] < .5
        self.assertTrue(selected.any())
        event = solve_event(space, young_pa=young, poisson=0., selected_cells=selected,
                            boundary_conditions=bcs, external_load=load, options=EventSolverOptions())
        stress = event.material.stress.x.array.reshape(-1, 3, 3)[event.material.tensor_dofs]
        self.assertLess(np.linalg.eigvalsh(stress[selected])[:, -1].max(), .01)
        self.assertGreater(np.linalg.eigvalsh(stress[centres[:, 0] >= .5])[:, -1].max(), 9e5)
        self.assertLess(event.report["stages"][-1]["scaled_free_residual"], 1e-8)
        np.testing.assert_allclose(event.report["stages"][-1]["support_reaction_n"], [-1e6, 0., 0.], atol=.02)

    def spring_fixture(self, traction):
        structure = column()
        imported = import_structure(structure, 1e-12)
        domain = imported.domain
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        bottom = fem.locate_dofs_topological(space, 2, imported.facet_tags.find(10))
        bc = fem.dirichletbc(np.zeros(3), bottom, space)
        springs = InterfaceSprings(imported, structure, space)
        coefficients = materials(imported, structure)
        top = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 2.))
        tags = mesh.meshtags(domain, 2, top, np.ones(len(top), dtype=np.int32))
        ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
        return imported, space, [bc], springs, coefficients, lambda test: traction * test[2] * ds(1)

    def test_existing_mortar_springs_preserve_analytical_compressive_event(self):
        imported, space, bcs, springs, coefficients, load = self.spring_fixture(-1e4)
        selected = imported.cell_tags.values == 1
        with tempfile.TemporaryDirectory() as directory:
            event = solve_event(
                space, young_pa=coefficients["youngs_modulus"], poisson=coefficients["poisson_ratio"],
                selected_cells=selected, boundary_conditions=bcs, external_load=load,
                interface_springs=springs, options=EventSolverOptions(), output=Path(directory) / "event",
            )
            with np.load(Path(directory) / "event" / "interface_springs.npz", allow_pickle=False) as saved:
                offsets, columns, values = springs.matrix.getValuesCSR()
                np.testing.assert_array_equal(saved["indptr"], offsets)
                np.testing.assert_array_equal(saved["indices"], columns)
                np.testing.assert_array_equal(saved["data"], values)
                np.testing.assert_array_equal(saved["displacement_cell_dofs"], space.dofmap.list)
            with np.load(Path(directory) / "event" / "native_inputs.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["young_pa"], np.where(selected, 1e7, 2e7), rtol=0., atol=0.)
        z = space.tabulate_dof_coordinates()[:, 2]
        # Source identity distinguishes the two coincident z=1 interface lips.
        body_at_dof = np.zeros(space.dofmap.index_map.size_local, dtype=int)
        for cell, body in zip(imported.cell_tags.indices, imported.cell_tags.values):
            body_at_dof[space.dofmap.list[cell]] = body
        expected = np.where(body_at_dof == 2, -1e4 * z / 2e7,
                            -1e4 * (1 / 2e7 + 1 / 3e6 + (z - 1) / 1e7))
        np.testing.assert_allclose(event.intact_displacement.x.array.reshape(-1, 3)[:, 2], expected, atol=1e-12)
        np.testing.assert_allclose(event.post_event_displacement.x.array, event.intact_displacement.x.array, atol=1e-12)
        for row in event.report["stages"]:
            np.testing.assert_allclose(row["support_reaction_n"], [0., 0., 1e4], atol=1e-6)
            self.assertAlmostEqual(row["interface_spring_energy_j"], .5 * 1e8 / 3e6, places=9)
            self.assertLess(row["scaled_force_imbalance"], 1e-8)
        self.assertIn("wall_bedrock", event.report["interface_springs"])
        springs.matrix.destroy()

    def test_mortar_operator_remains_identical_during_selected_tensile_loss(self):
        imported, space, bcs, springs, coefficients, load = self.spring_fixture(1e4)
        selected = np.zeros(len(imported.cell_tags.values), dtype=bool)
        selected[imported.cell_tags.find(1)[0]] = True
        event = solve_event(
            space, young_pa=coefficients["youngs_modulus"], poisson=coefficients["poisson_ratio"],
            selected_cells=selected, boundary_conditions=bcs, external_load=load,
            interface_springs=springs, options=EventSolverOptions(),
        )
        for row in event.report["stages"]:
            np.testing.assert_allclose(row["support_reaction_n"], [0., 0., -1e4], atol=1e-5)
        self.assertGreater(event.report["stages"][0]["maximum_selected_principal_stress_pa"], 9e3)
        self.assertLess(event.report["stages"][1]["maximum_selected_principal_stress_pa"], 1e-5)
        self.assertGreater(event.report["stages"][1]["solver_iterations"], 0)
        np.testing.assert_allclose(springs.matrix.getValuesCSR()[2], springs._operator.data, rtol=0., atol=0.)
        springs.matrix.destroy()

    def test_constrained_spring_dofs_preserve_nonzero_imposed_displacement(self):
        for imposed in (0., -.003):
            imported, space, bcs, springs, coefficients, load = self.spring_fixture(-1e4)
            interface = fem.locate_dofs_topological(space, 2, imported.facet_tags.find(11))
            bcs.append(fem.dirichletbc(np.array([0., 0., imposed]), interface, space))
            event = solve_event(
                space, young_pa=coefficients["youngs_modulus"], poisson=coefficients["poisson_ratio"],
                selected_cells=imported.cell_tags.values == 1, boundary_conditions=bcs, external_load=load,
                interface_springs=springs, options=EventSolverOptions(),
            )
            z = space.tabulate_dof_coordinates()[:, 2]
            body = np.zeros(space.dofmap.index_map.size_local, dtype=int)
            for cell, region in zip(imported.cell_tags.indices, imported.cell_tags.values):
                body[space.dofmap.list[cell]] = region
            rock_top = 3e6 / (2e7 + 3e6) * imposed
            expected = np.where(body == 2, rock_top * z, imposed - 1e4 / 1e7 * (z - 1.))
            np.testing.assert_allclose(event.intact_displacement.x.array.reshape(-1, 3)[:, 2], expected, atol=1e-12)
            np.testing.assert_allclose(event.post_event_displacement.x.array.reshape(-1, 3)[:, 2], expected, atol=1e-12)
            for row in event.report["stages"]:
                np.testing.assert_allclose(row["support_reaction_n"], [0., 0., 1e4], atol=1e-7)
                self.assertLess(row["scaled_free_residual"], 1e-8)
            np.testing.assert_allclose(springs.matrix.getValuesCSR()[2], springs._operator.data, rtol=0., atol=0.)
            springs.matrix.destroy()

    def test_constrained_mortar_nonlinear_release_keeps_imposed_displacement(self):
        imported, space, bcs, springs, coefficients, load = self.spring_fixture(1e4)
        interface = fem.locate_dofs_topological(space, 2, imported.facet_tags.find(11))
        bcs.append(fem.dirichletbc(np.array([0., 0., .001]), interface, space))
        selected = np.zeros(len(imported.cell_tags.values), dtype=bool)
        selected[imported.cell_tags.find(1)[0]] = True
        event = solve_event(
            space, young_pa=coefficients["youngs_modulus"], poisson=coefficients["poisson_ratio"],
            selected_cells=selected, boundary_conditions=bcs, external_load=load,
            interface_springs=springs, options=EventSolverOptions(),
        )
        for field in (event.intact_displacement, event.post_event_displacement):
            np.testing.assert_allclose(field.x.array.reshape(-1, 3)[interface], [np.array([0., 0., .001])] * len(interface),
                                       atol=1e-12)
        self.assertGreater(event.report["stages"][1]["solver_iterations"], 0)
        self.assertLess(event.report["stages"][1]["maximum_selected_principal_stress_pa"], 1e-5)
        for row in event.report["stages"]:
            np.testing.assert_allclose(row["support_reaction_n"], [0., 0., -1e4], atol=1e-7)
        springs.matrix.destroy()

    def test_distributed_support_force_is_included_in_total_reaction(self):
        space, selected, _, load, _ = self.fixture(2, traction=-1e6)
        domain = space.mesh
        bottom = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 0.))
        top = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 1.))
        indices = np.r_[top, bottom]
        order = np.argsort(indices)
        tags = mesh.meshtags(domain, 2, indices[order], np.r_[np.ones(len(top)), np.full(len(bottom), 2)][order].astype(np.int32))
        ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
        vertices = np.arange(domain.topology.index_map(0).size_local, dtype=np.int32)
        bcs = [fem.dirichletbc(0., fem.locate_dofs_topological(space.sub(i), 0, vertices), space.sub(i))
               for i in (1, 2)]
        selected[:] = True
        stiffness = 2e9
        structure = SimpleNamespace(rock_support={"model": "rollers", "faces": {
            "2": {"outward_normal": [-1., 0., 0.], "normal_stiffness_n_per_m3": stiffness},
        }})
        event = solve_event(
            space, young_pa=35e9, poisson=0., selected_cells=selected,
            boundary_conditions=bcs, external_load=lambda test: -1e6 * test[0] * ds(1),
            options=EventSolverOptions(),
            linear_support=lambda trial, test: support_form(trial, test, ds, structure),
        )
        x = space.tabulate_dof_coordinates()[:, 0]
        exact = -1e6 * (1 / stiffness + x / 35e9)
        np.testing.assert_allclose(event.post_event_displacement.x.array.reshape(-1, 3)[:, 0], exact, atol=1e-12)
        for row in event.report["stages"]:
            np.testing.assert_allclose(row["linear_support_reaction_n"], [1e6, 0., 0.], atol=.01)
            np.testing.assert_allclose(row["support_reaction_n"], [1e6, 0., 0.], atol=.01)
            self.assertAlmostEqual(row["linear_support_energy_j"], .5 * 1e12 / stiffness, places=7)
            self.assertLess(row["scaled_force_imbalance"], 1e-8)

    def test_gravity_reservoir_and_tailwater_forces_are_preserved_after_event(self):
        space, selected, bcs, _, _ = self.fixture(2)
        domain = space.mesh
        right = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 1.))
        left = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 0.))
        indices = np.r_[right, left]
        order = np.argsort(indices)
        tags = mesh.meshtags(domain, 2, indices[order],
                            np.r_[np.ones(len(right)), np.full(len(left), 2)][order].astype(np.int32))
        ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
        z = ufl.SpatialCoordinate(domain)[2]
        normal = ufl.FacetNormal(domain)
        # Synthetic reservoir/tailwater heads on a unit cube, not dam defaults.
        water_weight = 1000. * 9.81
        reservoir = water_weight * (2. - z)
        tailwater = water_weight * ufl.max_value(.5 - z, 0.)
        def load(test):
            return (-2400. * 9.81 * test[2] * ufl.dx
                    - reservoir * ufl.dot(normal, test) * ds(1)
                    - tailwater * ufl.dot(normal, test) * ds(2))
        selected[:] = True
        event = solve_event(
            space, young_pa=35e9, poisson=0., selected_cells=selected,
            boundary_conditions=bcs, external_load=load, options=EventSolverOptions(),
        )
        applied = np.array([water_weight * (-1.5 + .125), 0., -2400. * 9.81])
        for row in event.report["stages"]:
            np.testing.assert_allclose(row["applied_force_n"], applied, atol=1e-8)
            np.testing.assert_allclose(row["support_reaction_n"], -applied, atol=1e-7)
            self.assertLess(row["scaled_free_residual"], 1e-8)
        # A nonuniform pressure produces shear; compressive resultant alone
        # does not guarantee every intact principal stress is compressive.
        self.assertGreater(event.report["stages"][0]["maximum_selected_principal_stress_pa"], 0.)
        self.assertLessEqual(event.report["stages"][1]["maximum_selected_principal_stress_pa"], .01)
        self.assertGreater(event.report["stages"][1]["solver_iterations"], 0)

    def test_source_adapter_uses_source_rows_and_canonical_gravity(self):
        structure = column()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            event = solve_source_event(
                structure, [1], maximum_volume_change=1e-12,
                options=EventSolverOptions(), output=output,
            )
            with np.load(output / "native_inputs.npz", allow_pickle=False) as inputs:
                np.testing.assert_array_equal(inputs["selected_local_cells"], inputs["source_cell_rows"] == 1)
                np.testing.assert_allclose(inputs["young_pa"],
                                           np.where(inputs["source_cell_rows"] == 1, 1e7, 2e7))
            for row in event.report["stages"]:
                np.testing.assert_allclose(row["applied_force_n"], [0., 0., -30000.], atol=1e-8)
                np.testing.assert_allclose(row["support_reaction_n"], [0., 0., 30000.], atol=1e-7)
            self.assertTrue((output / "interface_springs.npz").is_file())
            self.assertFalse(event.report["solver_ready"])

    def wet_source_fixture(self):
        structure = column()
        structure.wall_connection = "single_monolithic_body"
        # Existing tags 11/12 are retained on the split spring seat.
        boundaries = []
        for tag, nodes in structure.boundaries:
            xyz = structure.nodes[list(nodes)]
            if tag == 7:
                if np.all(xyz[:, 0] == 1.):
                    tag = 2
                elif np.all(xyz[:, 0] == 0.):
                    tag = 3
                elif np.all(xyz[:, 2] == 2.):
                    tag = 4
            boundaries.append((tag, nodes))
        structure.boundaries = boundaries
        structure.loads.upstream_surface_z_m = 3.
        structure.loads.tailwater_head_m = 1.
        return structure

    def test_source_event_intact_state_matches_existing_baseline_with_water_loads(self):
        structure = self.wet_source_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            source_displacement, _, report = solve_elastic_baseline(
                import_structure(structure, 1e-12), structure,
                {"relative_tolerance": 1e-10, "maximum_iterations": 500}, baseline_dir,
            )
            event = solve_source_event(structure, [1], maximum_volume_change=1e-12,
                                       options=EventSolverOptions(), output=root / "event")
        # Waterlines lie on mesh vertices: upstream integral 4, tailwater
        # integral 2 and top 3 m over unit areas; body gravity totals 30000 N.
        expected = [-1000. * 10. * (4. - 2.), 0., -60000.]
        for row in event.report["stages"]:
            np.testing.assert_allclose(row["applied_force_n"], expected, atol=1e-7)
            np.testing.assert_allclose(row["support_reaction_n"], -np.array(expected), atol=1e-6)
        _, actual = source_vector_values(event.intact_displacement)
        np.testing.assert_allclose(actual, source_displacement, rtol=1e-8, atol=1e-11)
        self.assertLess(report["relative_force_balance_error"], 1e-8)

    def test_source_intact_only_persists_native_cell_tensors_for_automatic_screening(self):
        from automatic_screening import screening_from_archives

        structure = self.wet_source_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "must not receive affected source rows"):
                solve_source_event(structure, [1], maximum_volume_change=1e-12, options=EventSolverOptions(),
                                   output=root / "rejected", intact_only=True)
            reference = solve_source_event(structure, [1], maximum_volume_change=1e-12,
                                           options=EventSolverOptions(), output=root / "reference")
            output = root / "intact"
            event = solve_source_event(structure, [], maximum_volume_change=1e-12,
                                       options=EventSolverOptions(), output=output, intact_only=True)
            self.assertEqual(event.report["status"], "passed_intact_loaded_only")
            self.assertEqual([row["stage"] for row in event.report["stages"]], ["intact_loaded"])
            self.assertFalse((output / "post_event_state.npz").exists())
            np.testing.assert_allclose(event.intact_displacement.x.array,
                                       reference.intact_displacement.x.array, rtol=1e-10, atol=1e-14)
            with np.load(output / "intact_cell_stress.npz", allow_pickle=False) as cells, \
                    np.load(output / "intact_state.npz", allow_pickle=False) as state, \
                    np.load(output / "native_inputs.npz", allow_pickle=False) as inputs:
                self.assertFalse(inputs["selected_local_cells"].any())
                np.testing.assert_array_equal(cells["cell_region_tags"], inputs["cell_region_tags"])
                np.testing.assert_array_equal(cells["source_cell_rows"], inputs["source_cell_rows"])
                np.testing.assert_allclose(cells["cell_stress_pa"].reshape(-1),
                                           state["stress_pa"].reshape(-1, 3, 3)[event.material.tensor_dofs].reshape(-1))
                wall = cells["cell_region_tags"] == 1
                self.assertTrue(wall.any())
                peak = np.linalg.eigvalsh(cells["cell_stress_pa"][wall])[:, -1].max()
            for strength, status in ((peak * 2, "no_overstressed_cells"), (peak / 2, "overstressed_cells_found")):
                result = screening_from_archives(output / "intact_cell_stress.npz", output / "native_inputs.npz",
                                                 strength, relative_tolerance=1e-6)
                self.assertEqual(result["status"], status)
                self.assertAlmostEqual(result["peak_principal_stress_pa"], peak)
                self.assertEqual(result["wall_cells"], int(wall.sum()))

    def test_source_adapter_integrates_waterline_crossing_exactly(self):
        for boundary, head, horizontal in (("tailwater_head_m", .5, -28750.),
                                           ("upstream_surface_z_m", 1.5, 8750.)):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                structure = self.wet_source_fixture()
                setattr(structure.loads, boundary, head)
                output = Path(directory) / "event"
                event = solve_source_event(structure, [1], maximum_volume_change=1e-12,
                                           options=EventSolverOptions(), output=output)
                expected = [horizontal, 0., -30000. - head * 10000. if boundary == "upstream_surface_z_m" else -60000.]
                for row in event.report["stages"]:
                    np.testing.assert_allclose(row["applied_force_n"], expected, atol=1e-8)
                    np.testing.assert_allclose(row["support_reaction_n"], -np.array(expected), atol=1e-6)
                with np.load(output / "native_inputs.npz", allow_pickle=False) as inputs:
                    water = inputs["additional_nodal_force_n"].reshape(-1, 3).sum(axis=0)
                    np.testing.assert_allclose(water, [horizontal, 0., expected[2] + 30000.], atol=1e-8)

    def test_source_adapter_failure_retains_source_mapping_and_intact_state(self):
        structure = column()
        # Synthetic upward body load removes every wall tensile load path.
        structure.loads.gravity_z_m_s2 = 10.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            with self.assertRaisesRegex(RuntimeError, "no equilibrium or opening verdict accepted"):
                solve_source_event(structure, [1], maximum_volume_change=1e-12,
                                   options=EventSolverOptions(maximum_iterations=5), output=output)
            report = json.loads((output / "event_report.json").read_text())
            self.assertEqual(report["status"], "failed_reduced_tensile_loss_event")
            self.assertEqual(report["stages"][0]["stage"], "intact_loaded")
            self.assertTrue((output / "intact_state.npz").is_file())
            self.assertFalse((output / "post_event_state.npz").exists())
            with np.load(output / "native_inputs.npz", allow_pickle=False) as inputs:
                np.testing.assert_array_equal(inputs["selected_local_cells"], inputs["source_cell_rows"] == 1)

    def test_bound_event_retains_provenance_and_markers_without_claiming_growth(self):
        source_hash = hashlib.sha256(b"source").hexdigest()
        baseline = {"source_sha256": {"three_body.msh": source_hash}, "verification": {"passed": True}}
        binding = {
            "affected_element_ids": [1], "crack_start_edge_ids": [42], "opening_limit_m": .002,
            "source_mesh_sha256": source_hash, "baseline_fields_sha256": "fields",
            "configuration_sha256": "config",
            "catalog_sha256": "catalog", "solver_ready": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "three_body.msh").write_bytes(b"source")
            (root / "manifest.json").write_text(json.dumps({"crack_seed_source": {"sha256": "fields"}}))
            structure = column()
            structure.submitted_configuration_sha256 = "config"
            with patch("run.submitted_structure", return_value=structure), patch(
                "three_body_model.require_verified_baseline", return_value=baseline,
            ), patch("damage_selection.bind_source_selection", return_value=binding):
                event = solve_bound_event(root / "config.json", root, root, root / "fields.npz", {},
                                          maximum_volume_change=1e-12, options=EventSolverOptions(),
                                          output=root / "event")
            saved = json.loads((root / "event" / "event_report.json").read_text())
            provenance = saved["selection_provenance"]
            self.assertEqual(provenance["crack_start_edge_ids"], [42])
            self.assertEqual(provenance["opening_limit_m"], .002)
            self.assertEqual(provenance["verified_baseline"], baseline)
            self.assertIn("not crack growth", provenance["component_scope"])
            self.assertFalse(event.report["solver_ready"])

    def test_bound_event_rejects_foreign_baseline_fields_before_native_solve(self):
        baseline = {"source_sha256": {"three_body.msh": "mesh"}}
        binding = {"affected_element_ids": [1], "source_mesh_sha256": "mesh", "baseline_fields_sha256": "other"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"crack_seed_source": {"sha256": "fields"}}))
            with patch("run.submitted_structure", return_value=column()), patch(
                "three_body_model.require_verified_baseline", return_value=baseline,
            ), patch("damage_selection.bind_source_selection", return_value=binding), patch(
                "tensile_loss_event.solve_source_event",
            ) as solve:
                with self.assertRaisesRegex(ValueError, "verified baseline evidence"):
                    solve_bound_event(root / "config.json", root, root, root / "fields.npz", {},
                                      maximum_volume_change=1e-12, options=EventSolverOptions(),
                                      output=root / "event")
                solve.assert_not_called()
                self.assertFalse((root / "event").exists())

    def test_source_adapter_requires_explicit_persistence_and_import_tolerance(self):
        for output, tolerance in ((None, 1e-12), (Path("unused"), float("nan")),
                                  (Path("unused"), True), (Path("unused"), -1.)):
            with self.subTest(output=output, tolerance=tolerance), self.assertRaises(ValueError):
                solve_source_event(column(), [1], maximum_volume_change=tolerance,
                                   options=EventSolverOptions(), output=output)

    def test_explicit_nodal_loads_obey_constraints_and_reactions_without_mortar(self):
        space, selected, bcs, load, _ = self.fixture(2, traction=0.)
        domain = space.mesh
        right = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 1.))
        tags = mesh.meshtags(domain, 2, right, np.ones(len(right), dtype=np.int32))
        from dolfinx.fem.petsc import assemble_vector
        ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
        vector = assemble_vector(fem.form(-1e6 * ufl.TestFunction(space)[0] * ds(1)))
        try:
            forces = vector.array.copy()
        finally:
            vector.destroy()
        # A load directly at a constrained node must be included in the support
        # reaction but must not contaminate the free nonlinear residual.
        fixed = bcs[0].dof_indices()[0][0]
        forces[fixed] += 1234.
        snapshot = forces.copy()
        selected[:] = True
        event = solve_event(
            space, young_pa=35e9, poisson=0., selected_cells=selected,
            boundary_conditions=bcs, external_load=load, additional_nodal_forces=forces,
            options=EventSolverOptions(),
        )
        for row in event.report["stages"]:
            np.testing.assert_allclose(row["applied_force_n"], [-1e6 + 1234., 0., 0.], atol=1e-8)
            np.testing.assert_allclose(row["support_reaction_n"], [1e6 - 1234., 0., 0.], atol=1e-7)
            self.assertLess(row["scaled_free_residual"], 1e-8)
        exact = -1e6 * space.tabulate_dof_coordinates()[:, 0] / 35e9
        np.testing.assert_allclose(event.post_event_displacement.x.array.reshape(-1, 3)[:, 0], exact, atol=1e-12)
        np.testing.assert_array_equal(forces, snapshot)
        for invalid in (np.zeros(len(forces) + 1), np.full(len(forces), np.nan),
                        np.ones(len(forces), dtype=bool)):
            with self.assertRaisesRegex(ValueError, "Additional nodal forces"):
                solve_event(space, young_pa=35e9, poisson=0., selected_cells=selected,
                            boundary_conditions=bcs, external_load=load, additional_nodal_forces=invalid,
                            options=EventSolverOptions())

    def test_source_adapter_excludes_bedrock_and_concrete_foundation(self):
        for structure, rows in ((column(), [0]), (foundation_column(), [1]),
                                (column(), [1, 1]), (column(), [99])):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "event"
                with self.assertRaises(ValueError):
                    solve_source_event(structure, rows, maximum_volume_change=1e-12,
                                       options=EventSolverOptions(), output=output)
                self.assertFalse(output.exists())

    def test_source_adapter_retains_foundation_material_and_both_spring_interfaces(self):
        structure = foundation_column()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            event = solve_source_event(structure, [2], maximum_volume_change=1e-12,
                                       options=EventSolverOptions(), output=output)
            with np.load(output / "native_inputs.npz", allow_pickle=False) as inputs:
                np.testing.assert_array_equal(inputs["selected_local_cells"], inputs["source_cell_rows"] == 2)
                np.testing.assert_allclose(inputs["young_pa"],
                                           np.where(inputs["source_cell_rows"] == 0, 2e7, 1e7))
            for name in ("wall_foundation", "foundation_bedrock"):
                self.assertIn(name, event.report["interface_springs"])
            for row in event.report["stages"]:
                np.testing.assert_allclose(row["applied_force_n"], [0., 0., -40000.], atol=1e-7)
                np.testing.assert_allclose(row["support_reaction_n"], [0., 0., 40000.], atol=1e-6)

    def test_bad_options_selection_and_reused_output_are_rejected(self):
        for options in ({"relative_tolerance": True}, {"relative_tolerance": 1.},
                        {"maximum_iterations": 0}, {"absolute_tolerance_n": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EventSolverOptions(**options)
        space, selected, bcs, load, _ = self.fixture(2)
        with self.assertRaisesRegex(ValueError, "At least one"):
            solve_event(space, young_pa=35e9, poisson=0., selected_cells=np.zeros_like(selected),
                        boundary_conditions=bcs, external_load=load, options=EventSolverOptions())
        selected[:] = True
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileExistsError):
            solve_event(space, young_pa=35e9, poisson=0., selected_cells=selected,
                        boundary_conditions=bcs, external_load=load, options=EventSolverOptions(), output=Path(directory))


if __name__ == "__main__":
    unittest.main()
