"""Native 3-D nonpenetration, reopening and source-identity verification."""
import unittest
import json
from datetime import datetime
from pathlib import Path
import tempfile

from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import spsolve
import ufl

from fracture_faces import PairedCrackFaces
from paired_contact import (
    ContactOptions, gap_constraints, mixed_constraints, solve_contact, solve_contact_equilibrium,
)
from tensile_loss_event import EventSolverOptions, solve_contact_event
from fracture_steady_flow import SteadyCrackHydraulics, HydrostaticEquilibriumHydraulics
import test_open_crack_equilibrium as bulk_fixture


class PairedContactTests(unittest.TestCase):
    def fixture(self):
        space, faces, _, _, bcs = bulk_fixture.BulkFlowTests().fixture()
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        operator = assemble_matrix(fem.form(
            1e8 * ufl.inner(ufl.sym(ufl.grad(trial)), ufl.sym(ufl.grad(test))) * ufl.dx))
        operator.assemble()
        try:
            offsets, columns, values = operator.getValuesCSR()
            stiffness = sparse.csr_matrix((values.copy(), columns.copy(), offsets.copy()), shape=operator.getSize())
        finally:
            operator.destroy()
        source = space.mesh.geometry.input_global_indices[space.mesh.geometry.dofmaps[0]]
        mapping = np.empty(len(faces.coordinates), dtype=np.int32)
        mapping[source] = space.dofmap.list
        fixed = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs]))
        return stiffness, faces, mapping, fixed

    def solve(self, stiffness, faces, mapping, fixed, pressure, *, options=None, prescribed=None):
        nodal = np.empty_like(faces.coordinates)
        nodal[mapping] = faces.pressure_forces(np.full(faces.minus.shape, abs(pressure))) * np.sign(pressure)
        return solve_contact(stiffness, nodal.ravel(), fixed,
                             np.zeros(len(fixed)) if prescribed is None else prescribed, faces, mapping,
                             options=options or ContactOptions(1e-12, 1e-7, 1e-9, 100))

    def test_native_pressure_release_reopening_and_reclosure(self):
        stiffness, faces, mapping, fixed = self.fixture()
        # Repeated equilibrium states, not fabricated physical seconds.
        for pressure in (-1e5, 0., 4e4, 1e5, -2e4):
            with self.subTest(pressure=pressure):
                result = self.solve(stiffness, faces, mapping, fixed, pressure)
                values = result.displacement_m.reshape(-1, 3)[mapping]
                expected_gap = 2 * max(pressure, 0.) / 1e8
                np.testing.assert_allclose(faces.measure(values)["signed_vertex_gaps_m"], expected_gap, atol=1e-12)
                self.assertLess(result.report["relative_free_residual"], 1e-9)
                self.assertGreaterEqual(result.multipliers_n.min(), -1e-7)
                self.assertLess(np.linalg.norm(result.contact_nodal_forces_n.reshape(-1, 3).sum(axis=0)), 1e-8)
                if pressure <= 0.:
                    np.testing.assert_allclose(result.displacement_m, 0., atol=1e-12)
                    self.assertAlmostEqual(result.multipliers_n.sum(), abs(pressure), places=6)
                else:
                    np.testing.assert_allclose(result.multipliers_n, 0., atol=1e-7)
                    self.assertAlmostEqual(result.report["elastic_energy_j"], pressure**2 / 1e8, places=7)
                    reaction = result.support_reaction_n.reshape(-1, 3)[mapping]
                    self.assertAlmostEqual(reaction[faces.coordinates[:, 0] == -1., 0].sum(), pressure, places=6)
                    self.assertAlmostEqual(reaction[faces.coordinates[:, 0] == 1., 0].sum(), -pressure, places=6)

    def test_pressure_offsets_compression_without_inventing_gap(self):
        stiffness, faces, mapping, fixed = self.fixture()
        for closure in (2e-4, 1e-3):
            prescribed = np.zeros(len(fixed))
            source_for_dof = np.argsort(mapping)
            for i, dof in enumerate(fixed):
                if dof % 3 == 0:
                    x = faces.coordinates[source_for_dof[dof // 3], 0]
                    prescribed[i] = -np.sign(x) * closure
            for pressure in (0., .5e8 * closure, 2e8 * closure):
                result = self.solve(stiffness, faces, mapping, fixed, pressure, prescribed=prescribed)
                values = result.displacement_m.reshape(-1, 3)[mapping]
                np.testing.assert_allclose(faces.measure(values)["signed_vertex_gaps_m"],
                                           2 * max(pressure / 1e8 - closure, 0.), atol=1e-12)
                self.assertAlmostEqual(result.multipliers_n.sum(), max(1e8 * closure - pressure, 0.), places=6)

    def test_vertex_constraints_preserve_identity_and_virtual_work(self):
        _, faces, mapping, _ = self.fixture()
        operator = gap_constraints(faces, mapping)
        # Eight triangles share vertices; only nine independent nodal gaps.
        self.assertEqual(operator.shape, (9, len(mapping) * 3))
        rng = np.random.default_rng(42)
        displacement = rng.normal(size=operator.shape[1])
        multiplier = rng.random(operator.shape[0])
        self.assertAlmostEqual(float(displacement @ (operator.T @ multiplier)),
                               float((operator @ displacement) @ multiplier), places=12)
        exchange = PairedCrackFaces(faces.coordinates, faces.plus, faces.minus, -faces.normals)
        np.testing.assert_array_equal(gap_constraints(exchange, mapping).toarray(), operator.toarray())
        # Tangential translation has exactly zero normal gap and contact work.
        tangent = np.zeros_like(faces.coordinates)
        tangent[len(tangent) // 2:, 1:] = [2., -3.]
        native = np.empty_like(tangent)
        native[mapping] = tangent
        np.testing.assert_array_equal(operator @ native.ravel(), 0.)

    def test_partial_contact_matches_independent_constrained_energy_minimum(self):
        stiffness, faces, mapping, fixed = self.fixture()
        source_load = (faces.pressure_forces(np.full(faces.minus.shape, 7e4))
                       - faces.pressure_forces(1.4e5 * faces.coordinates[faces.minus, 1]))
        native_load = np.empty_like(source_load)
        native_load[mapping] = source_load
        result = solve_contact(
            stiffness, native_load.ravel(), fixed, np.zeros(len(fixed)), faces, mapping,
            options=ContactOptions(1e-12, 1e-7, 1e-9, 100),
        )
        self.assertGreater(result.report["active_constraints"], 0)
        self.assertLess(result.report["active_constraints"], result.report["constraints"])
        free = np.setdiff1d(np.arange(stiffness.shape[0]), fixed)
        matrix = stiffness[free][:, free].toarray()
        load = native_load.ravel()[free]
        constraints = gap_constraints(faces, mapping)[:, free].toarray()
        # Optimise in millimetres and scale potential energy for SLSQP.
        displacement_scale = 1e-3
        energy_scale = 1e8 * displacement_scale**2
        oracle = minimize(
            lambda x: (.5 * (x * displacement_scale) @ matrix @ (x * displacement_scale)
                       - load @ (x * displacement_scale)) / energy_scale,
            np.zeros(len(free)),
            jac=lambda x: displacement_scale * (matrix @ (x * displacement_scale) - load) / energy_scale,
            constraints=[LinearConstraint(constraints, 0., np.inf)],
            method="SLSQP", options={"ftol": 1e-14, "maxiter": 300},
        )
        self.assertTrue(oracle.success, oracle.message)
        np.testing.assert_allclose(result.displacement_m[free], oracle.x * displacement_scale, atol=1e-9)

    def test_failure_is_explicit_for_iteration_limit_and_incompatible_supports(self):
        stiffness, faces, mapping, fixed = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "iteration limit"):
            self.solve(stiffness, faces, mapping, fixed, 1e5,
                       options=ContactOptions(1e-12, 1e-7, 1e-9, 1))
        fixed = np.arange(stiffness.shape[0], dtype=np.int32)
        native = np.zeros_like(faces.coordinates)
        native[mapping[faces.plus.ravel()], 0] = -1e-3
        with self.assertRaisesRegex(ValueError, "free displacement"):
            self.solve(stiffness, faces, mapping, fixed, 0., prescribed=native.ravel())

    def native_fixture(self):
        space, faces, _, _, _ = bulk_fixture.BulkFlowTests().fixture()
        cells = space.mesh.geometry.input_global_indices[space.mesh.geometry.dofmaps[0]]
        first = np.array([1., 2., 3.]) / np.sqrt(14.)
        second = np.cross(first, [0., 0., 1.])
        second /= np.linalg.norm(second)
        rotation = np.column_stack((first, second, np.cross(first, second)))
        points = faces.coordinates @ rotation.T
        domain = mesh.create_mesh(MPI.COMM_WORLD, cells, space.mesh.ufl_domain(), points)
        domain.topology.create_connectivity(0, 3)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        faces = PairedCrackFaces(points, faces.minus, faces.plus, faces.normals @ rotation.T)
        ends = mesh.locate_entities(domain, 0, lambda x: np.isclose(np.abs(first @ x), 1.))
        bc = fem.dirichletbc(np.zeros(3), fem.locate_dofs_topological(space, 0, ends), space)
        source = domain.geometry.input_global_indices[domain.geometry.dofmaps[0]]
        mapping = np.empty(len(points), dtype=np.int32)
        mapping[source] = space.dofmap.list
        compression = np.empty_like(points)
        compression[mapping] = -faces.pressure_forces(np.full(faces.minus.shape, 1e5))
        return space, faces, [bc], mapping, compression, rotation

    def test_native_driver_oblique_contact_pressure_and_persistent_exports(self):
        space, faces, bcs, mapping, compression, _ = self.native_fixture()
        for pressure in (0., 5e4, 2e5):
            with self.subTest(pressure=pressure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "contact"
                field, contact, report = solve_contact_equilibrium(
                    space, faces,
                    elastic_form=lambda u, v: 1e8 * ufl.inner(
                        ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx,
                    external_load=lambda v: ufl.dot(fem.Constant(space.mesh, np.zeros(3)), v) * ufl.dx,
                    boundary_conditions=bcs, additional_nodal_forces=compression.ravel(),
                    face_pressures_pa=np.full(faces.minus.shape, pressure),
                    options=ContactOptions(1e-12, 1e-7, 1e-9, 100), output=root,
                )
                self.assertEqual(report["status"], "passed_reduced_native_contact")
                np.testing.assert_allclose(report["opening"]["signed_vertex_gaps_m"],
                                           2 * max(pressure - 1e5, 0.) / 1e8, atol=1e-12)
                self.assertAlmostEqual(contact.multipliers_n.sum(), max(1e5 - pressure, 0.), places=6)
                with np.load(root / "contact_state.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["displacement_m"], field.x.array.reshape(-1, 3)[mapping])
                    np.testing.assert_array_equal(saved["nodal_contact_forces_n"],
                                                   contact.contact_nodal_forces_n.reshape(-1, 3)[mapping])
                with np.load(root / "native_inputs.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["pressure_pa"], pressure)
                self.assertTrue((root / "displacement.pvd").is_file())
                self.assertFalse(report["solver_ready"])

    def test_native_frictionless_contact_allows_both_tangential_components(self):
        space, faces, bcs, mapping, compression, rotation = self.native_fixture()
        # Apply two tangential tractions while holding the lips in compression.
        tangent = rotation[:, 1] + .7 * rotation[:, 2]
        weights = faces.areas[:, None] / 3 * np.ones(faces.minus.shape) * 1e4
        traction = np.zeros_like(faces.coordinates)
        np.add.at(traction, faces.plus, weights[:, :, None] * tangent)
        np.add.at(traction, faces.minus, -weights[:, :, None] * tangent)
        native = np.empty_like(traction)
        native[mapping] = traction
        with tempfile.TemporaryDirectory() as directory:
            _, contact, report = solve_contact_equilibrium(
                space, faces,
                elastic_form=lambda u, v: 1e8 * ufl.inner(
                    ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx,
                external_load=lambda v: ufl.dot(fem.Constant(space.mesh, np.zeros(3)), v) * ufl.dx,
                boundary_conditions=bcs, additional_nodal_forces=(compression + native).ravel(),
                options=ContactOptions(1e-12, 1e-7, 1e-9, 100), output=Path(directory) / "contact",
            )
            contact_local = contact.contact_nodal_forces_n.reshape(-1, 3) @ rotation
            np.testing.assert_allclose(contact_local[:, 1:], 0., atol=1e-9)
            jumps = np.asarray(report["opening"]["tangential_vertex_jumps_m"]) @ rotation
            self.assertGreater(np.max(np.abs(jumps[:, :, 1])), 1e-5)
            self.assertGreater(np.max(np.abs(jumps[:, :, 2])), 1e-5)

    def test_native_failure_preserves_inputs_but_not_accepted_output(self):
        space, faces, bcs, _, _, _ = self.native_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "failed"
            with self.assertRaisesRegex(RuntimeError, "no equilibrium accepted"):
                solve_contact_equilibrium(
                    space, faces,
                    elastic_form=lambda u, v: 1e8 * ufl.inner(
                        ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx,
                    external_load=lambda v: ufl.dot(fem.Constant(space.mesh, np.zeros(3)), v) * ufl.dx,
                    boundary_conditions=bcs, face_pressures_pa=np.full(faces.minus.shape, 1e5),
                    options=ContactOptions(1e-12, 1e-7, 1e-9, 1), output=root,
                )
            report = json.loads((root / "contact_report.json").read_text())
            self.assertEqual(report["status"], "failed_reduced_native_contact")
            self.assertTrue((root / "native_inputs.npz").is_file())
            self.assertEqual(len(report["iterations"]), 1)
            self.assertFalse((root / "contact_state.npz").exists())
            self.assertFalse((root / "displacement.pvd").exists())

    def test_bonded_intact_state_transfers_tension_and_both_shear_components(self):
        stiffness, faces, mapping, fixed = self.fixture()
        nodal = np.empty_like(faces.coordinates)
        nodal[mapping] = faces.pressure_forces(np.full(faces.minus.shape, 1e5))
        bonded = solve_contact(stiffness, nodal.ravel(), fixed, np.zeros(len(fixed)), faces, mapping,
                               options=ContactOptions(1e-12, 1e-7, 1e-9, 100), bonded=True)
        np.testing.assert_allclose(bonded.displacement_m, 0., atol=1e-12)
        self.assertEqual(bonded.report["status"], "passed_reduced_bonded_equilibrium")
        self.assertEqual(bonded.report["constraints"], 27)

    def test_partial_release_keeps_unbroken_faces_bonded_in_all_components(self):
        stiffness, faces, mapping, fixed = self.fixture()
        nodal = np.empty_like(faces.coordinates)
        nodal[mapping] = faces.pressure_forces(np.full(faces.minus.shape, 1e5))
        mask = np.ones(len(faces.minus), dtype=bool)
        mask[0] = False
        result = solve_contact(
            stiffness, nodal.ravel(), fixed, np.zeros(len(fixed)), faces, mapping,
            options=ContactOptions(1e-12, 1e-7, 1e-9, 100), bonded_face_mask=mask)
        source = result.displacement_m.reshape(-1, 3)[mapping]
        np.testing.assert_allclose(source[faces.plus[mask]] - source[faces.minus[mask]], 0., atol=1e-12)
        self.assertGreater(result.report["bonded_equality_constraints"], 0)
        self.assertGreaterEqual(result.report["minimum_contact_multiplier_n"], -1e-7)
        self.assertLess(result.report["relative_free_residual"], 1e-9)
        for invalid in (np.ones(len(mask), dtype=int), mask[:-1]):
            with self.assertRaises(ValueError):
                solve_contact(stiffness, nodal.ravel(), fixed, np.zeros(len(fixed)), faces, mapping,
                              options=ContactOptions(1e-12, 1e-7, 1e-9, 100), bonded_face_mask=invalid)

    def test_exact_condensation_matches_multiplier_oracle_on_native_partial_contact(self):
        stiffness, faces, mapping, fixed = self.fixture()
        rng = np.random.default_rng(20261028)
        load = rng.normal(size=stiffness.shape[0]) * 100.
        prescribed = np.zeros(len(fixed))
        prescribed[fixed % 3 == 0] = rng.normal(size=np.count_nonzero(fixed % 3 == 0)) * 1e-6
        for release in (False, True):
            mask = np.ones(len(faces.minus), dtype=bool)
            if release:
                mask[:2] = False
            results = []
            for condensed in (False, True):
                result = solve_contact(
                    stiffness, load, fixed, prescribed, faces, mapping,
                    options=ContactOptions(1e-12, 1e-7, 1e-9, 100, condense_bonds=condensed),
                    bonded_face_mask=mask)
                results.append(result)
            original, condensed = results
            np.testing.assert_allclose(condensed.displacement_m, original.displacement_m, rtol=1e-9, atol=1e-14)
            np.testing.assert_allclose(condensed.support_reaction_n, original.support_reaction_n, atol=1e-7)
            np.testing.assert_allclose(condensed.contact_nodal_forces_n, original.contact_nodal_forces_n, atol=1e-7)
            np.testing.assert_allclose(condensed.multipliers_n, original.multipliers_n, atol=1e-7)
            self.assertAlmostEqual(condensed.report["elastic_energy_j"], original.report["elastic_energy_j"], places=10)
            self.assertLess(condensed.report["relative_free_residual"], 1e-9)
            report = condensed.report["bond_condensation"]
            self.assertLess(report["condensed_displacement_dofs"], report["original_displacement_dofs"])
            self.assertEqual(report["eliminated_bond_multipliers"],
                             condensed.report["bonded_equality_constraints"])
            gap, equality = mixed_constraints(faces, mapping, mask)
            np.testing.assert_allclose(gap[equality] @ condensed.displacement_m, 0., atol=0.)
            self.assertAlmostEqual(
                float(condensed.displacement_m @ condensed.contact_nodal_forces_n),
                float((gap @ condensed.displacement_m) @ condensed.multipliers_n), places=11)

    def test_transitive_bond_forest_force_recovery_with_reordered_dofs_and_fixed_branches(self):
        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.)] * 4)
        faces = PairedCrackFaces(
            points, [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[3, 4, 5], [6, 7, 8], [9, 10, 11]], [[1., 0., 0.]] * 3)
        rng = np.random.default_rng(94)
        mapping = rng.permutation(12)
        raw = rng.normal(size=(36, 36))
        stiffness = sparse.csr_matrix(raw.T @ raw + np.eye(36))
        load = rng.normal(size=36)
        for supports in ((0,), (0, 9), (0, 1, 2), (0, 1, 2, 9, 10, 11)):
            fixed = np.sort(np.concatenate([3 * mapping[node] + np.arange(3) for node in supports]))
            prescribed = np.full(len(fixed), .001)
            result = solve_contact(
                stiffness, load, fixed, prescribed, faces, mapping,
                options=ContactOptions(1e-12, 1e-9, 1e-10, 100), bonded=True)
            original_fixed = result.displacement_m[fixed]
            np.testing.assert_array_equal(original_fixed, prescribed)
            native = result.displacement_m.reshape(-1, 3)[mapping]
            for first in range(3):
                np.testing.assert_array_equal(native[first::3], np.tile(native[first], (4, 1)))
            residual = stiffness @ result.displacement_m - load - result.contact_nodal_forces_n
            free = np.setdiff1d(np.arange(36), fixed)
            np.testing.assert_allclose(residual[free], 0., atol=1e-10)
            np.testing.assert_allclose(residual[fixed], result.support_reaction_n[fixed], atol=1e-10)
            np.testing.assert_allclose(result.contact_nodal_forces_n.reshape(-1, 3).sum(axis=0), 0., atol=1e-10)

    def test_conflicting_support_values_on_a_bond_fail_without_changing_imposed_displacement(self):
        stiffness, faces, mapping, fixed = self.fixture()
        extra = 3 * mapping[np.array([faces.minus[0, 0], faces.plus[0, 0]])]
        fixed = np.concatenate((fixed, extra))
        prescribed = np.zeros(len(fixed))
        prescribed[-1] = 1e-4
        with self.assertRaisesRegex(ValueError, "conflict across an exact bonded"):
            solve_contact(
                stiffness, np.zeros(stiffness.shape[0]), fixed, prescribed, faces, mapping,
                options=ContactOptions(1e-12, 1e-7, 1e-9, 100), bonded=True)
        with self.assertRaisesRegex(ValueError, "condense_bonds"):
            ContactOptions(1e-12, 1e-7, 1e-9, 100, condense_bonds=1)

    def test_iterative_contact_matches_direct_with_and_without_bond_condensation(self):
        stiffness, faces, mapping, fixed = self.fixture()
        nodal = np.empty_like(faces.coordinates)
        nodal[mapping] = (faces.pressure_forces(np.full(faces.minus.shape, 7e4))
                          - faces.pressure_forces(1.4e5 * faces.coordinates[faces.minus, 1]))
        for bonded_mask in (None, np.ones(len(faces.minus), dtype=bool),
                            np.arange(len(faces.minus)) >= 2):
            prescribed = np.zeros(len(fixed))
            prescribed[fixed % 3 == 0] = 1e-5
            results = []
            for method in ("direct", "minres"):
                result = solve_contact(
                    stiffness, nodal.ravel(), fixed, prescribed, faces, mapping,
                    options=ContactOptions(
                        1e-11, 1e-6, 1e-8, 100, linear_solver=method,
                        linear_relative_tolerance=1e-13, linear_maximum_iterations=500),
                    bonded_face_mask=bonded_mask)
                results.append(result)
            direct, iterative = results
            np.testing.assert_allclose(iterative.displacement_m, direct.displacement_m, atol=1e-11)
            np.testing.assert_allclose(iterative.support_reaction_n, direct.support_reaction_n, atol=1e-4)
            np.testing.assert_allclose(iterative.contact_nodal_forces_n, direct.contact_nodal_forces_n, atol=1e-4)
            self.assertAlmostEqual(iterative.report["elastic_energy_j"], direct.report["elastic_energy_j"], places=6)
            self.assertTrue(all(row["method"] == "minres" for row in iterative.report["linear_systems"]))
            self.assertGreater(sum(row["iterations"] for row in iterative.report["linear_systems"]), 0)
            self.assertLess(iterative.report["relative_free_residual"], 1e-8)

    def test_mumps_contact_matches_direct_with_and_without_bond_condensation(self):
        from unittest.mock import patch
        stiffness, faces, mapping, fixed = self.fixture()
        nodal = np.empty_like(faces.coordinates)
        nodal[mapping] = (faces.pressure_forces(np.full(faces.minus.shape, 7e4))
                          - faces.pressure_forces(1.4e5 * faces.coordinates[faces.minus, 1]))
        for bonded_mask in (None, np.ones(len(faces.minus), dtype=bool),
                            np.arange(len(faces.minus)) >= 2):
            prescribed = np.zeros(len(fixed))
            prescribed[fixed % 3 == 0] = 1e-5
            results = []
            for method in ("direct", "mumps"):
                with patch("paired_contact.spsolve", wraps=spsolve) as superlu:
                    results.append(solve_contact(
                        stiffness, nodal.ravel(), fixed, prescribed, faces, mapping,
                        options=ContactOptions(1e-11, 1e-6, 1e-8, 100, linear_solver=method),
                        bonded_face_mask=bonded_mask))
                if method == "mumps":
                    superlu.assert_not_called()
            direct, mumps = results
            np.testing.assert_allclose(mumps.displacement_m, direct.displacement_m, atol=1e-11)
            np.testing.assert_allclose(mumps.support_reaction_n, direct.support_reaction_n, atol=1e-4)
            np.testing.assert_allclose(mumps.contact_nodal_forces_n, direct.contact_nodal_forces_n, atol=1e-4)
            np.testing.assert_allclose(mumps.multipliers_n, direct.multipliers_n, atol=1e-4)
            self.assertEqual(mumps.report["active_constraints"], direct.report["active_constraints"])
            self.assertTrue(all(row["method"] == "mumps" for row in mumps.report["linear_systems"]))
            self.assertLess(mumps.report["relative_free_residual"], 1e-8)

    def test_mumps_failure_has_no_fallback(self):
        from unittest.mock import patch
        stiffness, faces, mapping, fixed = self.fixture()
        load = np.random.default_rng(7).normal(size=stiffness.shape[0])
        with patch("paired_contact._solve_mumps_symmetric",
                   side_effect=RuntimeError("Contact MUMPS factorization failed")), \
                patch("paired_contact.spsolve") as direct, patch("paired_contact.minres") as iterative:
            with self.assertRaisesRegex(RuntimeError, "MUMPS factorization failed"):
                solve_contact(stiffness, load, fixed, np.zeros(len(fixed)), faces, mapping,
                              options=ContactOptions(1e-12, 1e-7, 1e-9, 100, linear_solver="mumps"))
            direct.assert_not_called()
            iterative.assert_not_called()

    def test_iterative_failure_has_no_direct_fallback_and_validates_solver_controls(self):
        from unittest.mock import patch
        stiffness, faces, mapping, fixed = self.fixture()
        load = np.random.default_rng(42).normal(size=stiffness.shape[0])
        with patch("paired_contact.spsolve") as direct:
            with self.assertRaisesRegex(RuntimeError, "MINRES did not converge"):
                solve_contact(
                    stiffness, load, fixed, np.zeros(len(fixed)), faces, mapping,
                    options=ContactOptions(
                        1e-12, 1e-7, 1e-9, 100, linear_solver="minres", linear_maximum_iterations=1))
            direct.assert_not_called()
        with patch("paired_contact.spsolve") as direct:
            with self.assertRaisesRegex(RuntimeError, "positive mechanical diagonal"):
                solve_contact(
                    sparse.csr_matrix(stiffness.shape), load, fixed, np.zeros(len(fixed)), faces, mapping,
                    options=ContactOptions(1e-12, 1e-7, 1e-9, 100, linear_solver="minres"))
            direct.assert_not_called()
        for kwargs in (
            {"linear_solver": "unknown"}, {"linear_solver": True},
            {"linear_relative_tolerance": True}, {"linear_relative_tolerance": float("nan")},
            {"linear_relative_tolerance": 1.}, {"linear_relative_tolerance": 0.},
            {"linear_maximum_iterations": 0}, {"linear_maximum_iterations": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ContactOptions(1e-12, 1e-7, 1e-9, 100, **kwargs)

    def test_semidefinite_mechanism_can_balance_forces_without_proving_unique_displacement(self):
        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.)] * 2)
        faces = PairedCrackFaces(points, [[0, 1, 2]], [[3, 4, 5]], [[1., 0., 0.]])
        block = sparse.csr_matrix([[2., -1.], [-1., 2.]])
        rigid = sparse.csr_matrix([[1., -1.], [-1., 1.]])
        # Positive diagonal does not exclude zero-energy tangential translation.
        stiffness = sparse.lil_matrix((18, 18))
        for component in range(3):
            for vertex in range(3):
                ids = [3 * vertex + component, 3 * (vertex + 3) + component]
                stiffness[np.ix_(ids, ids)] = block if component == 0 else rigid
        stiffness = stiffness.tocsr()
        load = np.zeros(18)
        load[9::3] = 1.
        result = solve_contact(
            stiffness, load, np.empty(0, dtype=int), np.empty(0), faces, np.arange(6),
            options=ContactOptions(1e-12, 1e-9, 1e-10, 100,
                                   linear_solver="minres", linear_relative_tolerance=1e-13))
        null_mode = np.zeros(18)
        null_mode[1::3] = 1.
        np.testing.assert_array_equal(stiffness @ null_mode, 0.)
        np.testing.assert_array_equal(gap_constraints(faces, np.arange(6)) @ null_mode, 0.)
        shifted = result.displacement_m + .1 * null_mode
        np.testing.assert_allclose(stiffness @ shifted - load - result.contact_nodal_forces_n, 0., atol=1e-10)
        self.assertAlmostEqual(float(.5 * shifted @ stiffness @ shifted),
                               result.report["elastic_energy_j"], places=12)
        self.assertFalse(result.report["displacement_uniqueness_verified"])
        self.assertFalse(result.report["stability_verified"])

    def contact_event(self, output, *, compression=False, all_selected=False, iterations=50,
                      flow=False, dry_open=True, opening_limit_m=.002, dry_growth=False,
                      release_cut=True, support_stiffness=0., common_head_equilibrium=False):
        space, faces, parents, boundaries, bcs = bulk_fixture.BulkFlowTests().fixture()
        if common_head_equilibrium:
            from fracture_hydrostatic import HydrostaticBoundary
            boundaries = [HydrostaticBoundary(b.name, 4., b.inlet_source_edges) for b in boundaries]
        count = space.mesh.topology.index_map(3).size_local
        centres = mesh.compute_midpoints(space.mesh, 3, np.arange(count, dtype=np.int32))
        selected = ((np.abs(centres[:, 0]) > .2) & (np.abs(centres[:, 0]) < .8)
                    & (centres[:, 1] > .2) & (centres[:, 1] < .8)
                    & (centres[:, 2] > .2) & (centres[:, 2] < .8))
        if all_selected:
            selected[:] = True
        self.assertGreater(selected.sum(), 0)
        if compression:
            vertices = np.arange(space.mesh.topology.index_map(0).size_local, dtype=np.int32)
            bcs = [fem.dirichletbc(0., fem.locate_dofs_topological(space.sub(i), 0, vertices), space.sub(i))
                   for i in (1, 2)]
            for x, motion in ((-1., 1e-4), (1., -1e-4)):
                ids = mesh.locate_entities(space.mesh, 0, lambda points: np.isclose(points[0], x))
                bcs.append(fem.dirichletbc(motion, fem.locate_dofs_topological(space.sub(0), 0, ids), space.sub(0)))
            load = lambda v: ufl.dot(fem.Constant(space.mesh, np.zeros(3)), v) * ufl.dx
        else:
            vertices = np.arange(space.mesh.topology.index_map(0).size_local, dtype=np.int32)
            ends = mesh.locate_entities(space.mesh, 0, lambda points: np.isclose(np.abs(points[0]), 1.))
            bottom = mesh.locate_entities(space.mesh, 0, lambda points: np.isclose(points[1], 0.))
            bcs = [fem.dirichletbc(0., fem.locate_dofs_topological(space.sub(i), 0, ids), space.sub(i))
                   for i, ids in ((0, ends), (1, bottom), (2, vertices))]
            if (flow or dry_growth) and dry_open:
                bcs = bcs[1:]
                for x, motion in ((-1., -5e-4), (1., 5e-4)):
                    ids = mesh.locate_entities(space.mesh, 0, lambda points: np.isclose(points[0], x))
                    bcs.append(fem.dirichletbc(motion, fem.locate_dofs_topological(space.sub(0), 0, ids), space.sub(0)))
            top = mesh.locate_entities_boundary(space.mesh, 2, lambda points: np.isclose(points[1], 1.))
            tags = mesh.meshtags(space.mesh, 2, top, np.ones(len(top), dtype=np.int32))
            ds = ufl.Measure("ds", domain=space.mesh, subdomain_data=tags)
            load = lambda v: fem.Constant(space.mesh, 1e5) * v[1] * ds(1)
        return solve_contact_event(
            space, faces, young_pa=1e8, poisson=0., selected_cells=selected,
            boundary_conditions=bcs,
            external_load=load,
            options=EventSolverOptions(maximum_iterations=iterations),
            contact_options=ContactOptions(1e-12, 1e-7, 1e-9, 100),
            face_pressures_pa=None if flow or dry_growth else np.full(faces.minus.shape, 0. if compression else 1e5),
            hydraulics=(HydrostaticEquilibriumHydraulics if common_head_equilibrium else SteadyCrackHydraulics)(
                parents, tuple(boundaries), 1000., 9.81, np.ones(len(faces.minus), dtype=bool), 1e-12, 1e-10
            ) if flow else None, output=output, opening_limit_m=opening_limit_m,
            release_cut=release_cut,
            linear_support=(lambda u, v: support_stiffness * u[0] * v[0] * ufl.dx)
            if support_stiffness else None,
        ), faces

    def test_selected_tensile_loss_and_cut_release_re_equilibrate_together(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            event, faces = self.contact_event(output)
            self.assertEqual(event.report["status"], "passed_reduced_tensile_contact_event")
            np.testing.assert_allclose(event.intact_displacement.x.array.reshape(-1, 3)[:, 0], 0., atol=1e-12)
            xyz = event.intact_displacement.function_space.tabulate_dof_coordinates()
            np.testing.assert_allclose(event.intact_displacement.x.array.reshape(-1, 3)[:, 1],
                                       .001 * xyz[:, 1], atol=1e-12)
            selected_stress = event.material.stress.x.array.reshape(-1, 3, 3)[
                event.material.tensor_dofs][event.material.selected]
            self.assertLess(np.linalg.eigvalsh(selected_stress)[:, -1].max(), .01)
            unselected_stress = event.material.stress.x.array.reshape(-1, 3, 3)[
                event.material.tensor_dofs][~event.material.selected]
            self.assertGreater(np.linalg.eigvalsh(unselected_stress)[:, -1].max(), 1e4)
            self.assertGreater(event.report["opening"]["maximum_opening_m"], .0019)
            self.assertGreater(event.post_event_displacement.x.array.reshape(-1, 3)[:, 1].max(),
                               event.intact_displacement.x.array.reshape(-1, 3)[:, 1].max())
            self.assertLess(event.report["iterations"][-1]["scaled_free_residual"], 1e-8)
            self.assertTrue((output / "intact_state.npz").is_file())
            self.assertTrue((output / "post_event_state.npz").is_file())
            self.assertTrue((output / "displacement.pvd").is_file())
            with np.load(output / "native_inputs.npz") as saved:
                np.testing.assert_array_equal(saved["pressure_pa"], 1e5)
                np.testing.assert_array_equal(saved["selected_local_cells"], event.material.selected)
            self.assertFalse(event.report["solver_ready"])
            self.assertFalse(event.report["fracture_growth_solved"])
            self.assertTrue(event.report["cut_is_prescribed"])

    def test_each_equilibrium_iteration_prints_a_progress_line(self):
        import contextlib
        import io
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(stream):
            event, _ = self.contact_event(Path(directory) / "event")
        rows = [json.loads(line[len("Iteration: "):]) for line in stream.getvalue().splitlines()
                if line.startswith("Iteration: ")]
        self.assertEqual([row["iteration"] for row in rows],
                         [row["iteration"] for row in event.report["iterations"]])
        self.assertEqual([row["scaled_residual"] for row in rows],
                         [row["scaled_free_residual"] for row in event.report["iterations"]])
        self.assertTrue(all(row["target"] == 1e-8 for row in rows))
        self.assertTrue(all(datetime.fromisoformat(row["time"]).tzinfo is not None for row in rows))

    def test_selected_compression_and_contact_survive_cut_release(self):
        with tempfile.TemporaryDirectory() as directory:
            event, _ = self.contact_event(Path(directory) / "event", compression=True, all_selected=True)
            np.testing.assert_allclose(event.post_event_displacement.x.array,
                                       event.intact_displacement.x.array, atol=1e-12)
            self.assertAlmostEqual(event.report["opening"]["maximum_opening_m"], 0., places=12)
            last = event.report["iterations"][-1]
            self.assertGreater(last["contact"]["active_constraints"], 0)
            self.assertLess(last["scaled_free_residual"], 1e-8)

    def test_failed_material_contact_event_preserves_intact_without_post_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            with self.assertRaisesRegex(RuntimeError, "no post-event equilibrium accepted"):
                self.contact_event(output, iterations=1)
            self.assertTrue((output / "intact_state.npz").is_file())
            self.assertFalse((output / "post_event_state.npz").exists())
            self.assertFalse((output / "displacement.pvd").exists())
            self.assertEqual(json.loads((output / "event_report.json").read_text())["status"],
                             "failed_reduced_tensile_contact_event")

    def test_selected_tensile_loss_contact_and_steady_water_share_same_equilibrium(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            event, faces = self.contact_event(output, flow=True)
            self.assertEqual(event.report["status"], "passed_reduced_tensile_contact_event")
            self.assertTrue((output / "dry_post_event_state.npz").is_file())
            self.assertTrue((output / "hydraulic_state.npz").is_file())
            xyz = event.intact_displacement.function_space.tabulate_dof_coordinates()
            np.testing.assert_allclose(event.intact_displacement.x.array.reshape(-1, 3)[:, 0],
                                       5e-4 * xyz[:, 0], atol=1e-12)
            self.assertLess(event.report["iterations"][-1]["scaled_free_residual"], 1e-8)
            self.assertLess(event.report["iterations"][-1]["scaled_force_imbalance"], 1e-8)
            _, _, parents, boundaries, _ = bulk_fixture.BulkFlowTests().fixture()
            hydraulics = SteadyCrackHydraulics(
                parents, tuple(boundaries), 1000., 9.81, np.ones(len(faces.minus), dtype=bool), 1e-12, 1e-10)
            from fracture_faces import source_vector_values
            expected = hydraulics.load(faces, source_vector_values(event.post_event_displacement)[1])
            with np.load(output / "hydraulic_state.npz") as saved:
                np.testing.assert_array_equal(saved["pressure_pa"], expected.vertex_pressures_pa)
                np.testing.assert_array_equal(saved["head_m"], expected.vertex_heads_m)
                np.testing.assert_array_equal(saved["nodal_water_forces_n"], expected.nodal_forces_n)
            self.assertEqual(event.report["iterations"][0]["stage"], "dry_post_event")
            self.assertEqual(event.report["hydraulic_mode"], "fully saturated open-cut steady flow")
            self.assertIsNone(event.report["conservative_hydrostatic_potential_j"])
            self.assertIsNone(event.report["hydrostatic_reservoir_work_j"])
            selected = event.material.stress.x.array.reshape(-1, 3, 3)[
                event.material.tensor_dofs][event.material.selected]
            self.assertLess(np.linalg.eigvalsh(selected)[:, -1].max(), .01)

    def test_closed_event_cannot_get_an_artificial_hydraulic_aperture(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            with self.assertRaisesRegex(RuntimeError, "no post-event equilibrium accepted"):
                self.contact_event(output, flow=True, compression=True, all_selected=True)
            self.assertTrue((output / "intact_state.npz").is_file())
            self.assertTrue((output / "dry_post_event_state.npz").is_file())
            self.assertFalse((output / "hydraulic_state.npz").exists())
            self.assertFalse((output / "post_event_state.npz").exists())

    def test_explicit_common_head_equilibrium_pressure_opens_a_dry_closed_cut(self):
        from fracture_faces import source_vector_values
        with tempfile.TemporaryDirectory() as directory:
            dry, _ = self.contact_event(
                Path(directory) / "dry", compression=True, all_selected=True)
            self.assertAlmostEqual(dry.report["opening"]["maximum_opening_m"], 0., places=12)
            output = Path(directory) / "wet"
            wet, faces = self.contact_event(
                output, flow=True, compression=True, all_selected=True, common_head_equilibrium=True)
            self.assertGreater(wet.report["opening"]["maximum_opening_m"], 0.)
            self.assertFalse(wet.report["hydraulic_requires_dry_opening"])
            self.assertFalse((output / "dry_post_event_state.npz").exists())
            self.assertTrue((output / "post_event_state.npz").is_file())
            self.assertFalse(wet.report["fluid_history_solved"])
            source = source_vector_values(wet.post_event_displacement)[1]
            head = 9810 * (4 - faces.coordinates[faces.minus, 2])
            with np.load(output / "hydraulic_state.npz") as saved:
                np.testing.assert_allclose(saved["pressure_pa"], head, atol=1e-8)
                np.testing.assert_allclose(saved["nodal_water_forces_n"], faces.pressure_forces(head), atol=1e-8)
            self.assertLess(wet.report["iterations"][-1]["scaled_free_residual"], 1e-8)
            self.assertLess(wet.report["iterations"][-1]["scaled_force_imbalance"], 1e-8)
            work = float(np.sum(faces.pressure_forces(head) * source))
            self.assertAlmostEqual(wet.report["hydrostatic_reservoir_work_j"], work, places=8)
            self.assertIsNone(wet.report["dry_mechanical_potential_j"])

    def test_native_material_contact_water_equilibrium_with_exact_tied_crack_front(self):
        from fracture_topology import split_interior_wall_faces
        from fracture_faces import bulk_face_mapping, source_vector_values
        from fracture_hydrostatic import HydrostaticBoundary
        points = np.array([(0., 0., -1.), (0., 0., 1.), (1., 0., 0.),
                           (-.5, .866, 0.), (-.5, -.866, 0.)])
        cells = np.array([[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]])
        split = split_interior_wall_faces(points, cells, [1] * 3, [[0, 1, 2]])
        reference = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        domain = mesh.create_mesh(MPI.COMM_WORLD, split.cells, reference.ufl_domain(), split.coordinates_m)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        faces = split.faces
        _, _, mapping = bulk_face_mapping(space, faces)
        tips = np.array([faces.minus[0, 2], faces.plus[0, 2]])
        fixed_sources = np.setdiff1d(np.arange(len(split.coordinates_m)), tips)
        bcs = [fem.dirichletbc(0., np.asarray(3 * mapping + i, dtype=np.int32), space.sub(i))
               for i in (0, 2)]
        bcs.append(fem.dirichletbc(
            0., np.asarray(3 * mapping[fixed_sources] + 1, dtype=np.int32), space.sub(1)))
        selected = np.all(
            np.isin(domain.geometry.input_global_indices[domain.geometry.dofmaps[0]],
                    np.flatnonzero(np.isin(split.parent_node_ids, [0, 1, 3, 4]))), axis=1)
        nodal = np.zeros_like(split.coordinates_m)
        nodal[mapping[tips[0]]] = -1000. * faces.normals[0]
        nodal[mapping[tips[1]]] = 1000. * faces.normals[0]
        hydraulics = SteadyCrackHydraulics(
            split.parent_node_ids, (HydrostaticBoundary("reservoir", 4., [[0, 2]]),),
            1000., 9.81, np.ones(1, dtype=bool), 1e-12, 1e-10, allow_tied_fronts=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            event = solve_contact_event(
                space, faces, young_pa=1e8, poisson=.2, selected_cells=selected,
                boundary_conditions=bcs,
                external_load=lambda v: ufl.dot(fem.Constant(domain, np.zeros(3)), v) * ufl.dx,
                additional_nodal_forces=nodal.ravel(), options=EventSolverOptions(),
                contact_options=ContactOptions(1e-12, 1e-7, 1e-9, 100),
                hydraulics=hydraulics, output=output)
            source_state = source_vector_values(event.post_event_displacement)[1]
            gaps = np.asarray(faces.measure(source_state)["signed_vertex_gaps_m"])
            np.testing.assert_array_equal(gaps[0, :2], 0.)
            self.assertGreater(gaps[0, 2], 0.)
            expected = hydraulics.load(faces, source_state)
            np.testing.assert_allclose(expected.vertex_pressures_pa,
                                       9810 * (4 - faces.coordinates[faces.minus, 2]), atol=1e-8)
            self.assertEqual(expected.report["aperture_regularisation_m"], 0.)
            self.assertTrue(event.report["hydraulic_parameters"]["allow_tied_fronts"])
            self.assertEqual(event.report["hydraulic_mode"], "fully saturated steady flow with exact tied front")
            self.assertLess(event.report["iterations"][-1]["scaled_free_residual"], 1e-8)
            self.assertLess(event.report["iterations"][-1]["scaled_force_imbalance"], 1e-8)
            with np.load(output / "hydraulic_state.npz") as saved:
                np.testing.assert_array_equal(saved["pressure_pa"], expected.vertex_pressures_pa)
            self.assertTrue((output / "post_event_state.npz").is_file())

    def test_hydraulic_boundary_data_are_snapshots_not_mutable_iteration_inputs(self):
        _, faces, parents, boundaries, _ = bulk_fixture.BulkFlowTests().fixture()
        saturated = np.ones(len(faces.minus), dtype=bool)
        data = SteadyCrackHydraulics(parents, tuple(boundaries), 1000., 9.81, saturated, 1e-12, 1e-10)
        parents[:] = 0
        saturated[:] = False
        self.assertTrue(data.saturated_face_mask.all())
        self.assertGreater(data.parent_node_ids.max(), 0)
        self.assertFalse(data.parent_node_ids.flags.writeable)
        with self.assertRaises(ValueError):
            SteadyCrackHydraulics(parents, tuple(boundaries), 1000., 9.81, saturated, 1e-12, 1e-10)

    def test_partial_inventory_water_and_unbroken_bonds_share_native_equilibrium(self):
        from fracture_faces import source_vector_values, bulk_face_mapping
        space, original_faces, parents, boundaries, _ = bulk_fixture.BulkFlowTests().fixture()
        points, _ = source_vector_values(fem.Function(space))
        cells = space.mesh.geometry.input_global_indices[space.mesh.geometry.dofmaps[0]]
        count = len(points)
        coordinates = np.vstack((points, points + [0., 0., 3.]))
        domain = mesh.create_mesh(
            MPI.COMM_WORLD, np.vstack((cells, cells + count)), space.mesh.ufl_domain(), coordinates)
        domain.topology.create_connectivity(0, 3)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        faces = PairedCrackFaces(
            coordinates, np.vstack((original_faces.minus, original_faces.minus + count)),
            np.vstack((original_faces.plus, original_faces.plus + count)),
            np.vstack((original_faces.normals, original_faces.normals)))
        _, _, mapping = bulk_face_mapping(space, faces)
        vertices = np.arange(domain.topology.index_map(0).size_local, dtype=np.int32)
        ends = mesh.locate_entities(domain, 0, lambda x: np.isclose(np.abs(x[0]), 1.))
        bcs = [fem.dirichletbc(0., fem.locate_dofs_topological(space.sub(i), 0, ids), space.sub(i))
               for i, ids in ((0, ends), (1, vertices), (2, vertices))]
        local_cells = domain.geometry.input_global_indices[domain.geometry.dofmaps[0]]
        selected = np.all(local_cells >= count, axis=1)
        released = np.zeros(len(faces.minus), dtype=bool)
        released[:len(original_faces.minus)] = True
        hydraulic_ids = np.flatnonzero(released)[::-1]
        hydraulics = SteadyCrackHydraulics(
            np.r_[parents, parents + count], tuple(boundaries), 1000., 9.81,
            np.ones(len(hydraulic_ids), dtype=bool), 1e-12, 1e-10, inventory_face_ids=hydraulic_ids)
        x = ufl.SpatialCoordinate(domain)
        args = dict(
            young_pa=1e8, poisson=0., selected_cells=selected, boundary_conditions=bcs,
            external_load=lambda v: fem.Constant(domain, 1e5) * ufl.conditional(x[0] < 0, -1., 1.) * v[0] * ufl.dx,
            options=EventSolverOptions(), contact_options=ContactOptions(1e-12, 1e-7, 1e-9, 100),
            hydraulics=hydraulics, released_face_mask=released)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            event = solve_contact_event(space, faces, output=output, **args)
            source = source_vector_values(event.post_event_displacement)[1]
            np.testing.assert_allclose(
                source[faces.plus[~released]] - source[faces.minus[~released]], 0., atol=1e-12)
            self.assertGreater(event.report["opening"]["maximum_opening_m"], .001)
            self.assertEqual(event.report["hydraulic_inventory_face_ids"], hydraulic_ids.tolist())
            self.assertEqual(event.report["hydraulic_mode"], "explicit saturated released-face subset steady flow")
            self.assertFalse(event.report["fluid_history_solved"])
            self.assertGreater(event.report["iterations"][-1]["contact"]["bonded_equality_constraints"], 0)
            self.assertLess(event.report["iterations"][-1]["scaled_free_residual"], 1e-8)
            self.assertLess(event.report["iterations"][-1]["scaled_force_imbalance"], 1e-8)
            self.assertIsNone(event.report["dry_mechanical_potential_j"])
            expected = hydraulics.load(faces, source)
            np.testing.assert_array_equal(expected.nodal_forces_n[count:], 0.)
            for archive in ("hydraulic_inputs.npz", "hydraulic_state.npz"):
                with np.load(output / archive) as saved:
                    np.testing.assert_array_equal(saved["mechanical_inventory_face_ids"], hydraulic_ids)
            with np.load(output / "hydraulic_state.npz") as saved:
                np.testing.assert_array_equal(saved["pressure_pa"], expected.vertex_pressures_pa)
            args["released_face_mask"] = ~released
            with self.assertRaisesRegex(ValueError, "only released faces"):
                solve_contact_event(space, faces, output=Path(directory) / "invalid", **args)
            self.assertFalse((Path(directory) / "invalid").exists())

    def test_connected_wet_subset_uses_mechanical_bonds_for_exact_front_identity(self):
        from fracture_faces import source_vector_values, bonded_node_representatives
        from fracture_hydrostatic import HydrostaticBoundary
        from itertools import combinations
        space, faces, parents, _, bcs = bulk_fixture.BulkFlowTests().fixture()
        released = np.ones(len(faces.minus), dtype=bool)
        released[0] = False
        ids = np.flatnonzero(released)
        representatives = bonded_node_representatives(faces, ~released)
        edges = {}
        for triangle in representatives[faces.minus[ids]]:
            for a, b in combinations(triangle, 2):
                edge = tuple(sorted((int(a), int(b))))
                edges[edge] = edges.get(edge, 0) + 1
        tied = set(representatives[faces.minus[~released]].ravel())
        inlet = next(edge for edge, owners in edges.items()
                     if owners == 1 and not any(node in tied for node in edge))
        hydraulics = SteadyCrackHydraulics(
            parents, (HydrostaticBoundary("reservoir", 4., [parents[list(inlet)]]),),
            1000., 9.81, np.ones(len(ids), dtype=bool), 1e-12, 1e-10,
            allow_tied_fronts=True, inventory_face_ids=ids)
        x = ufl.SpatialCoordinate(space.mesh)
        count = space.mesh.topology.index_map(3).size_local
        centres = mesh.compute_midpoints(space.mesh, 3, np.arange(count, dtype=np.int32))
        selected = centres[:, 0] < -.75
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            event = solve_contact_event(
                space, faces, young_pa=1e8, poisson=0., selected_cells=selected,
                boundary_conditions=bcs,
                external_load=lambda v: fem.Constant(space.mesh, 1e5) * ufl.conditional(
                    x[0] < 0, -1., 1.) * v[0] * ufl.dx,
                options=EventSolverOptions(), contact_options=ContactOptions(1e-12, 1e-7, 1e-9, 100),
                hydraulics=hydraulics, released_face_mask=released, output=output)
            source = source_vector_values(event.post_event_displacement)[1]
            np.testing.assert_allclose(source[faces.plus[~released]], source[faces.minus[~released]], atol=1e-12)
            expected = hydraulics.load(faces, source, bonded_face_mask=~released)
            self.assertGreater(len(expected.report["tied_front_vertex_ids"]), 0)
            self.assertEqual(expected.report["minimum_vertex_aperture_m"], 0.)
            self.assertEqual(expected.report["aperture_regularisation_m"], 0.)
            np.testing.assert_allclose(expected.vertex_pressures_pa,
                                       9810 * (4 - faces.coordinates[faces.minus[ids], 2]), atol=1e-8)
            np.testing.assert_allclose(expected.nodal_forces_n,
                                       faces.pressure_forces(np.concatenate((
                                           np.zeros((1, 3)), expected.vertex_pressures_pa))), atol=1e-9)
            self.assertLess(event.report["iterations"][-1]["scaled_free_residual"], 1e-8)
            with np.load(output / "hydraulic_state.npz") as saved:
                np.testing.assert_array_equal(saved["pressure_pa"], expected.vertex_pressures_pa)
            broken_tie = source.copy()
            broken_tie[faces.plus[0, 0], 1] += .001
            with self.assertRaisesRegex(ValueError, "converged all-component"):
                hydraulics.load(faces, broken_tie, bonded_face_mask=~released)

    def test_integrated_opening_limit_warns_without_capping_or_failing_equilibrium(self):
        with tempfile.TemporaryDirectory() as directory:
            event, _ = self.contact_event(Path(directory) / "event", opening_limit_m=1e-4)
            assessment = event.report["opening_assessment"]
            self.assertEqual(assessment["opening_limit_status"], "limit_exceeded")
            self.assertTrue(assessment["warnings"])
            self.assertGreater(assessment["final_opening_m"], .0019)
            self.assertNotIn("physical_time_s", assessment)
            self.assertEqual(event.report["status"], "passed_reduced_tensile_contact_event")

    def test_loss_of_every_tensile_path_cannot_report_successful_contact_event(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event"
            with self.assertRaisesRegex(RuntimeError, "no post-event equilibrium accepted"):
                self.contact_event(output, all_selected=True)
            self.assertTrue((output / "intact_state.npz").is_file())
            self.assertFalse((output / "post_event_state.npz").exists())
            self.assertFalse((output / "hydraulic_state.npz").exists())
            self.assertFalse((output / "displacement.pvd").exists())

    def test_selected_material_dry_energy_before_and_after_surface_release_is_bound(self):
        from dry_fracture_growth import contact_event_growth_state
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bonded, _ = self.contact_event(root / "bonded", dry_growth=True, release_cut=False)
            released, _ = self.contact_event(root / "released", dry_growth=True)
            before, after = map(contact_event_growth_state, (bonded, released))
            self.assertEqual(before.problem_sha256, after.problem_sha256)
            self.assertGreater(before.mechanical_potential_j - after.mechanical_potential_j, 0.)
            self.assertFalse(bonded.report["cut_released"])
            self.assertTrue(released.report["cut_released"])
            self.assertEqual(bonded.report["opening_assessment"]["fracture_initiated"], False)
            wet, _ = self.contact_event(root / "wet", flow=True)
            with self.assertRaisesRegex(ValueError, "accepted dry"):
                contact_event_growth_state(wet)

    def test_growth_potential_includes_support_energy_and_original_exterior_work(self):
        with tempfile.TemporaryDirectory() as directory:
            event, _ = self.contact_event(Path(directory) / "event", dry_growth=True, support_stiffness=1e6)
            u = event.post_event_displacement
            expected_energy = float(fem.assemble_scalar(fem.form(
                (event.material.energy + .5e6 * u[0]**2) * ufl.dx)))
            with np.load(Path(directory) / "event" / "native_inputs.npz") as saved:
                work = float(saved["assembled_exterior_force_n"] @ u.x.array)
            self.assertAlmostEqual(event.report["stored_mechanical_energy_j"], expected_energy, places=10)
            self.assertAlmostEqual(event.report["exterior_dead_load_work_j"], work, places=10)
            self.assertAlmostEqual(event.report["dry_mechanical_potential_j"], expected_energy - work, places=10)
            self.assertLess(event.report["iterations"][-1]["scaled_force_imbalance"], 1e-8)


if __name__ == "__main__":
    unittest.main()
