"""Reduced verification problems for the coupled AT2 phase-field solver."""
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from crack_field import CrackSpecification, PhaseFieldParameters, PhaseFieldSolverOptions, signed_distance
from phase_field_solver import (
    degradation,
    elastic_parameters,
    solve_staggered,
    tensile_energy,
    undamaged_stress,
    _solve_damage_obstacle,
    _strain_breakdown_details,
    _aitken_factor,
    load_schedule,
    _update_history,
    _damage_change_details,
)
from elasticity_solver import rigid_body_near_nullspace


class ForceControlledPatchTests(unittest.TestCase):
    def solve_pressure(self, fraction):
        young, poisson, gc, ell = 1000.0, 0.2, 1.0, 0.3
        _, _, bulk = elastic_parameters(young, poisson)
        # Uniform dilation e gives alpha=c*e^2/(1+c*e^2), c=9*K*ell/Gc.
        # Its zero-residual-stiffness pressure has a maximum at c*e^2=1/3.
        critical_strain = np.sqrt(gc / (27 * bulk * ell))
        critical_pressure = 3 * bulk * critical_strain * (3 / 4)**2
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        normal = ufl.FacetNormal(domain)

        def boundaries(space, _factor):
            result = []
            for component in range(3):
                facets = mesh.locate_entities_boundary(
                    domain, 2, lambda x, i=component: np.isclose(x[i], 0),
                )
                dofs = fem.locate_dofs_topological(space.sub(component), 2, facets)
                result.append(fem.dirichletbc(PETSc.ScalarType(0), dofs, space.sub(component)))
            return result

        pressure = fraction * critical_pressure
        result = solve_staggered(
            domain, young, poisson, [], PhaseFieldSolverOptions(
                maximum_staggered_iterations=80, staggered_tolerance=1e-7,
                mechanics_relative_tolerance=1e-10,
            ),
            lambda test, factor, _damage: factor * pressure * ufl.inner(normal, test) * ufl.ds,
            boundary_conditions=boundaries,
            parameters=PhaseFieldParameters(
                length_scale_m=ell, critical_energy_release_rate_j_m2=gc,
                internal_pressure_pa=0.0, load_steps=1, residual_stiffness=1e-8,
            ),
            mechanics_formulation="experimental_amor",
        )
        return result, pressure, bulk, gc, ell

    def test_below_peak_matches_homogeneous_force_controlled_equilibrium(self):
        result, pressure, bulk, gc, ell = self.solve_pressure(0.8)
        coordinates = result.displacement.function_space.tabulate_dof_coordinates()
        values = result.displacement.x.array.reshape(-1, 3)
        dilation = float(np.sum(coordinates * values) / np.sum(coordinates**2))
        damage = 9 * bulk * ell * dilation**2 / (gc + 9 * bulk * ell * dilation**2)
        np.testing.assert_allclose(result.damage.x.array, damage, atol=2e-7)
        np.testing.assert_allclose(values, dilation * coordinates, atol=1e-10)
        np.testing.assert_allclose(
            3 * bulk * dilation * ((1 - damage)**2 + 1e-8), pressure, rtol=1e-6,
        )
        self.assertLess(damage, 0.25)

    def test_above_peak_does_not_report_invalid_softening_as_a_solution(self):
        with self.assertRaisesRegex(
            RuntimeError, "Small-strain fracture model is invalid|failed to converge",
        ):
            self.solve_pressure(1.1)


class ConstitutiveTests(unittest.TestCase):
    def test_damage_change_location_retains_signed_unrelaxed_candidate_and_bounds(self):
        domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
        space = fem.functionspace(domain, ("Lagrange", 1))
        damage, lower = fem.Function(space), fem.Function(space)
        previous = np.full_like(damage.x.array, 0.5)
        damage.x.array[:] = 0.6
        damage.x.array[2] = 0.1
        lower.x.array[:] = 0.05
        details = _damage_change_details(damage, previous, lower)
        self.assertAlmostEqual(details["maximum_absolute_change"], 0.4)
        self.assertAlmostEqual(details["signed_change"], -0.4)
        self.assertEqual(details["candidate_damage"], 0.1)
        self.assertEqual(details["previous_iterate_damage"], 0.5)
        self.assertEqual(details["previous_load_damage"], 0.05)
        self.assertEqual(details["coordinate_xyz_m"], space.tabulate_dof_coordinates()[2].tolist())
        json.dumps(details, allow_nan=False)

    def test_accepted_history_does_not_retain_trial_overshoot(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        space = fem.functionspace(domain, ("DG", 0))
        history, candidate = fem.Function(space), fem.Function(space)
        history.x.array[:] = 2.0
        accepted = history.x.array.copy()
        candidate.x.array[:] = 10.0
        _update_history(history, candidate, accepted)
        np.testing.assert_array_equal(history.x.array, 10.0)
        candidate.x.array[:] = 3.0
        _update_history(history, candidate, accepted)
        np.testing.assert_array_equal(history.x.array, 3.0)
        candidate.x.array[:] = 1.0
        _update_history(history, candidate, accepted)
        np.testing.assert_array_equal(history.x.array, 2.0)
        np.testing.assert_array_equal(accepted, 2.0)
        history.x.array[:] = 10.0
        _update_history(history, candidate)
        np.testing.assert_array_equal(history.x.array, 10.0)

    def test_final_interval_subdivision_preserves_configured_loads_and_endpoint(self):
        self.assertEqual(load_schedule(3), [1 / 3, 2 / 3, 1.0])
        expected = [step / 10 for step in range(1, 10)] + [0.925, 0.95, 0.975, 1.0]
        np.testing.assert_allclose(load_schedule(10, 4), expected, atol=1e-15)
        self.assertEqual(load_schedule(1, 2), [0.5, 1.0])
        for steps, substeps in ((0, 1), (2, 0), (2, -1), (2, True), (2, 1.5)):
            with self.subTest(steps=steps, substeps=substeps), self.assertRaises(ValueError):
                load_schedule(steps, substeps)

    def test_aitken_factor_bounds_and_constant_residual(self):
        residual = np.array([0.5, 0.5])
        previous = np.array([1.0, 1.0])
        self.assertEqual(_aitken_factor(residual, None, 1, 2, MPI.COMM_SELF), 1)
        self.assertEqual(_aitken_factor(residual, previous, 1, 2, MPI.COMM_SELF), 2)
        self.assertEqual(_aitken_factor(previous, residual, 1, 2, MPI.COMM_SELF), 1.0)
        self.assertEqual(_aitken_factor(previous, previous, 1, 2, MPI.COMM_SELF), 1)
        self.assertEqual(_aitken_factor(-previous, previous, 1, 2, MPI.COMM_SELF), 0.5)

    def test_nonfinite_strain_details_remain_json_serializable(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        displacement = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        norms = fem.Function(fem.functionspace(domain, ("DG", 0)))
        norms.x.array[:] = float("nan")
        details = _strain_breakdown_details(displacement, damage, norms)
        self.assertIsNone(details["maximum_strain_norm"])
        self.assertIsNone(details["principal_strains"])
        self.assertEqual(len(details["centroid_xyz_m"]), 3)
        json.dumps(details, allow_nan=False)

    def test_hybrid_compression_patch_loses_stiffness_without_creating_tensile_history(self):
        from dolfinx.fem.petsc import LinearProblem

        lam, mu, bulk = elastic_parameters(1000.0, 0.2)
        kappa = 1e-8
        # Synthetic pressure keeps even the residual-stiffness response small.
        pressure = 3 * bulk * 1e-10
        for divisions in (2, 4):
            domain = mesh.create_unit_cube(MPI.COMM_WORLD, divisions, divisions, divisions)
            space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
            scalar_space = fem.functionspace(domain, ("Lagrange", 1))
            damage = fem.Function(scalar_space)
            trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
            normal = ufl.FacetNormal(domain)
            bcs = []
            for component in range(3):
                facets = mesh.locate_entities_boundary(
                    domain, 2, lambda x, i=component: np.isclose(x[i], 0),
                )
                dofs = fem.locate_dofs_topological(space.sub(component), 2, facets)
                bcs.append(fem.dirichletbc(PETSc.ScalarType(0), dofs, space.sub(component)))
            for alpha in (0.0, 0.99, 1.0):
                with self.subTest(divisions=divisions, damage=alpha):
                    damage.x.array[:] = alpha
                    problem = LinearProblem(
                        degradation(damage, kappa)
                        * ufl.inner(undamaged_stress(trial, lam, mu), ufl.sym(ufl.grad(test)))
                        * ufl.dx,
                        ufl.inner(-pressure * normal, test) * ufl.ds,
                        bcs=bcs,
                        petsc_options_prefix=f"compression_patch_{divisions}_{int(alpha * 100)}_",
                        petsc_options={
                            "ksp_type": "preonly", "pc_type": "lu",
                            "ksp_error_if_not_converged": True,
                        },
                    )
                    displacement = problem.solve()
                    factor = (1 - alpha)**2 + kappa
                    strain = pressure / (3 * bulk * factor)
                    np.testing.assert_allclose(
                        displacement.x.array.reshape(-1, 3),
                        -strain * space.tabulate_dof_coordinates(),
                        rtol=1e-9, atol=1e-15,
                    )
                    history_space = fem.functionspace(domain, ("DG", 0))
                    history = fem.Function(history_space)
                    history.interpolate(fem.Expression(
                        tensile_energy(displacement, bulk, mu),
                        history_space.element.interpolation_points,
                    ))
                    np.testing.assert_allclose(history.x.array, 0, atol=1e-20)
                    lower = fem.Function(scalar_space)
                    lower.x.array[:] = alpha
                    _solve_damage_obstacle(
                        damage, lower, history, 1.0, 0.1, PhaseFieldSolverOptions(),
                    )
                    np.testing.assert_allclose(damage.x.array, alpha, atol=1e-12)
                    residual = problem.b.duplicate()
                    try:
                        problem.A.mult(problem.x, residual)
                        residual.axpy(-1, problem.b)
                        self.assertLess(residual.norm() / problem.b.norm(), 1e-10)
                    finally:
                        residual.destroy()

    def test_elasticity_near_nullspace_contains_six_orthonormal_rigid_modes(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        nullspace = rigid_body_near_nullspace(space)
        vectors = nullspace.getVecs()
        self.assertEqual(len(vectors), 6)
        for i, left in enumerate(vectors):
            for j, right in enumerate(vectors):
                self.assertAlmostEqual(left.dot(right), float(i == j), places=12)

    def test_elastic_parameters_match_isotropic_relations(self):
        lam, mu, bulk = elastic_parameters(35e9, 0.2)
        self.assertAlmostEqual(mu, 35e9 / 2.4)
        self.assertAlmostEqual(lam, 35e9 * 0.2 / (1.2 * 0.6))
        self.assertAlmostEqual(bulk, lam + 2 * mu / 3)

    def test_degradation_keeps_only_the_configured_residual_at_full_damage(self):
        domain = mesh.create_unit_interval(MPI.COMM_WORLD, 1)
        space = fem.functionspace(domain, ("Lagrange", 1))
        damage = fem.Function(space)
        damage.x.array[:] = 1.0
        expression = fem.Expression(degradation(damage, 1e-8), space.element.interpolation_points)
        value = fem.Function(space)
        value.interpolate(expression)
        np.testing.assert_allclose(value.x.array, 1e-8)

    def test_tensile_history_is_zero_for_pure_volumetric_compression(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        displacement = fem.Function(space)
        displacement.interpolate(lambda x: -0.01 * x)
        history_space = fem.functionspace(domain, ("DG", 0))
        history = fem.Function(history_space)
        _, mu, bulk = elastic_parameters(1000.0, 0.2)
        history.interpolate(
            fem.Expression(
                tensile_energy(displacement, bulk, mu),
                history_space.element.interpolation_points,
            )
        )
        np.testing.assert_allclose(history.x.array, 0.0, atol=1e-12)
        norms = fem.Function(history_space)
        norms.x.array[:] = np.sqrt(3) * 0.01
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        details = _strain_breakdown_details(displacement, damage, norms)
        np.testing.assert_allclose(details["principal_strains"], [-0.01, -0.01, -0.01])


class StaggeredSolverTests(unittest.TestCase):
    def test_gross_strain_breakdown_is_rejected_before_damage_and_success_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Small-strain fracture model is invalid"):
                self.solve(strain=2.0, output=Path(directory))
            records = [
                json.loads(line)
                for line in (Path(directory) / "mechanics_diagnostics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[-1]["event"], "model_invalid")
            self.assertGreaterEqual(records[-1]["maximum_strain_norm"], 1)
            self.assertEqual(len(records[-1]["centroid_xyz_m"]), 3)
            self.assertAlmostEqual(
                np.linalg.norm(records[-1]["principal_strains"]),
                records[-1]["maximum_strain_norm"],
            )
            self.assertTrue(all(0 <= value <= 1 for value in records[-1]["damage_at_vertices"]))
            self.assertFalse((Path(directory) / "phase_field_report.json").exists())

    def test_positive_tao_reason_cannot_override_failed_projected_gradient(self):
        original_tao = PETSc.TAO

        class PermissiveTAO:
            Type = original_tao.Type

            def create(self, comm):
                self.tao = original_tao().create(comm)
                return self

            def __getattr__(self, name):
                return getattr(self.tao, name)

            def setTolerances(self, **_):
                self.tao.setTolerances(gatol=1e20, grtol=0, gttol=0)

        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        space = fem.functionspace(domain, ("Lagrange", 1))
        damage, lower = fem.Function(space), fem.Function(space)
        history = fem.Function(fem.functionspace(domain, ("DG", 0)))
        history.x.array[:] = 10
        with patch("phase_field_solver.PETSc.TAO", PermissiveTAO):
            with self.assertRaisesRegex(RuntimeError, "Damage projected gradient exceeds tolerance"):
                _solve_damage_obstacle(damage, lower, history, 1, 0.1, self.options())

    def test_damage_convergence_is_not_scaled_by_large_negative_objective(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
        space = fem.functionspace(domain, ("Lagrange", 1))
        history = fem.Function(fem.functionspace(domain, ("DG", 0)))
        history.x.array[:] = 1e10
        damage, lower = fem.Function(space), fem.Function(space)
        report = _solve_damage_obstacle(damage, lower, history, 2700.0, 0.1, self.options())
        expected = 2e10 / (2700.0 / 0.1 + 2e10)
        np.testing.assert_allclose(damage.x.array, expected, rtol=0, atol=1e-9)
        self.assertLess(report["objective"], -1e9)
        self.assertNotEqual(report["convergence_reason"], PETSc.TAO.ConvergedReason.CONVERGED_GRTOL)
        self.assertLessEqual(
            report["gradient_norm"],
            self.options().damage_gradient_tolerance * max(1.0, report["initial_gradient_norm"]),
        )

    def crack(self, *, load_steps=3):
        return CrackSpecification(
            type="phase_field",
            shape="rectangular",
            centre_m=(0.4, 0.4, 0.4),
            size_m=(0.2, 0.5, 0.02),
            normal=(0.0, 0.0, 1.0),
            length_scale_m=0.15,
            critical_energy_release_rate_j_m2=1.0,
            internal_pressure_pa=0.0,
            residual_stiffness=1e-6,
            load_steps=load_steps,
        )

    def options(self):
        return PhaseFieldSolverOptions(
            maximum_staggered_iterations=50,
            staggered_tolerance=1e-4,
            mechanics_relative_tolerance=1e-10,
            mechanics_maximum_iterations=500,
            damage_gradient_tolerance=1e-10,
            damage_maximum_iterations=500,
            field_output_interval=3,
            minimum_elements_per_length_scale=1.0,
            maximum_refinement_levels=0,
        )

    def solve(
        self, strain=0.2, output=None, mechanics_formulation="hybrid",
        damage_acceleration=False, final_load_substeps=1,
    ):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        facets = mesh.exterior_facet_indices(domain.topology)

        def boundary_conditions(space, load_factor):
            exact = fem.Function(space)
            exact.interpolate(
                lambda x: np.vstack(
                    (
                        load_factor * strain * x[0],
                        -0.2 * load_factor * strain * x[1],
                        -0.2 * load_factor * strain * x[2],
                    )
                )
            )
            dofs = fem.locate_dofs_topological(space, domain.topology.dim - 1, facets)
            return [fem.dirichletbc(exact, dofs)]

        def external_work(test, _load_factor, _damage):
            zero = fem.Constant(domain, np.zeros(3, dtype=PETSc.ScalarType))
            return ufl.inner(zero, test) * ufl.dx

        result = solve_staggered(
            domain,
            1000.0,
            0.2,
            [self.crack()],
            self.options(),
            external_work,
            boundary_conditions=boundary_conditions,
            output=output,
            mechanics_formulation=mechanics_formulation,
            damage_acceleration=damage_acceleration,
            final_load_substeps=final_load_substeps,
        )
        return domain, result

    def test_subdivided_final_load_preserves_endpoint_and_writes_final_fields(self):
        _, reference = self.solve(strain=0.002, mechanics_formulation="experimental_amor")
        with tempfile.TemporaryDirectory() as directory:
            _, result = self.solve(
                strain=0.002, output=Path(directory), mechanics_formulation="experimental_amor",
                damage_acceleration=True, final_load_substeps=2,
            )
            self.assertEqual(result.report["configured_load_steps"], 3)
            self.assertEqual(result.report["final_load_substeps"], 2)
            self.assertEqual(len(result.report["steps"]), 4)
            self.assertEqual(result.report["steps"][-1]["load_factor"], 1.0)
            self.assertIn('timestep="1"', (Path(directory) / "damage.pvd").read_text())
            self.assertTrue(all(
                step["iterations"][-1]["staggered_residual"] <= self.options().staggered_tolerance
                for step in result.report["steps"]
            ))
            np.testing.assert_allclose(
                result.damage.x.array, reference.damage.x.array, rtol=1e-3, atol=5e-4,
            )
            np.testing.assert_allclose(
                result.displacement.x.array, reference.displacement.x.array, rtol=1e-3, atol=1e-5,
            )

    def test_accelerated_amor_matches_unaccelerated_coupled_solution(self):
        _, expected = self.solve(strain=0.03, mechanics_formulation="experimental_amor")
        _, actual = self.solve(
            strain=0.03, mechanics_formulation="experimental_amor", damage_acceleration=True,
        )
        np.testing.assert_allclose(actual.damage.x.array, expected.damage.x.array, atol=5e-4, rtol=1e-3)
        np.testing.assert_allclose(actual.displacement.x.array, expected.displacement.x.array, atol=1e-5, rtol=1e-3)
        for step, reference in zip(actual.report["steps"], expected.report["steps"], strict=True):
            self.assertLessEqual(step["iterations"][-1]["staggered_residual"], self.options().staggered_tolerance)
            self.assertEqual(step["iterations"][-1]["damage_relaxation_factor"], 1.0)
            for field in ("elastic_energy_j", "fracture_energy_j"):
                np.testing.assert_allclose(step[field], reference[field], rtol=1e-3, atol=1e-6)
        self.assertTrue(np.all(actual.damage.x.array >= 0))
        self.assertTrue(np.all(actual.damage.x.array <= 1))

    def test_experimental_amor_couples_to_damage_without_changing_default(self):
        _, result = self.solve(strain=0.002, mechanics_formulation="experimental_amor")
        self.assertEqual(result.report["mechanics_formulation"], "experimental_amor")
        self.assertEqual(
            result.report["history_update"],
            "maximum of previous accepted load-step history and current tensile energy",
        )
        self.assertFalse(result.report["compression_contact"])
        self.assertEqual(len(result.report["steps"]), 3)
        for step in result.report["steps"]:
            for iteration in step["iterations"]:
                nonlinear = iteration["nonlinear_mechanics"]
                self.assertLessEqual(nonlinear["true_residual_norm"], nonlinear["residual_limit"])
        self.assertGreaterEqual(result.report["steps"][-1]["fracture_energy_j"], 0)

    def test_solver_converges_and_preserves_the_initial_crack(self):
        domain, result = self.solve()
        self.assertEqual(len(result.report["steps"]), 3)
        self.assertTrue(all(step["staggered_iterations"] <= 50 for step in result.report["steps"]))
        coordinates = result.damage.function_space.tabulate_dof_coordinates()
        inside = signed_distance(coordinates, self.crack()) <= 0
        self.assertTrue(inside.any())
        np.testing.assert_allclose(result.damage.x.array[inside], 1.0, atol=1e-12)
        self.assertGreater(float(result.damage.x.array[~inside].max()), 0.0)
        self.assertTrue(np.all((result.damage.x.array >= 0.0) & (result.damage.x.array <= 1.0 + 1e-12)))

    def test_damage_and_fracture_energy_are_irreversible(self):
        _, result = self.solve()
        maxima = [step["maximum_damage"] for step in result.report["steps"]]
        fracture = [step["fracture_energy_j"] for step in result.report["steps"]]
        self.assertTrue(all(right >= left - 1e-12 for left, right in zip(maxima, maxima[1:])))
        self.assertTrue(all(right >= left - 1e-8 for left, right in zip(fracture, fracture[1:])))

    def test_outputs_contain_all_primary_fields_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result = self.solve(output=Path(directory))
            expected = {
                "displacement.pvd",
                "damage.pvd",
                "tensile_history.pvd",
                "phase_field_report.json",
                "mechanics_diagnostics.jsonl",
                "iteration_checkpoint.npz",
                "checkpoint_layout.npz",
                "load_step_progress.json",
            }
            self.assertTrue(expected.issubset({path.name for path in Path(directory).iterdir()}))
            self.assertEqual(len(list(Path(directory).glob("displacement_p0_*.vtu"))), 1)
            self.assertEqual(len(list(Path(directory).glob("damage_p0_*.vtu"))), 2)
            self.assertEqual(len(list(Path(directory).glob("tensile_history_p0_*.vtu"))), 1)
            self.assertFalse(result.report["engineering_validated"])
            self.assertFalse(result.report["compression_contact"])
            checkpoint = np.load(Path(directory) / "iteration_checkpoint.npz")
            self.assertEqual(int(checkpoint["step"]), 3)
            np.testing.assert_array_equal(checkpoint["coordinates"], result.damage.function_space.mesh.geometry.x)
            self.assertTrue(np.isfinite(checkpoint["damage"]).all())
            with np.load(Path(directory) / "checkpoint_layout.npz") as layout:
                self.assertEqual(len(layout["cell_input_indices"]), len(layout["history_cell_dofs"]))
                np.testing.assert_array_equal(
                    layout["displacement_cell_dofs"], result.displacement.function_space.dofmap.list,
                )
            progress = json.loads((Path(directory) / "load_step_progress.json").read_text())
            self.assertEqual(len(progress["steps"]), 3)
            self.assertEqual(progress["status"], "partial_progress")
            self.assertEqual(result.report["mechanics_solver"]["ksp_type"], "gmres")
            diagnostics = [
                json.loads(line)
                for line in (Path(directory) / "mechanics_diagnostics.jsonl").read_text().splitlines()
            ]
            starts = [record for record in diagnostics if record["event"] == "started"]
            completed = [record for record in diagnostics if record["event"] == "completed"]
            count = sum(step["staggered_iterations"] for step in result.report["steps"])
            self.assertEqual(len(starts), count)
            self.assertEqual(len(completed), count)
            damage_completed = [record for record in diagnostics if record["event"] == "damage_completed"]
            self.assertEqual(len(damage_completed), count)
            for record in completed + damage_completed:
                self.assertGreaterEqual(record["setup_seconds"], 0)
                self.assertGreaterEqual(record["solve_seconds"], 0)
                self.assertGreater(record["timestamp"], 0)
            for record in completed:
                self.assertGreater(record["reason"], 0)
                self.assertLessEqual(record["relative_true_residual"], self.options().mechanics_relative_tolerance)
            self.assertEqual(len([
                record for record in diagnostics if record["event"] == "staggered_completed"
            ]), count)

    def test_mechanics_failure_records_step_and_iteration_without_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("phase_field_solver.LinearProblem.solve", side_effect=PETSc.Error(91)):
                with self.assertRaisesRegex(RuntimeError, "load step 1/3, staggered iteration 1"):
                    self.solve(output=Path(directory))
            diagnostics = [
                json.loads(line)
                for line in (Path(directory) / "mechanics_diagnostics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([record["event"] for record in diagnostics], ["started", "failed"])
            self.assertEqual(diagnostics[-1]["step"], 1)
            self.assertEqual(diagnostics[-1]["iteration"], 1)
            self.assertFalse((Path(directory) / "phase_field_report.json").exists())

    def test_positive_solver_reason_does_not_override_failed_true_residual(self):
        from dolfinx.fem.petsc import LinearProblem
        original = LinearProblem.solve

        def corrupt_solution(problem):
            displacement = original(problem)
            displacement.x.array[:] = 0.0
            return displacement

        with patch("phase_field_solver.LinearProblem.solve", corrupt_solution):
            with self.assertRaisesRegex(RuntimeError, "true residual .* exceeds tolerance"):
                self.solve()

    def test_mechanics_warm_start_copies_previous_displacement_into_petsc_vector(self):
        from dolfinx.fem.petsc import LinearProblem
        original = LinearProblem.solve
        initial_norms = []

        def inspect_initial_guess(problem):
            self.assertTrue(problem.solver.getInitialGuessNonzero())
            initial_norms.append(problem.x.norm())
            np.testing.assert_allclose(
                problem.x.array, problem.u.x.petsc_vec.array, rtol=0, atol=0,
            )
            return original(problem)

        with patch("phase_field_solver.LinearProblem.solve", inspect_initial_guess):
            self.solve()
        self.assertEqual(initial_norms[0], 0)
        self.assertTrue(any(norm > 0 for norm in initial_norms[1:]))


if __name__ == "__main__":
    unittest.main()
