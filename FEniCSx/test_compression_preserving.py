"""Synthetic 3-D patch checks; not dam engineering or contact validation."""
import unittest
from unittest.mock import patch

from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from compression_preserving import MechanicsProblem, elastic_energy, solve_mechanics, stress
from phase_field_solver import elastic_parameters


class CompressionPreservingTests(unittest.TestCase):
    def test_positive_snes_reason_cannot_bypass_independent_equilibrium_check(self):
        from dolfinx.fem.petsc import NonlinearProblem

        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        displacement = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        force = fem.Constant(domain, np.array([0, 0, -1], dtype=PETSc.ScalarType))

        original_solve = NonlinearProblem.solve

        def permissive_solve(problem):
            problem.solver.setTolerances(atol=1e20)
            return original_solve(problem)

        with patch("compression_preserving.NonlinearProblem.solve", permissive_solve):
            with self.assertRaisesRegex(RuntimeError, "compression-preserving mechanics failed"):
                solve_mechanics(
                    displacement, damage, 1000, 400, 1e-8,
                    lambda test, _damage: ufl.inner(force, test) * ufl.dx,
                )

    def test_nearly_converged_warm_start_uses_load_not_tiny_initial_residual(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        displacement = fem.Function(space)
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        exterior = mesh.locate_entities_boundary(
            domain, 2, lambda x: np.isclose(x[0], 0),
        )
        bcs = [fem.dirichletbc(
            np.zeros(3, dtype=PETSc.ScalarType),
            fem.locate_dofs_topological(space, 2, exterior), space,
        )]
        force = fem.Constant(domain, np.array([0, 0, -1], dtype=PETSc.ScalarType))
        work = lambda test, _damage: ufl.inner(force, test) * ufl.dx
        for _ in range(2):
            report = solve_mechanics(displacement, damage, 1000, 400, 1e-8, work, bcs=bcs)
        self.assertEqual(report["nonlinear_iterations"], 0)
        self.assertEqual(report["residual_reference"], "applied load")
        self.assertLess(report["initial_residual_norm"], report["load_vector_norm"] * 1e-8)
        self.assertLessEqual(report["true_residual_norm"], report["residual_limit"])
        self.assertAlmostEqual(
            report["linear_absolute_tolerance"], 0.1 * report["residual_limit"],
        )

    def test_reused_problem_reassembles_load_and_damage_and_matches_fresh_solve(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        displacement = fem.Function(space)
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        exterior = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 0))
        bcs = [fem.dirichletbc(
            np.zeros(3, dtype=PETSc.ScalarType),
            fem.locate_dofs_topological(space, 2, exterior), space,
        )]
        force = fem.Constant(domain, np.array([0, 0, -1], dtype=PETSc.ScalarType))
        work = lambda test, _damage: ufl.inner(force, test) * ufl.dx
        problem = MechanicsProblem(displacement, damage, 1000, 400, 1e-8, work, bcs=bcs)
        for alpha, load in ((0.0, -1), (0.1, -2), (0.8, -3)):
            damage.x.array[:] = alpha
            force.value = np.array([0, 0, load], dtype=PETSc.ScalarType)
            report = problem.solve()
            fresh = fem.Function(space)
            solve_mechanics(fresh, damage, 1000, 400, 1e-8, work, bcs=bcs)
            np.testing.assert_allclose(displacement.x.array, fresh.x.array, rtol=1e-6, atol=1e-10)
            self.assertLessEqual(report["true_residual_norm"], report["residual_limit"])

    def test_stress_is_the_energy_derivative_in_both_volumetric_branches(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        displacement, direction = fem.Function(space), fem.Function(space)
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        damage.x.array[:] = 0.8
        direction.interpolate(lambda x: np.vstack((x[0] + x[1], 0.3*x[1], -0.2*x[2])))
        _, mu, bulk = elastic_parameters(1000, 0.2)
        delta = 1e-7
        for sign in (-1, 1):
            displacement.interpolate(lambda x: sign * np.vstack((0.003*x[0], -0.001*x[1], 0.002*x[2])))
            expected = fem.assemble_scalar(fem.form(ufl.inner(
                stress(displacement, damage, bulk, mu, 1e-8), ufl.sym(ufl.grad(direction)),
            ) * ufl.dx))
            original = displacement.x.array.copy()
            energies = []
            for perturbation in (-delta, delta):
                displacement.x.array[:] = original + perturbation * direction.x.array
                energies.append(fem.assemble_scalar(fem.form(
                    elastic_energy(displacement, damage, bulk, mu, 1e-8) * ufl.dx,
                )))
            self.assertAlmostEqual((energies[1] - energies[0]) / (2 * delta), expected, places=8)

    def test_failure_does_not_return_a_success_report(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        displacement = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        with patch(
            "compression_preserving.NonlinearProblem.solve",
            side_effect=PETSc.Error(91),
        ):
            with self.assertRaisesRegex(RuntimeError, "compression-preserving mechanics failed"):
                solve_mechanics(
                    displacement, damage, 1000, 400, 1e-8,
                    lambda test, _damage: ufl.inner(fem.Constant(
                        domain, np.zeros(3, dtype=PETSc.ScalarType),
                    ), test) * ufl.dx,
                )

    def test_pressure_closure_reopening_and_shear_with_direct_solver(self):
        self.check_pressure_cycles(direct_solver=True)

    def test_pressure_closure_reopening_and_shear_with_gmres(self):
        self.check_pressure_cycles(direct_solver=False)

    def check_pressure_cycles(self, *, direct_solver):
        _, mu, bulk = elastic_parameters(1000.0, 0.2)
        kappa = 1e-8
        for divisions in (2, 4):
            domain = mesh.create_unit_cube(MPI.COMM_WORLD, divisions, divisions, divisions)
            space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
            damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
            displacement = fem.Function(space)
            scalar_space = fem.functionspace(domain, ("DG", 0))
            energy = fem.Function(scalar_space)
            bcs = []
            for component in range(3):
                facets = mesh.locate_entities_boundary(
                    domain, 2, lambda x, i=component: np.isclose(x[i], 0),
                )
                dofs = fem.locate_dofs_topological(space.sub(component), 2, facets)
                bcs.append(fem.dirichletbc(PETSc.ScalarType(0), dofs, space.sub(component)))
            normal = ufl.FacetNormal(domain)
            for alpha in ((0.0, 0.99, 1.0) if direct_solver else (0.0, 0.99)):
                damage.x.array[:] = alpha
                factor = (1 - alpha)**2 + kappa
                # Start in tension, close, then reopen; retain the previous solution.
                for sign in (1, -1, 1):
                    with self.subTest(divisions=divisions, damage=alpha, sign=sign):
                        dilation = sign * 1e-4
                        coefficient = factor if sign > 0 else 1.0
                        pressure = 3 * bulk * dilation * coefficient
                        report = solve_mechanics(
                            displacement, damage, bulk, mu, kappa,
                            lambda test, _damage: ufl.inner(pressure * normal, test) * ufl.ds,
                            bcs=bcs, relative_tolerance=1e-10, absolute_tolerance=1e-13,
                            direct_solver=direct_solver,
                        )
                        np.testing.assert_allclose(
                            displacement.x.array.reshape(-1, 3),
                            dilation * space.tabulate_dof_coordinates(),
                            rtol=1e-6, atol=1e-10,
                        )
                        self.assertLessEqual(report["true_residual_norm"], report["residual_limit"])
                        self.assertFalse(report["compression_contact"])
                        np.testing.assert_array_equal(damage.x.array, alpha)
                        energy.interpolate(fem.Expression(
                            elastic_energy(displacement, damage, bulk, mu, kappa),
                            scalar_space.element.interpolation_points,
                        ))
                        np.testing.assert_allclose(
                            energy.x.array, 0.5 * bulk * (3 * dilation)**2 * coefficient,
                            rtol=1e-6, atol=1e-17,
                        )
            # Pure shear remains degraded: do not mislabel this as full contact.
            damage.x.array[:] = 1.0
            displacement.interpolate(lambda x: np.vstack((1e-4*x[1], 0*x[0], 0*x[0])))
            tensor_space = fem.functionspace(domain, ("DG", 0, (3, 3)))
            tensor = fem.Function(tensor_space)
            tensor.interpolate(fem.Expression(
                stress(displacement, damage, bulk, mu, kappa),
                tensor_space.element.interpolation_points,
            ))
            np.testing.assert_allclose(tensor.x.array.reshape(-1, 3, 3)[:, 0, 1], mu*kappa*1e-4)


if __name__ == "__main__":
    unittest.main()
