"""Principal-strain constitutive and fixed-damage mechanics benchmarks."""
import unittest

from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from unittest.mock import patch

from compression_preserving import solve_mechanics
from phase_field_solver import elastic_parameters
from spectral_split import SpectralSplit, positive_spectrum
import test_phase_field_solver


class SpectralSplitTests(unittest.TestCase):
    def test_nonfinite_strain_is_rejected_explicitly(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        displacement = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        displacement.x.array[:] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite strain"):
            SpectralSplit(displacement, damage, 1000, 400, 1e-8)

    def test_positive_snes_reason_does_not_bypass_spectral_equilibrium(self):
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
                    lambda test, _damage: ufl.inner(force, test) * ufl.dx, spectral=True,
                )

    def test_accelerated_spectral_patch_matches_unaccelerated_fields_and_energies(self):
        fixture = test_phase_field_solver.StaggeredSolverTests()
        _, expected = fixture.solve(strain=0.03, mechanics_formulation="experimental_spectral")
        _, actual = fixture.solve(
            strain=0.03, mechanics_formulation="experimental_spectral", damage_acceleration=True,
        )
        np.testing.assert_allclose(actual.damage.x.array, expected.damage.x.array, rtol=1e-3, atol=5e-4)
        np.testing.assert_allclose(actual.displacement.x.array, expected.displacement.x.array, rtol=1e-3, atol=1e-5)
        for step, reference in zip(actual.report["steps"], expected.report["steps"], strict=True):
            for key in ("elastic_energy_j", "fracture_energy_j"):
                np.testing.assert_allclose(step[key], reference[key], rtol=1e-3, atol=1e-6)
            self.assertLessEqual(
                step["iterations"][-1]["staggered_residual"], fixture.options().staggered_tolerance,
            )
            self.assertEqual(step["iterations"][-1]["damage_relaxation_factor"], 1.0)

    def test_coupled_spectral_patch_accepts_and_reports_consistent_energy(self):
        fixture = test_phase_field_solver.StaggeredSolverTests()
        _, result = fixture.solve(strain=0.002, mechanics_formulation="experimental_spectral")
        self.assertEqual(result.report["mechanics_formulation"], "experimental_spectral")
        self.assertEqual(result.report["energy_split"], "spectral_principal_strain")
        self.assertEqual(len(result.report["steps"]), 3)
        for step in result.report["steps"]:
            np.testing.assert_allclose(
                step["total_internal_energy_j"], step["elastic_energy_j"] + step["fracture_energy_j"],
            )
            self.assertLessEqual(
                step["iterations"][-1]["staggered_residual"], fixture.options().staggered_tolerance,
            )

    def test_positive_matrix_part_derivative_and_repeated_eigenvalues(self):
        rotation, _ = np.linalg.qr(np.array([[1, 2, 3], [2, -1, 1], [1, 1, -1.0]]))
        direction = np.array([[0.1, 0.2, -0.1], [0.2, -0.3, 0.05], [-0.1, 0.05, 0.2]])
        for eigenvalues in ((-0.4, -0.1, 0.2), (-0.2, -0.2, 0.3), (0.2, 0.2, 0.2)):
            tensor = rotation @ np.diag(eigenvalues) @ rotation.T
            basis, positive, derivative = positive_spectrum(tensor)
            actual = basis @ (derivative * (basis.T @ direction @ basis)) @ basis.T

            def positive_part(value):
                q, d, _ = positive_spectrum(value)
                return q @ np.diag(d) @ q.T

            delta = 1e-6
            expected = (positive_part(tensor + delta * direction) - positive_part(
                tensor - delta * direction,
            )) / (2 * delta)
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(basis @ np.diag(positive) @ basis.T, positive_part(tensor))
        _, _, derivative = positive_spectrum(np.zeros((3, 3)))
        np.testing.assert_array_equal(derivative, 1.0)

    def test_energy_stress_and_tangent_are_consistent_in_mixed_and_repeated_branches(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        displacement, direction = fem.Function(space), fem.Function(space)
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        damage.interpolate(lambda x: 0.4 + 0.2 * x[0])
        _, mu, bulk = elastic_parameters(1000, 0.2)
        law = SpectralSplit(displacement, damage, bulk, mu, 1e-8)
        direction.interpolate(lambda x: np.array([
            0.1*x[0] + 0.2*x[1], 0.2*x[0] - 0.3*x[1], 0.1*x[2],
        ]))
        tensor_space = fem.functionspace(domain, ("DG", 0, (3, 3)))

        def stress_values():
            law.update()
            result = fem.Function(tensor_space)
            result.interpolate(fem.Expression(law.stress, tensor_space.element.interpolation_points))
            return result.x.array.copy()

        for diagonal in ((-0.004, -0.001, 0.002), (-0.002, -0.002, 0.003), (0.002, 0.002, 0.002)):
            displacement.interpolate(lambda x: np.asarray(diagonal)[:, None] * x)
            law.update()
            expected_energy_derivative = fem.assemble_scalar(fem.form(
                ufl.inner(law.stress, ufl.sym(ufl.grad(direction))) * ufl.dx,
            ))
            tangent = fem.Function(tensor_space)
            tangent.interpolate(fem.Expression(
                law.tangent_stress(direction), tensor_space.element.interpolation_points,
            ))
            original = displacement.x.array.copy()
            energies, stresses = [], []
            delta = 1e-7
            for sign in (-1, 1):
                displacement.x.array[:] = original + sign * delta * direction.x.array
                stresses.append(stress_values())
                energies.append(fem.assemble_scalar(fem.form(law.energy * ufl.dx)))
            np.testing.assert_allclose(
                (energies[1] - energies[0]) / (2 * delta), expected_energy_derivative, rtol=1e-7,
            )
            np.testing.assert_allclose(
                (stresses[1] - stresses[0]) / (2 * delta), tangent.x.array, rtol=1e-6, atol=1e-6,
            )

    def test_fully_damaged_nonhydrostatic_compression_retains_stiffness(self):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        displacement = fem.Function(space)
        damage = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        damage.x.array[:] = 1.0
        lam, mu, bulk = elastic_parameters(1000, 0.2)
        exact_strain = np.diag([-1e-4, -2e-4, -3e-4])
        exact_stress = 2 * mu * exact_strain + lam * np.trace(exact_strain) * np.eye(3)
        stress_constant = fem.Constant(domain, exact_stress.astype(PETSc.ScalarType))
        normal = ufl.FacetNormal(domain)
        bcs = []
        for component in range(3):
            facets = mesh.locate_entities_boundary(
                domain, 2, lambda x, i=component: np.isclose(x[i], 0),
            )
            dofs = fem.locate_dofs_topological(space.sub(component), 2, facets)
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0), dofs, space.sub(component)))
        report = solve_mechanics(
            displacement, damage, bulk, mu, 1e-8,
            lambda test, _damage: ufl.inner(stress_constant * normal, test) * ufl.ds,
            bcs=bcs, spectral=True, relative_tolerance=1e-10,
        )
        np.testing.assert_allclose(
            displacement.x.array.reshape(-1, 3),
            space.tabulate_dof_coordinates() @ exact_strain, rtol=1e-6, atol=1e-10,
        )
        self.assertLessEqual(report["true_residual_norm"], report["residual_limit"])
        law = SpectralSplit(displacement, damage, bulk, mu, 1e-8)
        self.assertLess(abs(fem.assemble_scalar(fem.form(law.tensile_energy * ufl.dx))), 1e-20)
