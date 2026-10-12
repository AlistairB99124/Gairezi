"""Reduced verification problems for the coupled AT2 phase-field solver."""
import tempfile
from pathlib import Path
import unittest

from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from crack_field import CrackSpecification, PhaseFieldSolverOptions, signed_distance
from phase_field_solver import (
    degradation,
    elastic_parameters,
    solve_staggered,
    tensile_energy,
)


class ConstitutiveTests(unittest.TestCase):
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


class StaggeredSolverTests(unittest.TestCase):
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

    def solve(self, strain=0.2, output=None):
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
        )
        return domain, result

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
            }
            self.assertTrue(expected.issubset({path.name for path in Path(directory).iterdir()}))
            self.assertEqual(len(list(Path(directory).glob("displacement_p0_*.vtu"))), 1)
            self.assertEqual(len(list(Path(directory).glob("damage_p0_*.vtu"))), 2)
            self.assertEqual(len(list(Path(directory).glob("tensile_history_p0_*.vtu"))), 1)
            self.assertFalse(result.report["engineering_validated"])
            self.assertFalse(result.report["compression_contact"])


if __name__ == "__main__":
    unittest.main()
