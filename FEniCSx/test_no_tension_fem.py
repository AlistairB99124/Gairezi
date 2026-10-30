"""Native DOLFINx material patches; not fracture/hydraulic validation."""
import unittest

from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from no_tension import response
from no_tension_fem import NoTensionMaterial


class NoTensionFemTests(unittest.TestCase):
    def test_selected_finite_strength_and_energy_are_applied_only_to_selection(self):
        _, _, displacement, selected = self.fixture()
        selected[::2] = False
        displacement.interpolate(lambda x: .001 * x)
        material = NoTensionMaterial(displacement, 35e9, .2, selected, 1e6)
        stress = material.stress.x.array.reshape(-1, 3, 3)[material.tensor_dofs]
        np.testing.assert_allclose(
            stress[selected], np.broadcast_to(np.eye(3) * 1e6, stress[selected].shape), atol=1e-6,
        )
        np.testing.assert_allclose(
            stress[~selected], np.broadcast_to(np.eye(3) * 35e6 / .6, stress[~selected].shape),
            atol=1e-6,
        )
        expected_energy = 3 * 1e6 * .001 - .5 * 3 * 1e12 * .6 / 35e9
        np.testing.assert_allclose(
            material.energy.x.array[material.scalar_dofs][selected], expected_energy, atol=1e-7,
        )

    def fixture(self, divisions=2):
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, divisions, divisions, divisions)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        field = fem.Function(space)
        cells = domain.topology.index_map(3)
        selected = np.ones(cells.size_local + cells.num_ghosts, dtype=bool)
        return domain, space, field, selected

    def test_selected_cells_release_tension_but_unselected_stay_elastic(self):
        domain, _, displacement, selected = self.fixture()
        selected[::2] = False
        displacement.interpolate(lambda x: .001 * x)
        material = NoTensionMaterial(displacement, 35e9, .2, selected)
        stress = material.stress.x.array.reshape(-1, 3, 3)[material.tensor_dofs]
        np.testing.assert_allclose(stress[selected], 0, atol=1e-7)
        np.testing.assert_allclose(stress[~selected],
                                   np.broadcast_to(np.eye(3) * 35e6 / .6, stress[~selected].shape),
                                   atol=1e-6)
        displacement.interpolate(lambda x: -.001 * x)
        material.update()
        stress = material.stress.x.array.reshape(-1, 3, 3)[material.tensor_dofs]
        np.testing.assert_allclose(stress, np.broadcast_to(-np.eye(3) * 35e6 / .6, stress.shape),
                                   atol=1e-6)
        self.assertAlmostEqual(domain.comm.allreduce(
            fem.assemble_scalar(fem.form(material.energy * ufl.dx)), op=MPI.SUM), 87500., places=6)

    def test_native_stress_and_tangent_match_reference(self):
        _, space, displacement, selected = self.fixture()
        material = NoTensionMaterial(displacement, 35e9, .2, selected)
        increment = fem.Function(space)
        direction = np.array([[.2, .1, 0], [.1, -.3, .04], [0, .04, .1]])
        increment.interpolate(lambda x: direction @ x)
        for strain in (np.diag([-.003, -.002, -.001]), np.diag([-.003, -.002, .004]),
                       np.diag([-.003, .003, .004]), np.diag([.001, .002, .003])):
            displacement.interpolate(lambda x: strain @ x)
            material.update()
            tangent = fem.Function(material.stress.function_space)
            tangent.interpolate(fem.Expression(material.tangent_stress(increment),
                                               tangent.function_space.element.interpolation_points))
            expected = response(strain, 35e9, .2)
            stress = material.stress.x.array.reshape(-1, 3, 3)
            np.testing.assert_allclose(stress, np.broadcast_to(expected.stress_pa, stress.shape), atol=1e-6)
            values = tangent.x.array.reshape(-1, 3, 3)
            np.testing.assert_allclose(values, np.broadcast_to(expected.tangent(direction), values.shape),
                                       atol=1e-4)

    def test_post_event_mixed_patch_re_equilibrates_without_initial_gap(self):
        for divisions in (2, 3):
            with self.subTest(divisions=divisions):
                domain, space, displacement, selected = self.fixture(divisions)
                strain = np.diag([-.0003, -.0002, .0005])
                exact = fem.Function(space)
                exact.interpolate(lambda x: strain @ x)
                displacement.x.array[:] = exact.x.array
                material = NoTensionMaterial(displacement, 35e9, .2, np.zeros_like(selected))
                initial_stress = material.stress.x.array.copy()
                self.assertGreater(float(np.max(initial_stress)), 0)
                material.selected[:] = selected
                boundary = mesh.locate_entities_boundary(
                    domain, 2, lambda x: np.any(np.isclose(x, 0) | np.isclose(x, 1), axis=0),
                )
                fixed = fem.locate_dofs_topological(space, 2, boundary)
                bcs = [fem.dirichletbc(exact, fixed)]
                # Perturb only the initial guess, not the physical boundary data.
                coordinates = space.tabulate_dof_coordinates()
                displacement.x.array[:] += np.repeat(
                    .00001 * np.prod(np.sin(np.pi * coordinates), axis=1), 3,
                )
                material.update()
                test, trial = ufl.TestFunction(space), ufl.TrialFunction(space)
                problem = NonlinearProblem(
                    ufl.inner(material.stress, ufl.sym(ufl.grad(test))) * ufl.dx,
                    displacement, J=ufl.inner(material.tangent_stress(trial), ufl.sym(ufl.grad(test))) * ufl.dx,
                    bcs=bcs, petsc_options_prefix="no_tension_patch_",
                    petsc_options={"snes_type": "newtonls", "snes_rtol": 1e-10, "snes_atol": 1e-7,
                                   "snes_stol": 0., "snes_max_it": 25,
                                   "snes_error_if_not_converged": True,
                                   "ksp_type": "preonly", "pc_type": "lu",
                                   "pc_factor_mat_solver_type": "mumps",
                                   "ksp_error_if_not_converged": True},
                )

                def update(x):
                    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    fem.petsc.assign(x, displacement)
                    material.update()

                def residual(snes, x, vector):
                    update(x)
                    fem.petsc.assemble_residual(snes, x, vector, displacement, problem.F, problem.J, bcs)

                def jacobian(snes, x, matrix, preconditioner):
                    update(x)
                    fem.petsc.assemble_jacobian(snes, x, matrix, preconditioner, displacement,
                                               problem.J, problem.preconditioner, bcs)

                problem.solver.setFunction(residual, problem.b)
                problem.solver.setJacobian(jacobian, problem.A, problem.P_mat)
                problem.solve()
                self.assertGreater(problem.solver.getConvergedReason(), 0)
                self.assertGreater(problem.solver.getIterationNumber(), 0)
                np.testing.assert_allclose(displacement.x.array, exact.x.array, atol=1e-12)
                material.update()
                expected = response(strain, 35e9, .2).stress_pa
                values = material.stress.x.array.reshape(-1, 3, 3)
                np.testing.assert_allclose(values, np.broadcast_to(expected, values.shape), atol=.01)
                vector = problem.b.duplicate()
                try:
                    problem.solver.computeFunction(problem.x, vector)
                    self.assertLess(vector.norm(), 1e-6)
                finally:
                    vector.destroy()

    def test_invalid_mask_and_nonfinite_strain_are_rejected(self):
        _, _, displacement, selected = self.fixture()
        for bad in (selected.astype(int), selected[:-1]):
            with self.assertRaisesRegex(ValueError, "boolean mask"):
                NoTensionMaterial(displacement, 35e9, .2, bad)
        material = NoTensionMaterial(displacement, 35e9, .2, selected)
        displacement.x.array[:] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite strain"):
            material.update()

    def test_incompatible_displacement_spaces_fail_before_coefficient_assembly(self):
        domain, _, _, selected = self.fixture()
        for element in (("DG", 1, (3,)), ("Lagrange", 2, (3,)), ("Lagrange", 1)):
            with self.subTest(element=element), self.assertRaisesRegex(ValueError, "continuous blocked vector P1"):
                NoTensionMaterial(fem.Function(fem.functionspace(domain, element)), 35e9, .2, selected)

    def test_cellwise_materials_match_independent_scalar_laws_and_tangents(self):
        domain, space, displacement, selected = self.fixture()
        young = np.where(np.arange(len(selected)) % 2, 35e9, 50e9)
        poisson = np.where(np.arange(len(selected)) % 2, .2, .25)
        selected[::3] = False
        strain = np.array([[-.003, .0002, .0001], [.0002, .002, -.0003], [.0001, -.0003, .004]])
        direction = np.array([[.2, -.1, .03], [-.1, -.4, .02], [.03, .02, .1]])
        displacement.interpolate(lambda x: strain @ x)
        material = NoTensionMaterial(displacement, young, poisson, selected)
        increment = fem.Function(space)
        increment.interpolate(lambda x: direction @ x)
        tangent = fem.Function(material.stress.function_space)
        tangent.interpolate(fem.Expression(material.tangent_stress(increment),
                                           tangent.function_space.element.interpolation_points))
        stresses = material.stress.x.array.reshape(-1, 3, 3)[material.tensor_dofs]
        tangents = tangent.x.array.reshape(-1, 3, 3)[material.tensor_dofs]
        for cell in range(len(selected)):
            expected = response(strain, float(young[cell]), float(poisson[cell]))
            if selected[cell]:
                np.testing.assert_allclose(stresses[cell], expected.stress_pa, atol=1e-6)
                np.testing.assert_allclose(tangents[cell], expected.tangent(direction), atol=1e-4)
            else:
                mu = young[cell] / (2 * (1 + poisson[cell]))
                lam = young[cell] * poisson[cell] / ((1 + poisson[cell]) * (1 - 2 * poisson[cell]))
                np.testing.assert_allclose(stresses[cell], 2 * mu * strain + lam * np.trace(strain) * np.eye(3),
                                           atol=1e-6)
                np.testing.assert_allclose(tangents[cell], 2 * mu * direction + lam * np.trace(direction) * np.eye(3),
                                           atol=1e-4)
        scalar = fem.functionspace(domain, ("DG", 0))
        young_field, poisson_field = fem.Function(scalar), fem.Function(scalar)
        young_field.x.array[scalar.dofmap.list[:, 0]] = young
        poisson_field.x.array[scalar.dofmap.list[:, 0]] = poisson
        from_fields = NoTensionMaterial(displacement, young_field, poisson_field, selected)
        np.testing.assert_allclose(from_fields.stress.x.array, material.stress.x.array, atol=1e-7)
        young_field.x.array[:] = 1.
        from_fields.update()
        np.testing.assert_allclose(from_fields.stress.x.array, material.stress.x.array, atol=1e-7)

    def test_cellwise_material_inputs_reject_invalid_values_and_foreign_fields(self):
        domain, _, displacement, selected = self.fixture()
        count = len(selected)
        for young, poisson in (
            (np.ones(count - 1), .2), (np.full(count, np.nan), .2), (np.zeros(count), .2),
            (np.full(count, True), .2), (35e9, np.full(count, .5)),
            (35e9, np.full(count, -1.)), (np.ones((count, 1)), .2),
        ):
            with self.subTest(young_shape=np.shape(young), poisson_shape=np.shape(poisson)), self.assertRaises(ValueError):
                NoTensionMaterial(displacement, young, poisson, selected)
        foreign = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        for field in (fem.Function(fem.functionspace(foreign, ("DG", 0))),
                      fem.Function(fem.functionspace(domain, ("Lagrange", 1))),
                      fem.Function(fem.functionspace(domain, ("DG", 0, (3,))))):
            with self.subTest(shape=field.ufl_shape), self.assertRaisesRegex(ValueError, "scalar DG0"):
                NoTensionMaterial(displacement, field, .2, selected)


if __name__ == "__main__":
    unittest.main()
