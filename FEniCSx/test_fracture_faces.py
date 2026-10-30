"""Paired-face opening, pressure work and source-identity verification."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl

from dam_mesh import FACES, import_structure
from fracture_faces import PairedCrackFaces, source_vector_values, bonded_node_representatives, write_surface_state
from fracture_topology import split_interior_wall_faces


class FractureFaceTests(unittest.TestCase):
    def fixture(self):
        points = np.asarray([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.)] * 2)
        return points, PairedCrackFaces(points, [[0, 1, 2]], [[3, 4, 5]], [[1., 0., 0.]])

    def test_surface_vtu_retains_both_lips_true_opening_shear_pressure_and_identities(self):
        import meshio
        points, faces = self.fixture()
        displacement = np.zeros_like(points)
        displacement[3:] = [[.001, .01, 0.], [.002, 0., .02], [-.0001, .03, 0.]]
        for wet in (True, False):
            with self.subTest(wet=wet), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "surface.vtu"
                pressure = np.array([[1e4, 2e4, 3e4]]) if wet else np.zeros((1, 3))
                write_surface_state(
                    path, faces, displacement, released_face_mask=np.array([True]),
                    wet_face_mask=np.array([wet]), vertex_pressures_pa=pressure,
                    parent_node_ids=np.tile(np.arange(3), 2))
                result = meshio.read(path)
                np.testing.assert_array_equal(result.points, points)
                np.testing.assert_array_equal(result.cells[0].data, [[0, 1, 2], [3, 4, 5]])
                np.testing.assert_array_equal(result.point_data["source_node_id"], np.arange(6))
                np.testing.assert_array_equal(result.point_data["original_parent_node_id"], [0, 1, 2] * 2)
                np.testing.assert_array_equal(result.point_data["displacement_m"], displacement)
                np.testing.assert_allclose(result.point_data["signed_gap_m"], [0.001, .002, -.0001] * 2)
                np.testing.assert_allclose(result.point_data["opening_m"], [.001, .002, 0.] * 2)
                np.testing.assert_allclose(result.point_data["tangential_jump_m"],
                                           np.tile([[0., .01, 0.], [0., 0., .02], [0., .03, 0.]], (2, 1)))
                np.testing.assert_array_equal(result.point_data["water_pressure_pa"], np.tile(pressure[0], 2))
                np.testing.assert_array_equal(result.cell_data["lip_side"][0], [-1, 1])
                np.testing.assert_array_equal(result.cell_data["wet"][0], [int(wet)] * 2)
                np.testing.assert_array_equal(result.cell_data["released"][0], [1, 1])

    def test_surface_export_rejects_water_on_bonds_or_pressure_on_declared_dry_faces(self):
        points, faces = self.fixture()
        for released, wet, pressure in (
            ([False], [True], [[1., 2., 3.]]), ([True], [False], [[1., 2., 3.]]),
            ([True], [True], [[-1., 2., 3.]]), ([1], [True], [[1., 2., 3.]]),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.vtu"
                with self.assertRaises(ValueError):
                    write_surface_state(
                        path, faces, np.zeros_like(points), released_face_mask=np.array(released),
                        wet_face_mask=np.array(wet), vertex_pressures_pa=np.array(pressure))
                self.assertFalse(path.exists())

    def test_opening_is_normal_jump_not_translation_or_shear(self):
        points, faces = self.fixture()
        displacement = np.full_like(points, 3.)
        displacement[3:] += [[.001, .5, 0.], [.002, 0., .1], [-.001, 0., 0.]]
        report = faces.measure(displacement)
        self.assertAlmostEqual(report["maximum_opening_m"], .002, places=14)
        self.assertAlmostEqual(report["minimum_signed_gap_m"], -.001, places=14)
        np.testing.assert_array_equal(report["peak_location_m"], points[1])
        np.testing.assert_allclose(report["tangential_vertex_jumps_m"][0],
                                   [[0., .5, 0.], [0., 0., .1], [0., 0., 0.]], atol=1e-14)
        displacement[3:, 0] = 2.
        closed = faces.measure(displacement)
        self.assertEqual(closed["maximum_opening_m"], 0.)
        self.assertLess(closed["minimum_signed_gap_m"], 0.)

    def test_bond_representatives_are_transitive_not_coordinate_welds(self):
        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., 1.)] * 4)
        faces = PairedCrackFaces(points, [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                                 [[3, 4, 5], [6, 7, 8], [9, 10, 11]], [[1., 0., 0.]] * 3)
        mask = np.array([True, True, False])
        representatives = bonded_node_representatives(faces, mask)
        np.testing.assert_array_equal(representatives, [0, 1, 2, 0, 1, 2, 0, 1, 2, 9, 10, 11])
        reverse = PairedCrackFaces(points, faces.plus[::-1], faces.minus[::-1], -faces.normals[::-1])
        np.testing.assert_array_equal(
            bonded_node_representatives(reverse, mask[::-1]), representatives)
        for invalid in ([True], np.ones(3, dtype=int)):
            with self.assertRaises(ValueError):
                bonded_node_representatives(faces, invalid)

    def test_pressure_loads_have_exact_force_and_virtual_work(self):
        points, faces = self.fixture()
        pressure = np.array([[1., 2., 3.]]) * 1e5
        force = faces.pressure_forces(pressure)
        np.testing.assert_allclose(force.sum(axis=0), 0., atol=1e-10)
        np.testing.assert_allclose(force[3:].sum(axis=0), [1e5, 0., 0.], atol=1e-10)
        displacement = np.zeros_like(points)
        displacement[3:, 0] = [.001, .002, .003]
        mass = .5 / 12 * (np.ones((3, 3)) + np.eye(3))
        expected = pressure[0] @ mass @ displacement[3:, 0]
        self.assertAlmostEqual(np.sum(force * displacement), expected, places=10)
        np.testing.assert_allclose(np.cross(points, force).sum(axis=0), 0., atol=1e-10)

    def test_rigid_rotation_of_reference_preserves_measured_opening_and_load(self):
        points, faces = self.fixture()
        rotation, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(3, 3)))
        rotated = PairedCrackFaces(points @ rotation.T, [[0, 1, 2]], [[3, 4, 5]],
                                  np.array([[1., 0., 0.]]) @ rotation.T)
        displacement = np.zeros_like(points)
        displacement[3:] = [.002, .001, -.003]
        self.assertAlmostEqual(rotated.measure(displacement @ rotation.T)["maximum_opening_m"],
                               faces.measure(displacement)["maximum_opening_m"], places=14)
        np.testing.assert_allclose(rotated.pressure_forces([[1e5] * 3]),
                                   faces.pressure_forces([[1e5] * 3]) @ rotation.T, atol=1e-10)

    def test_adjacent_faces_may_have_opposite_owner_order_without_merging_lips(self):
        points = np.array([(0., 0., 0.), (0., 1., 0.), (0., 1., 1.), (0., 0., 1.)] * 2)
        faces = PairedCrackFaces(points, [[0, 1, 2], [4, 6, 7]], [[4, 5, 6], [0, 2, 3]],
                                 [[1., 0., 0.], [-1., 0., 0.]])
        state = np.zeros_like(points)
        state[4:, 0] = .002
        np.testing.assert_allclose(faces.measure(state)["signed_vertex_gaps_m"], .002)
        np.testing.assert_allclose(faces.pressure_forces([[1e5] * 3] * 2).sum(axis=0), 0., atol=1e-10)

    def test_native_3d_pressure_generates_analytical_face_opening(self):
        self.check_native_pressure_opening(hydrostatic=False)

    def test_native_drained_hydrostatic_head_generates_self_consistent_opening(self):
        self.check_native_pressure_opening(hydrostatic=True)

    def check_native_pressure_opening(self, *, hydrostatic):
        cube = np.array([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.),
                         (0., 0., 1.), (1., 0., 1.), (1., 1., 1.), (0., 1., 1.)])
        points = np.vstack((cube, [(2., 0., 0.), (2., 1., 0.), (2., 0., 1.), (2., 1., 1.)])) - [1., 0., 0.]
        axis = 2 if hydrostatic else 0
        if hydrostatic:
            points = points[:, [2, 1, 0]]
        source_cells = [(5, tuple(range(8))), (5, (1, 8, 9, 2, 5, 10, 11, 6))]
        interface = {1, 2, 5, 6}
        structure = SimpleNamespace(
            nodes=points, cells=source_cells,
            region_ids=[1, 1],
            boundaries=[(10 + side, tuple(cell[i] for i in pattern))
                        for side, (_, cell) in enumerate(source_cells) for pattern in FACES[5]
                        if {cell[i] for i in pattern} != interface],
        )
        imported = import_structure(structure, 1e-12)
        cut = split_interior_wall_faces(imported.points, imported.tetrahedra, imported.regions,
                                        [[1, 2, 6], [1, 6, 5]])
        self.assertEqual(len(imported.points), 12)
        self.assertEqual(len(cut.coordinates_m), 16)
        domain = mesh.create_mesh(imported.domain.comm, cut.cells, imported.domain.ufl_domain(),
                                  cut.coordinates_m)
        domain.topology.create_connectivity(0, 3)
        faces = cut.faces
        points = cut.coordinates_m
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        field = fem.Function(space)
        coordinates, initial = source_vector_values(field)
        np.testing.assert_allclose(coordinates, points, atol=1e-15)
        self.assertEqual(faces.measure(initial)["maximum_opening_m"], 0.)
        accepted = .00001 * imported.points
        np.testing.assert_array_equal(faces.measure(cut.transfer_nodal_values(accepted))["signed_vertex_gaps_m"], 0.)
        source = domain.geometry.input_global_indices[domain.geometry.dofmaps[0]]
        block_dofs = space.dofmap.list
        source_to_dof = np.empty(len(points), dtype=np.int32)
        source_to_dof[source] = block_dofs
        # nu=0 gives an exact independent axial solution, with no lateral strain.
        vertices = np.arange(domain.topology.index_map(0).size_local, dtype=np.int32)
        ends = mesh.locate_entities(domain, 0, lambda x: np.isclose(np.abs(x[axis]), 1.))
        bcs = [fem.dirichletbc(0., fem.locate_dofs_topological(space.sub(component), 0, ids),
                              space.sub(component))
               for component, ids in ((i, ends if i == axis else vertices) for i in range(3))]
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        young, pressure = 35e9, 1e6
        nodal_load = faces.pressure_forces(np.full((len(faces.minus), 3), pressure))
        if hydrostatic:
            from fracture_hydrostatic import HydrostaticBoundary, drained_crack_load
            boundary = HydrostaticBoundary("reservoir", pressure / (1000. * 9.81), [[1, 2]])
            def water_load(state):
                return drained_crack_load(
                    faces, state, cut.parent_node_ids, [boundary],
                    water_density_kg_m3=1000., gravity_m_s2=9.81,
                    saturated_face_mask=np.ones(len(faces.minus), dtype=bool),
                    previously_wet_face_mask=np.zeros(len(faces.minus), dtype=bool),
                    closure_tolerance_m=1e-12,
                )
            # Use the analytical solution only as an access-state candidate;
            # the native solve starts at zero and imposes no displacement gap.
            candidate = np.zeros_like(points)
            for minus, plus, normal in zip(faces.minus, faces.plus, faces.normals):
                candidate[minus] = -pressure / young * normal
                candidate[plus] = pressure / young * normal
            nodal_load = water_load(candidate).nodal_forces_n
        # Consistent face force equals integral of a P1 traction represented
        # here as an equivalent nodal functional assembled into the RHS.
        zero = fem.Constant(domain, np.zeros(3))
        problem = LinearProblem(
            young * ufl.inner(ufl.sym(ufl.grad(trial)), ufl.sym(ufl.grad(test))) * ufl.dx,
            ufl.dot(zero, test) * ufl.dx, u=field, bcs=bcs,
            petsc_options_prefix="crack_face_pressure_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        )
        problem.A.zeroEntries()
        from dolfinx.fem.petsc import assemble_matrix
        assemble_matrix(problem.A, problem.a, bcs=bcs)
        problem.A.assemble()
        problem.b.array[:] = 0.
        problem.b.array.reshape(-1, 3)[source_to_dof] = nodal_load
        problem.solver.solve(problem.b, field.x.petsc_vec)
        field.x.scatter_forward()
        self.assertGreater(problem.solver.getConvergedReason(), 0)
        _, values = source_vector_values(field)
        if hydrostatic:
            accepted_water = water_load(values)
            np.testing.assert_allclose(accepted_water.nodal_forces_n, nodal_load, atol=1e-10)
            np.testing.assert_allclose(accepted_water.vertex_pressures_pa, pressure, atol=1e-9)
        measured = faces.measure(values)
        if hydrostatic:
            from horizontal_fracture_contract import EquilibriumOpeningObservation, assess_equilibrium_opening
            accepted = EquilibriumOpeningObservation(
                "final_equilibrium", measured["maximum_opening_m"],
                tuple(measured["peak_location_m"]), True,
            )
            for limit, status in ((.002, "below_limit"),
                                  (measured["maximum_opening_m"], "limit_reached")):
                assessment = assess_equilibrium_opening(limit, (accepted,), completed=True)
                self.assertEqual(assessment["opening_limit_status"], status)
                self.assertTrue(assessment["completed"])
                self.assertNotIn("peak_opening_time_s", assessment)
            self.assertIn("not a solver error", assessment["warnings"][0]["message"])
        self.assertAlmostEqual(measured["maximum_opening_m"], 2 * pressure / young, places=13)
        np.testing.assert_allclose(np.asarray(measured["signed_vertex_gaps_m"]), 2 * pressure / young,
                                   atol=1e-13)
        self.assertAlmostEqual(measured["minimum_signed_gap_m"], 2 * pressure / young, places=13)
        unconstrained = assemble_matrix(problem.a)
        unconstrained.assemble()
        residual = field.x.petsc_vec.duplicate()
        try:
            unconstrained.mult(field.x.petsc_vec, residual)
            residual.array.reshape(-1, 3)[source_to_dof] -= nodal_load
            fixed_dofs = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs]))
            free_dofs = np.setdiff1d(np.arange(len(residual.array)), fixed_dofs)
            self.assertLess(np.max(np.abs(residual.array[free_dofs])) / pressure, 1e-12)
            np.testing.assert_allclose(residual.array.reshape(-1, 3).sum(axis=0), 0., atol=1e-8)
        finally:
            residual.destroy()
            unconstrained.destroy()
        stress = fem.Function(fem.functionspace(domain, ("DG", 0)))
        stress.interpolate(fem.Expression(young * field[axis].dx(axis), stress.function_space.element.interpolation_points))
        np.testing.assert_allclose(stress.x.array, -pressure, rtol=1e-12)

    def test_native_source_vector_rejects_discontinuous_or_wrong_rank_fields(self):
        from mpi4py import MPI
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        for element in (("DG", 1, (3,)), ("Lagrange", 2, (3,)), ("Lagrange", 1)):
            with self.subTest(element=element), self.assertRaisesRegex(ValueError, "serial blocked vector P1"):
                source_vector_values(fem.Function(fem.functionspace(domain, element)))

    def test_invalid_geometry_and_values_fail_explicitly(self):
        points, faces = self.fixture()
        for minus, plus, normals in (
            ([[0, 1, 2]], [[0, 1, 2]], [[1., 0., 0.]]),
            ([[0, 1, 2]], [[3, 5, 4]], [[1., 0., 0.]]),
            ([[0, 1, 2]], [[3, 4, 5]], [[0., 1., 0.]]),
            ([[0, 1, 2]], [[3, 4, 5]], [[2., 0., 0.]]),
            ([[0, 1, 9]], [[3, 4, 5]], [[1., 0., 0.]]),
            ([[0, 1, 1]], [[3, 4, 5]], [[1., 0., 0.]]),
        ):
            with self.subTest(minus=minus, plus=plus, normals=normals), self.assertRaises(ValueError):
                PairedCrackFaces(points, minus, plus, normals)
        for pressure in ([[1., -1., 0.]], [[float("nan")] * 3], [1.], [[True] * 3]):
            with self.subTest(pressure=pressure), self.assertRaises(ValueError):
                faces.pressure_forces(pressure)
        with self.assertRaises(ValueError):
            faces.measure(np.full_like(points, np.nan))


if __name__ == "__main__":
    unittest.main()
