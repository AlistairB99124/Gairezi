"""Actual 3-D bulk elasticity coupled to aperture-dependent steady surface flow."""
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_vector
from mpi4py import MPI
import numpy as np
import ufl

from fracture_faces import PairedCrackFaces, source_vector_values
from fracture_hydrostatic import HydrostaticBoundary
from fracture_steady_flow import SteadyCouplingOptions, steady_crack_load
from open_crack_equilibrium import solve_open_crack_equilibrium
from interface_springs import InterfaceSprings


class BulkFlowTests(unittest.TestCase):
    def fixture(self):
        source_mesh = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        coordinates, _ = source_vector_values(fem.Function(fem.functionspace(source_mesh, ("Lagrange", 1, (3,)))))
        cells = source_mesh.geometry.input_global_indices[source_mesh.geometry.dofmaps[0]]
        first = coordinates - [1., 0., 0.]
        second = first * [-1., 1., 1.]
        count = len(first)
        points = np.vstack((first, second))
        all_cells = np.vstack((cells, cells[:, [1, 0, 2, 3]] + count))
        surface = set()
        for cell in cells:
            for triangle in combinations(cell, 3):
                if np.all(first[list(triangle), 0] == 0.):
                    surface.add(tuple(sorted(triangle)))
        triangles = np.array(sorted(surface))
        faces = PairedCrackFaces(points, triangles, triangles + count, [[1., 0., 0.]] * len(triangles))
        domain = mesh.create_mesh(MPI.COMM_WORLD, all_cells, source_mesh.ufl_domain(), points)
        domain.topology.create_connectivity(0, 3)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        vertices = np.arange(domain.topology.index_map(0).size_local, dtype=np.int32)
        ends = mesh.locate_entities(domain, 0, lambda x: np.isclose(np.abs(x[0]), 1.))
        bcs = [fem.dirichletbc(0., fem.locate_dofs_topological(space.sub(i), 0, ids), space.sub(i))
               for i, ids in ((0, ends), (1, vertices), (2, vertices))]
        edges = {}
        for triangle in triangles:
            for a, b in combinations(triangle, 2):
                key = tuple(sorted((int(a), int(b))))
                edges[key] = edges.get(key, 0) + 1
        bottom = [edge for edge, owners in edges.items() if owners == 1 and np.all(first[list(edge), 2] == 0.)]
        top = [edge for edge, owners in edges.items() if owners == 1 and np.all(first[list(edge), 2] == 1.)]
        boundaries = [HydrostaticBoundary("reservoir", 4., bottom), HydrostaticBoundary("tailwater", 2., top)]
        return space, faces, np.tile(np.arange(count), 2), boundaries, bcs

    def solve(self, root, *, heads_equal=False, dry_force=1e5, maximum_iterations=150,
              nodal_external_load=False):
        space, faces, parents, boundaries, bcs = self.fixture()
        if heads_equal:
            boundaries[1] = HydrostaticBoundary("tailwater", 4., boundaries[1].inlet_source_edges)
        x = ufl.SpatialCoordinate(space.mesh)
        args = dict(
            elastic_form=lambda trial, test: 1e8 * ufl.inner(
                ufl.sym(ufl.grad(trial)), ufl.sym(ufl.grad(test))) * ufl.dx,
            # Body forces pull the two independent blocks towards their fixed
            # outer supports: the dry elastic solution opens the existing cut.
            external_load=lambda test: fem.Constant(space.mesh, float(dry_force)) * ufl.conditional(
                x[0] < 0, -1., 1.) * test[0] * ufl.dx,
            boundary_conditions=bcs, water_density_kg_m3=1000., gravity_m_s2=9.81,
            saturated_face_mask=np.ones(len(faces.minus), dtype=bool),
            closure_tolerance_m=1e-12,
            options=SteadyCouplingOptions(1e-8, 1e-13, maximum_iterations, .7, 1e-10),
            output=root / "equilibrium",
        )
        if nodal_external_load:
            load = assemble_vector(fem.form(args["external_load"](ufl.TestFunction(space))))
            try:
                args["additional_nodal_forces"] = load.array.copy()
            finally:
                load.destroy()
            args["external_load"] = lambda v: ufl.dot(fem.Constant(space.mesh, np.zeros(3)), v) * ufl.dx
        return solve_open_crack_equilibrium(space, faces, parents, boundaries, **args), faces, parents, boundaries

    def test_bulk_pressure_aperture_feedback_has_independent_equilibrium(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, faces, parents, boundaries = self.solve(root)
            self.assertEqual(result.report["status"], "passed_reduced_open_crack_equilibrium")
            _, dry = source_vector_values(result.dry_displacement)
            _, wet = source_vector_values(result.displacement)
            self.assertGreater(faces.measure(wet)["maximum_opening_m"], faces.measure(dry)["maximum_opening_m"])
            dry_load = steady_crack_load(
                faces, dry, parents, boundaries, water_density_kg_m3=1000., gravity_m_s2=9.81,
                saturated_face_mask=np.ones(len(faces.minus), dtype=bool),
                closure_tolerance_m=1e-12, relative_tolerance=1e-10,
            )
            self.assertGreater(np.max(np.abs(dry_load.vertex_heads_m - result.hydraulic_load.vertex_heads_m)), 1e-4)
            last = result.report["iterations"][-1]
            self.assertLess(last["mechanical_relative_residual"], 1e-8)
            self.assertLess(last["relative_global_force_imbalance"], 1e-8)
            self.assertLessEqual(last["displacement_correction_m"], last["displacement_limit_m"])
            self.assertTrue((root / "equilibrium" / "equilibrium_state.npz").is_file())
            self.assertTrue((root / "equilibrium" / "displacement.pvd").is_file())
            with np.load(root / "equilibrium" / "equilibrium_state.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["displacement_m"], wet)
                np.testing.assert_array_equal(archive["pressure_pa"], result.hydraulic_load.vertex_pressures_pa)
                np.testing.assert_array_equal(archive["nodal_water_forces_n"], result.hydraulic_load.nodal_forces_n)
            self.assertLess(result.report["dry_mechanical_relative_residual"], 1e-8)
            self.assertFalse(result.report["solver_ready"])

    def test_equal_heads_reproduce_hydrostatic_pressure_in_bulk_solve(self):
        with tempfile.TemporaryDirectory() as directory:
            result, faces, _, _ = self.solve(Path(directory), heads_equal=True)
            z = faces.coordinates[faces.minus, 2]
            np.testing.assert_allclose(result.hydraulic_load.vertex_pressures_pa, 9810. * (4 - z), atol=1e-9)

    def test_fixed_nodal_exterior_loads_are_retained_during_water_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            form, _, _, _ = self.solve(root / "form")
            nodal, _, _, _ = self.solve(root / "nodal", nodal_external_load=True)
            np.testing.assert_allclose(nodal.dry_displacement.x.array, form.dry_displacement.x.array, atol=1e-13)
            np.testing.assert_allclose(nodal.displacement.x.array, form.displacement.x.array, atol=1e-13)
            np.testing.assert_allclose(nodal.hydraulic_load.vertex_heads_m, form.hydraulic_load.vertex_heads_m,
                                       atol=1e-12)
            with np.load(root / "nodal" / "equilibrium" / "native_inputs.npz") as saved:
                np.testing.assert_array_equal(saved["assembled_external_force_n"], saved["additional_nodal_forces_n"])
                self.assertGreater(np.linalg.norm(saved["additional_nodal_forces_n"]), 0.)

    def test_existing_mortar_seats_with_imposed_support_motion_are_retained(self):
        original, faces, parents, boundaries, _ = self.fixture()
        count = len(faces.coordinates) // 2
        coordinates = np.vstack((faces.coordinates, faces.coordinates[:count] - [1., 0., 0.],
                                 faces.coordinates[count:] + [1., 0., 0.]))
        cells = original.mesh.geometry.input_global_indices[original.mesh.geometry.dofmaps[0]]
        cells = np.vstack((cells, cells + 2 * count))
        domain = mesh.create_mesh(MPI.COMM_WORLD, cells, original.mesh.ufl_domain(), coordinates)
        domain.topology.create_connectivity(0, 3)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        source = domain.geometry.input_global_indices[domain.geometry.dofmaps[0]]
        source_to_dof = np.empty(len(coordinates), dtype=np.int32)
        source_to_dof[source] = space.dofmap.list
        dof_to_source = np.argsort(source_to_dof)
        seats = mesh.locate_entities_boundary(
            domain, 2, lambda x: np.isclose(np.abs(x[0]), 1.) | np.isclose(x[0], 0.))
        tags = []
        for facet in seats:
            dofs = fem.locate_dofs_topological(space, 2, np.array([facet], dtype=np.int32))
            block = np.unique(dof_to_source[dofs] // count)
            self.assertEqual(len(block), 1)
            if np.allclose(coordinates[dof_to_source[dofs], 0], 0.):
                tags.append(21 + int(block[0]))
            else:
                tags.append((11, 13, 12, 14)[int(block[0])])
        imported = SimpleNamespace(domain=domain, facet_tags=mesh.meshtags(
            domain, 2, seats, np.array(tags, dtype=np.int32)))
        structure = SimpleNamespace(spring_pairs=[
            {"name": name, "tags": pair, "spring_n_per_m3": dict(x=2e8, y=2e8, z=2e8)}
            for name, pair in (("left_seat", (11, 12)), ("right_seat", (13, 14)))])
        springs = InterfaceSprings(imported, structure, space)
        vertices = np.arange(domain.topology.index_map(0).size_local, dtype=np.int32)
        bcs = [fem.dirichletbc(0., fem.locate_dofs_topological(space.sub(i), 0, vertices), space.sub(i))
               for i in (1, 2)]
        for sign in (-1., 1.):
            ids = mesh.locate_entities(domain, 0, lambda x: sign * x[0] >= 1. - 1e-12)
            # Only support-block identities, not the coincident moving seat.
            dofs = fem.locate_dofs_topological(space.sub(0), 0, ids)
            dofs = dofs[dof_to_source[dofs // 3] >= 2 * count]
            bcs.append(fem.dirichletbc(sign * 2e-4, dofs, space.sub(0)))
        force = fem.Constant(domain, 1e5)
        ds = ufl.Measure("ds", domain=domain, subdomain_data=imported.facet_tags)
        faces = PairedCrackFaces(coordinates, faces.minus, faces.plus, faces.normals)
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = solve_open_crack_equilibrium(
                    space, faces, np.concatenate((parents, np.arange(2 * count, 4 * count))),
                    boundaries,
                    elastic_form=lambda u, v: 1e8 * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx,
                    external_load=lambda v: force * v[0] * ds(22) - force * v[0] * ds(21),
                    boundary_conditions=bcs, water_density_kg_m3=1000., gravity_m_s2=9.81,
                    saturated_face_mask=np.ones(len(faces.minus), dtype=bool), closure_tolerance_m=1e-12,
                    options=SteadyCouplingOptions(1e-8, 1e-13, 150, .7, 1e-10),
                    interface_springs=springs, output=Path(directory) / "equilibrium")
                _, dry = source_vector_values(result.dry_displacement)
                np.testing.assert_allclose(dry[faces.plus, 0] - dry[faces.minus, 0], .0034, atol=1e-12)
                self.assertAlmostEqual(result.report["dry_interface_spring_energy_j"], 50., places=7)
                self.assertGreater(result.report["interface_spring_energy_j"], 50.)
                self.assertLess(result.report["iterations"][-1]["mechanical_relative_residual"], 1e-8)
                self.assertLess(result.report["iterations"][-1]["relative_global_force_imbalance"], 1e-8)
                with np.load(Path(directory) / "equilibrium" / "interface_springs.npz") as saved:
                    offsets, columns, values = springs.matrix.getValuesCSR()
                    np.testing.assert_array_equal(saved["indptr"], offsets)
                    np.testing.assert_array_equal(saved["indices"], columns)
                    np.testing.assert_array_equal(saved["data"], values)
                np.testing.assert_array_equal(springs.matrix.getValuesCSR()[2], springs._operator.data)
        finally:
            if springs.matrix is not None:
                springs.matrix.destroy()

    def test_closed_dry_cut_and_iteration_failure_preserve_diagnostics_not_success(self):
        for force, iterations in ((0., 10), (1e5, 1)):
            with self.subTest(force=force), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(RuntimeError, "no coupled state accepted"):
                    self.solve(root, dry_force=force, maximum_iterations=iterations)
                report = json.loads((root / "equilibrium" / "equilibrium_report.json").read_text())
                self.assertEqual(report["status"], "failed_reduced_open_crack_equilibrium")
                self.assertTrue((root / "equilibrium" / "dry_state.npz").is_file())
                self.assertFalse((root / "equilibrium" / "equilibrium_state.npz").exists())
                self.assertFalse((root / "equilibrium" / "displacement.pvd").exists())

    def test_bulk_face_orientation_and_coordinate_provenance_fail_before_output(self):
        space, faces, parents, boundaries, bcs = self.fixture()
        for change in ("coordinates", "normal"):
            coordinates = faces.coordinates.copy()
            normals = faces.normals.copy()
            if change == "coordinates":
                coordinates[:, 1] += .01
            else:
                normals *= -1
            invalid = PairedCrackFaces(coordinates, faces.minus, faces.plus, normals)
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "equilibrium"
                with self.assertRaises(ValueError):
                    solve_open_crack_equilibrium(
                        space, invalid, parents, boundaries,
                        elastic_form=lambda u, v: ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
                        external_load=lambda v: ufl.dot(fem.Constant(space.mesh, np.zeros(3)), v) * ufl.dx,
                        boundary_conditions=bcs, water_density_kg_m3=1000., gravity_m_s2=9.81,
                        saturated_face_mask=np.ones(len(faces.minus), dtype=bool),
                        closure_tolerance_m=1e-12,
                        options=SteadyCouplingOptions(1e-8, 1e-13, 10, .7, 1e-10), output=output,
                    )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
