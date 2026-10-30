"""Analytical clipped hydrostatic force, moment and native virtual-work checks."""
from types import SimpleNamespace
import unittest

import numpy as np

from hydrostatic_load import triangle_pressure_weights


class HydrostaticTriangleTests(unittest.TestCase):
    def setUp(self):
        self.triangle = np.array(((0., 0., 0.), (0., 1., 0.), (0., 0., 1.)))

    def test_partial_submergence_exact_force_and_moments(self):
        head = .5
        weights = triangle_pressure_weights(self.triangle, head, 1.)
        # Integral over 0<z<head, 0<y<1-z of (head-z).
        force = head**2 / 2 - head**3 / 6
        z_moment = head**3 / 6 - head**4 / 12
        y_moment = head**2 / 4 - head**3 / 6 + head**4 / 24
        self.assertAlmostEqual(weights.sum(), force, places=14)
        np.testing.assert_allclose(weights @ self.triangle, [0., y_moment, z_moment], atol=1e-15)
        np.testing.assert_allclose(weights, [force - y_moment - z_moment, y_moment, z_moment], atol=1e-15)

    def test_dry_waterline_full_and_zero_gradient(self):
        np.testing.assert_array_equal(triangle_pressure_weights(self.triangle, -1., 10.), 0.)
        np.testing.assert_array_equal(triangle_pressure_weights(self.triangle, 0., 10.), 0.)
        np.testing.assert_array_equal(triangle_pressure_weights(self.triangle, 2., 0.), 0.)
        area = .5
        pressure = np.array([2., 2., 1.])
        expected = area / 12 * (pressure + pressure.sum())
        np.testing.assert_allclose(triangle_pressure_weights(self.triangle, 2., 1.), expected, atol=1e-15)
        one_submerged = self.triangle.copy()
        one_submerged[:, 2] = [0., 1., 1.]
        # Original area is still .5, with a triangular wet part of area .125.
        np.testing.assert_allclose(triangle_pressure_weights(one_submerged, .5, 1.),
                                   [1/64, 1/384, 1/384], atol=1e-15)

    def test_orientation_translation_scaling_and_subdivision_invariance(self):
        weights = triangle_pressure_weights(self.triangle, .5, 10000.)
        for order in ((2, 0, 1), (2, 1, 0)):
            actual = triangle_pressure_weights(self.triangle[list(order)], .5, 10000.)
            np.testing.assert_allclose(actual, weights[list(order)], atol=1e-12)
        shifted = self.triangle + [7., -4., 3.]
        np.testing.assert_allclose(triangle_pressure_weights(shifted, 3.5, 10000.), weights, atol=1e-12)
        np.testing.assert_allclose(triangle_pressure_weights(self.triangle * 2, 1., 10000.), weights * 8, atol=1e-11)
        barycentric = np.array(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (.5, .5, 0.),
                                (0., .5, .5), (.5, 0., .5)))
        recovered = np.zeros(3)
        for ids in ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)):
            basis = barycentric[list(ids)]
            recovered += basis.T @ triangle_pressure_weights(basis @ self.triangle, .5, 10000.)
        np.testing.assert_allclose(recovered, weights, atol=1e-12)

    def test_invalid_geometry_or_parameters_raise(self):
        for points, head, gradient in ((np.zeros((3, 3)), .5, 1.),
                                       (self.triangle, float("nan"), 1.),
                                       (self.triangle, .5, -1.),
                                       (self.triangle, True, 1.),
                                       (np.ones((2, 3)), .5, 1.)):
            with self.subTest(head=head, gradient=gradient), self.assertRaises(ValueError):
                triangle_pressure_weights(points, head, gradient)


class NativeHydrostaticTests(unittest.TestCase):
    def test_native_force_moment_and_affine_virtual_work(self):
        from dolfinx import fem, mesh
        from mpi4py import MPI
        from hydrostatic_load import assemble_water_forces

        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 1.))
        tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 2, dtype=np.int32))
        imported = SimpleNamespace(domain=domain, facet_tags=tags)
        forces = assemble_water_forces(imported, space, 10000., {2: .5}, {}).reshape(-1, 3)
        coordinates = space.tabulate_dof_coordinates()
        # Unit rectangle, triangular pressure distribution over 0<z<.5.
        np.testing.assert_allclose(forces.sum(axis=0), [-1250., 0., 0.], atol=1e-10)
        np.testing.assert_allclose(np.cross(coordinates, forces).sum(axis=0),
                                   [0., -1250. / 6, 625.], atol=1e-10)
        displacement = np.column_stack((2 + 3 * coordinates[:, 1] + 4 * coordinates[:, 2],
                                        np.zeros(len(coordinates)), np.zeros(len(coordinates))))
        work = np.einsum("ni,ni->", forces, displacement)
        self.assertAlmostEqual(work, -1250. * (2 + 3 * .5 + 4 / 6), places=10)

    def test_constant_pressure_and_independent_opposing_faces(self):
        from dolfinx import fem, mesh
        from mpi4py import MPI
        from hydrostatic_load import assemble_water_forces

        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        first = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 1.))
        second = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 0.))
        top = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 1.))
        ids = np.r_[first, second, top]
        order = np.argsort(ids)
        values = np.r_[np.full(len(first), 2), np.full(len(second), 3), np.full(len(top), 4)].astype(np.int32)
        tags = mesh.meshtags(domain, 2, ids[order], values[order])
        imported = SimpleNamespace(domain=domain, facet_tags=tags)
        forces = assemble_water_forces(imported, space, 10000., {2: .4, 3: .7}, {4: 3000.})
        np.testing.assert_allclose(forces.reshape(-1, 3).sum(axis=0),
                                   [10000. * (.7**2 - .4**2) / 2, 0., -3000.], atol=1e-10)
        with self.assertRaisesRegex(ValueError, "both hydrostatic and constant"):
            assemble_water_forces(imported, space, 10000., {2: .4}, {2: 3000.})
        for gradient, heads, pressures in ((float("nan"), {}, {}), (1., {999: float("nan")}, {}),
                                           (1., {}, {999: -1.}), (True, {}, {})):
            with self.assertRaises(ValueError):
                assemble_water_forces(imported, space, gradient, heads, pressures)


if __name__ == "__main__":
    unittest.main()
