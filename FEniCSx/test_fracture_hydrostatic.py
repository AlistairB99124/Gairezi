"""Equilibrium hydraulic head assignment, exclusions and crack-face forces."""
import unittest

import numpy as np

if __package__:
    from .fracture_hydrostatic import HydrostaticBoundary, drained_crack_load
    from . import test_fracture_water_access as access_tests
else:
    from fracture_hydrostatic import HydrostaticBoundary, drained_crack_load
    import test_fracture_water_access as access_tests


class HydrostaticCrackTests(unittest.TestCase):
    def setUp(self):
        self.faces, self.state, self.parents = access_tests.FractureWaterAccessTests().fixture()
        self.boundaries = [HydrostaticBoundary("reservoir", 3., [[0, 1]])]

    def load(self, **overrides):
        arguments = dict(
            water_density_kg_m3=1000., gravity_m_s2=9.81,
            saturated_face_mask=np.ones(4, dtype=bool),
            previously_wet_face_mask=np.zeros(4, dtype=bool),
            closure_tolerance_m=1e-10,
        )
        arguments.update(overrides)
        return drained_crack_load(self.faces, self.state, self.parents, self.boundaries, **arguments)

    def test_connected_hydrostatic_gradient_force_moment_and_work(self):
        result = self.load()
        height = self.faces.coordinates[self.faces.minus, 2]
        np.testing.assert_allclose(result.vertex_pressures_pa, 9810. * (3. - height))
        forces = result.nodal_forces_n
        np.testing.assert_allclose(forces.sum(axis=0), 0., atol=1e-10)
        np.testing.assert_allclose(np.cross(self.faces.coordinates, forces).sum(axis=0), 0., atol=1e-10)
        # Rectangle y=0..1,z=0..2: integral rho*g*(3-z) dA = 4*rho*g.
        np.testing.assert_allclose(forces[6:].sum(axis=0), [4 * 9810., 0., 0.], atol=1e-10)
        self.assertAlmostEqual(np.einsum("ni,ni->", forces, self.state), .001 * 4 * 9810., places=10)
        self.assertFalse(result.report["mechanical_equilibrium_solved"])
        self.assertFalse(result.report["solver_ready"])

    def test_equal_heads_from_two_boundaries_do_not_double_pressure(self):
        expected = self.load()
        self.boundaries.append(HydrostaticBoundary("tailwater", 3., [[4, 5]]))
        actual = self.load()
        np.testing.assert_array_equal(actual.nodal_forces_n, expected.nodal_forces_n)

    def test_connected_different_heads_require_flow(self):
        self.boundaries.append(HydrostaticBoundary("tailwater", 4., [[4, 5]]))
        with self.assertRaisesRegex(ValueError, "steady-flow"):
            self.load()

    def test_disconnected_components_can_have_different_heads_and_dry_faces(self):
        self.state[[8, 9], 0] = 0.
        self.boundaries.append(HydrostaticBoundary("tailwater", 4., [[4, 5]]))
        result = self.load()
        heights = self.faces.coordinates[self.faces.minus, 2]
        np.testing.assert_allclose(result.vertex_pressures_pa[:2], 9810. * (3. - heights[:2]))
        np.testing.assert_allclose(result.vertex_pressures_pa[2:], 9810. * (4. - heights[2:]))
        self.boundaries.pop()
        result = self.load(saturated_face_mask=np.array([True, True, False, False]))
        np.testing.assert_array_equal(result.vertex_pressures_pa[2:], 0.)
        self.assertEqual(result.report["explicitly_dry_face_ids"], [2, 3])

    def test_wet_reclosed_or_isolated_faces_are_not_silently_drained(self):
        self.state[[8, 9], 0] = 0.
        with self.assertRaisesRegex(ValueError, "trapped-fluid"):
            self.load()
        with self.assertRaisesRegex(ValueError, "trapped-fluid"):
            self.load(saturated_face_mask=np.array([True, True, False, False]),
                      previously_wet_face_mask=np.array([False, False, True, False]))
        self.state[:] = 0.
        with self.assertRaisesRegex(ValueError, "trapped-fluid"):
            self.load()

    def test_saturation_and_submergence_cannot_be_inferred(self):
        with self.assertRaisesRegex(ValueError, "explicit saturation"):
            self.load(saturated_face_mask=np.zeros(4, dtype=bool))
        self.boundaries = [HydrostaticBoundary("reservoir", 1.5, [[0, 1]])]
        with self.assertRaisesRegex(ValueError, "fully submerged"):
            self.load()

    def test_invalid_inputs_and_boundary_snapshot(self):
        for field, value in (("water_density_kg_m3", 0.), ("gravity_m_s2", -9.81),
                             ("gravity_m_s2", True), ("saturated_face_mask", [1] * 4),
                             ("previously_wet_face_mask", [False]), ("closure_tolerance_m", -1.)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(**{field: value})
        edges = np.array([[0, 1]])
        boundary = HydrostaticBoundary("reservoir", 3., edges)
        edges[:] = 99
        np.testing.assert_array_equal(boundary.inlet_source_edges, [[0, 1]])
        for name, head, pairs in (("", 3., [[0, 1]]), ("a", True, [[0, 1]]),
                                   ("a", 3., [[0, 1], [1, 0]])):
            with self.assertRaises(ValueError):
                HydrostaticBoundary(name, head, pairs)


if __name__ == "__main__":
    unittest.main()
