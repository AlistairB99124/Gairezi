"""Pressure/aperture/steady-surface-flow feedback, not bulk dam mechanics."""
import unittest

import numpy as np

if __package__:
    from .fracture_faces import PairedCrackFaces
    from .fracture_hydrostatic import HydrostaticBoundary
    from .fracture_steady_flow import SteadyCouplingOptions, compliant_steady_crack
    from . import test_fracture_steady_flow as surface_tests
else:
    from fracture_faces import PairedCrackFaces
    from fracture_hydrostatic import HydrostaticBoundary
    from fracture_steady_flow import SteadyCouplingOptions, compliant_steady_crack
    import test_fracture_steady_flow as surface_tests


class CompliantSteadyTests(unittest.TestCase):
    def options(self, maximum_iterations=150):
        return SteadyCouplingOptions(1e-10, 1e-14, maximum_iterations, .5, 1e-10)

    def solve(self, fixture, **overrides):
        faces, _, parents, boundaries = fixture
        args = dict(
            reference_aperture_m=.001, normal_compliance_m_pa=1e-8, reference_pressure_pa=0.,
            water_density_kg_m3=1000., gravity_m_s2=9.81,
            saturated_face_mask=np.ones(len(faces.minus), dtype=bool),
            closure_tolerance_m=1e-12, options=self.options(),
        )
        args.update(overrides)
        return compliant_steady_crack(faces, parents, boundaries, **args)

    def horizontal_fixture(self, n):
        faces, state, parents, boundaries = surface_tests.SteadySurfaceTests().fixture(n)
        coordinates = faces.coordinates[:, [2, 1, 0]]
        horizontal = PairedCrackFaces(coordinates, faces.minus, faces.plus,
                                      np.tile([0., 0., 1.], (len(faces.minus), 1)))
        return horizontal, state, parents, boundaries

    def test_equal_heads_match_local_compliance_and_independent_flow(self):
        fixture = surface_tests.SteadySurfaceTests().fixture(4)
        faces, _, _, boundaries = fixture
        boundaries[1] = HydrostaticBoundary("tailwater", 4., boundaries[1].inlet_source_edges)
        result = self.solve(fixture)
        height = faces.coordinates[faces.minus, 2]
        pressure = 9810. * (4 - height)
        np.testing.assert_allclose(result.hydraulic_load.vertex_pressures_pa, pressure, atol=1e-9)
        np.testing.assert_allclose(result.vertex_apertures_m, .001 + 1e-8 * pressure, atol=2e-13)
        measured = faces.measure(result.displacement_m)["signed_vertex_gaps_m"]
        np.testing.assert_allclose(measured, result.vertex_apertures_m, atol=1e-15)
        self.assertFalse(result.report["coupled_to_dam_mechanics"])
        self.assertFalse(result.report["solver_ready"])

    def test_two_way_feedback_converges_to_integrated_nonlinear_solution(self):
        errors = []
        for n in (4, 16):
            fixture = self.horizontal_fixture(n)
            result = self.solve(fixture)
            x = fixture[0].coordinates[fixture[0].minus, 0]
            w0, coefficient = .001, 1e-8 * 9810.
            inlet, outlet = w0 + coefficient * 4., w0 + coefficient * 2.
            exact = ((inlet**4 * (1 - x) + outlet**4 * x)**.25 - w0) / coefficient
            errors.append(np.max(np.abs(result.hydraulic_load.vertex_heads_m - exact)))
            np.testing.assert_allclose(
                result.vertex_apertures_m,
                w0 + 1e-8 * result.hydraulic_load.vertex_pressures_pa, atol=2e-13,
            )
            # Local compliance changes conductivity enough that the accepted
            # head is not the prescribed-aperture linear head.
            self.assertGreater(np.max(np.abs(result.hydraulic_load.vertex_heads_m - (4 - 2 * x))), .1)
            last = result.report["iterations"][-1]
            self.assertLessEqual(last["maximum_aperture_residual_m"], last["aperture_residual_limit_m"])
            self.assertLess(result.hydraulic_load.report["relative_boundary_balance"], 1e-10)
        self.assertLess(errors[1], errors[0] / 8)
        self.assertLess(errors[1], .001)

    def test_zero_compliance_is_prescribed_aperture_limit(self):
        fixture = self.horizontal_fixture(4)
        result = self.solve(fixture, normal_compliance_m_pa=0.)
        x = fixture[0].coordinates[fixture[0].minus, 0]
        np.testing.assert_allclose(result.hydraulic_load.vertex_heads_m, 4 - 2 * x, atol=1e-13)
        np.testing.assert_allclose(result.vertex_apertures_m, .001, atol=1e-15)
        self.assertEqual(len(result.report["iterations"]), 1)

    def test_reference_pressure_and_owner_reversal_preserve_response(self):
        fixture = self.horizontal_fixture(4)
        expected = self.solve(fixture)
        faces, state, parents, boundaries = fixture
        minus, plus, normals = faces.minus.copy(), faces.plus.copy(), faces.normals.copy()
        minus[::2], plus[::2] = plus[::2].copy(), minus[::2].copy()
        normals[::2] *= -1
        reversed_faces = PairedCrackFaces(faces.coordinates, minus, plus, normals)
        result = self.solve((reversed_faces, state, parents, boundaries))
        np.testing.assert_allclose(result.vertex_apertures_m, expected.vertex_apertures_m, atol=1e-14)
        np.testing.assert_allclose(result.displacement_m, expected.displacement_m, atol=1e-14)
        shifted_reference = self.solve(fixture, reference_pressure_pa=1e4, reference_aperture_m=.0011)
        np.testing.assert_allclose(shifted_reference.vertex_apertures_m, expected.vertex_apertures_m, atol=3e-13)

    def test_nonconvergence_closure_and_invalid_options_are_explicit_failures(self):
        fixture = self.horizontal_fixture(4)
        with self.assertRaisesRegex(RuntimeError, "no coupled equilibrium accepted"):
            self.solve(fixture, options=self.options(maximum_iterations=1))
        with self.assertRaisesRegex(ValueError, "would close"):
            self.solve(fixture, reference_pressure_pa=1e6)
        for name, value in (("normal_compliance_m_pa", -1.), ("reference_aperture_m", 0.),
                             ("reference_pressure_pa", True), ("options", None)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.solve(fixture, **{name: value})
        for options in ((1e-8, 1e-12, 0, .5, 1e-10), (1e-8, 1e-12, 10, 2., 1e-10),
                        (True, 1e-12, 10, .5, 1e-10)):
            with self.assertRaises(ValueError):
                SteadyCouplingOptions(*options)


if __name__ == "__main__":
    unittest.main()
