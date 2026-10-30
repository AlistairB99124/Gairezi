"""Reduced pressure/opening feedback checks; not a dam or wetting-front solve."""
import unittest

import numpy as np

from fracture_flow import FractureFluid, solve_compliant_fracture, solve_prescribed_fracture


class CompliantFractureFlowTests(unittest.TestCase):
    def options(self):
        return dict(
            fluid=FractureFluid(.001, 1e-8, 1000., 9.81),
            normal_compliance_m_pa=1e-9, reference_pressure_pa=1e5,
            reference_apertures_m=[1e-4] * 2, length_m=2., elevation_change_m=0.,
            divisions=20, times_s=[0., .1], inlet_pressures_pa=[1e5] * 2,
            outlet_pressures_pa=[1e5] * 2,
            initial_pressure=lambda x: np.full(x.shape[1], 1e5),
        )

    def test_uniform_pressure_preserves_reference_opening(self):
        report = solve_compliant_fracture(**self.options())
        self.assertTrue(report["coupled_to_mechanics"])
        self.assertFalse(report["solver_ready"])
        self.assertFalse(report["engineering_validated"])
        for row in report["steps"]:
            np.testing.assert_allclose(row["pressure_pa"], 1e5, atol=1e-8)
            np.testing.assert_allclose(row["calculated_apertures_m"], 1e-4, atol=1e-15)
        np.testing.assert_allclose(report["steps"][-1]["boundary_inflows_m2_s"], 0., atol=1e-16)

    def test_steady_flux_converges_to_integrated_cubic_law(self):
        errors = []
        compliance, reference, width, length = 1e-9, 1e5, 1e-4, 2.
        inlet, outlet = 2e5, 1e5
        left, right = width + compliance * (inlet - reference), width
        expected_flux = (left**4 - right**4) / (48 * .001 * compliance * length)
        for divisions in (32, 256):
            options = self.options()
            options.update(
                divisions=divisions, times_s=[0., 1e8], inlet_pressures_pa=[inlet] * 2,
                initial_pressure=lambda x: inlet + (outlet - inlet) * x[0] / length,
            )
            report = solve_compliant_fracture(**options)
            x = np.asarray(report["positions_m"])
            exact_width = (left**4 + (right**4 - left**4) * x / length)**.25
            exact_pressure = reference + (exact_width - width) / compliance
            row = report["steps"][-1]
            np.testing.assert_allclose(row["pressure_pa"], exact_pressure, atol=.001)
            # Steady P1 nodal pressures are exact for the integrated cubic law;
            # measure convergence inside elements rather than a nodal proxy.
            midpoints = (x[1:] + x[:-1]) / 2
            midpoint_exact = reference + (
                (left**4 + (right**4 - left**4) * midpoints / length)**.25 - width
            ) / compliance
            midpoint_pressure = (np.asarray(row["pressure_pa"])[1:] + row["pressure_pa"][:-1]) / 2
            errors.append(np.max(np.abs(midpoint_pressure - midpoint_exact)) / (inlet - outlet))
            np.testing.assert_allclose(row["boundary_inflows_m2_s"], [expected_flux, -expected_flux], rtol=1e-5)
            np.testing.assert_allclose(
                row["calculated_apertures_m"],
                width + compliance * (np.asarray(row["pressure_pa"]) - reference), atol=1e-15,
            )
            self.assertLess(row["scaled_mass_balance_error"], 1e-10)
        self.assertLess(errors[1], 1e-4)
        self.assertLess(errors[1], errors[0] / 5)

    def test_pressure_opening_storage_slows_transient_diffusion(self):
        # A small perturbation independently checks the linearised coupled PDE.
        errors = []
        width, compliance, length, end = 1e-4, 1e-9, 2., 2.
        diffusivity = width**3 / (.012 * (1e-8 * width + compliance))
        for divisions, steps in ((10, 10), (80, 400)):
            count = steps + 1
            options = self.options()
            options.update(
                divisions=divisions, times_s=np.linspace(0., end, count),
                reference_apertures_m=np.full(count, width),
                inlet_pressures_pa=np.full(count, 1e5), outlet_pressures_pa=np.full(count, 1e5),
                initial_pressure=lambda x: 1e5 + np.sin(np.pi * x[0] / length),
            )
            report = solve_compliant_fracture(**options)
            x = np.asarray(report["positions_m"])
            exact = 1e5 + np.sin(np.pi * x / length) * np.exp(-diffusivity * (np.pi / length)**2 * end)
            errors.append(np.max(np.abs(report["steps"][-1]["pressure_pa"] - exact)))
            for row in report["steps"][1:]:
                self.assertLess(row["scaled_free_residual"], 1e-10)
                self.assertLess(row["scaled_mass_balance_error"], 1e-10)
            self.assertGreater(max(report["steps"][-1]["pressure_pa"]) - 1e5, .5)
        self.assertLess(errors[1], .001)
        self.assertLess(errors[1], errors[0] / 5)

    def test_stiff_wall_limit_matches_prescribed_aperture(self):
        options = self.options()
        options.update(normal_compliance_m_pa=1e-20,
                       inlet_pressures_pa=[2e5] * 2,
                       initial_pressure=lambda x: 2e5 - .5e5 * x[0])
        report = solve_compliant_fracture(**options)
        prescribed = {key: value for key, value in options.items()
                      if key not in ("normal_compliance_m_pa", "reference_pressure_pa", "reference_apertures_m")}
        prescribed["apertures_m"] = options["reference_apertures_m"]
        expected = solve_prescribed_fracture(**prescribed)
        np.testing.assert_allclose(report["steps"][-1]["pressure_pa"], expected["steps"][-1]["pressure_pa"],
                                   rtol=1e-9)
        np.testing.assert_allclose(report["steps"][-1]["boundary_inflows_m2_s"],
                                   expected["steps"][-1]["boundary_inflows_m2_s"], rtol=1e-9)

    def test_invalid_compliance_and_closed_paths_do_not_produce_results(self):
        for change in (
            {"normal_compliance_m_pa": 0.}, {"normal_compliance_m_pa": True},
            {"normal_compliance_m_pa": float("nan")}, {"reference_pressure_pa": -1.},
            {"reference_pressure_pa": 3e5}, {"reference_apertures_m": [0., 0.]},
            {"maximum_iterations": 0}, {"maximum_iterations": True}, {"relative_tolerance": 1.},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                solve_compliant_fracture(**{**self.options(), **change})

    def test_nonconverged_coupled_solve_is_not_reported_as_success(self):
        options = self.options()
        options.update(inlet_pressures_pa=[2e5] * 2, maximum_iterations=1, times_s=[0., 1e8],
                       initial_pressure=lambda x: 2e5 - .5e5 * x[0])
        with self.assertRaisesRegex(RuntimeError, "Fracture pressure solve failed.*iterations 1"):
            solve_compliant_fracture(**options)


if __name__ == "__main__":
    unittest.main()
