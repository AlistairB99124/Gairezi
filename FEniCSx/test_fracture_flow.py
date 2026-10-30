"""Poiseuille, diffusion, gravity, aperture storage and failure checks."""
import unittest

import numpy as np

from fracture_flow import FractureFluid, solve_prescribed_fracture


class FractureFlowTests(unittest.TestCase):
    def options(self):
        # Synthetic verification inputs, not production dam material defaults.
        return dict(
            fluid=FractureFluid(.001, 1e-8, 1000., 9.81),
            length_m=2., elevation_change_m=0., divisions=20,
            times_s=[0., .1, .2], apertures_m=[.0001] * 3,
            inlet_pressures_pa=[2e5] * 3, outlet_pressures_pa=[1e5] * 3,
            initial_pressure=lambda x: 2e5 - .5e5 * x[0],
        )

    def test_poiseuille_flux_and_cubic_aperture_scaling(self):
        expected = 1e-12 / (.012) * .5e5
        for factor in (1., 2.):
            options = self.options()
            options["apertures_m"] = [factor * .0001] * 3
            report = solve_prescribed_fracture(**options)
            self.assertFalse(report["engineering_validated"])
            self.assertFalse(report["solver_ready"])
            for step in report["steps"][1:]:
                np.testing.assert_allclose(step["boundary_inflows_m2_s"],
                                           [expected * factor**3, -expected * factor**3], rtol=1e-9)
                self.assertLess(step["scaled_mass_balance_error"], 1e-10)

    def test_hydrostatic_gradient_has_no_flow(self):
        options = self.options()
        options.update(elevation_change_m=1., outlet_pressures_pa=[2e5 - 9810.] * 3,
                       initial_pressure=lambda x: 2e5 - 4905. * x[0])
        report = solve_prescribed_fracture(**options)
        for step in report["steps"][1:]:
            np.testing.assert_allclose(step["boundary_inflows_m2_s"], 0, atol=1e-16)

    def test_transient_diffusion_converges_towards_analytical_decay(self):
        errors = []
        fluid = FractureFluid(.001, 1e-8, 1000., 9.81)
        length, aperture, end = 2., .0001, .02
        diffusivity = aperture**2 / (12 * fluid.viscosity_pa_s * fluid.compressibility_pa_inv)
        for divisions, steps in ((10, 10), (80, 400)):
            report = solve_prescribed_fracture(
                fluid, length_m=length, elevation_change_m=0., divisions=divisions,
                times_s=np.linspace(0, end, steps + 1), apertures_m=np.full(steps + 1, aperture),
                inlet_pressures_pa=np.full(steps + 1, 1e5), outlet_pressures_pa=np.full(steps + 1, 1e5),
                initial_pressure=lambda x: 1e5 + 2e4 * np.sin(np.pi * x[0] / length),
            )
            x = np.asarray(report["positions_m"])
            exact = 1e5 + 2e4 * np.sin(np.pi * x / length) * np.exp(-diffusivity * (np.pi / length)**2 * end)
            errors.append(np.max(np.abs(report["steps"][-1]["pressure_pa"] - exact)) / 2e4)
        self.assertLess(errors[1], .001)
        self.assertLess(errors[1], errors[0] / 5)

    def test_aperture_change_accounts_for_fluid_volume(self):
        options = self.options()
        options.update(apertures_m=[.0001, .0000999, .0000998],
                       inlet_pressures_pa=[1e5] * 3, outlet_pressures_pa=[1e5] * 3,
                       initial_pressure=lambda x: np.full(x.shape[1], 1e5))
        report = solve_prescribed_fracture(**options)
        for row in report["steps"][1:]:
            self.assertLess(row["net_inflow_m2_s"], 0)
            self.assertGreater(max(row["pressure_pa"]), 1e5)
            self.assertLess(row["scaled_mass_balance_error"], 1e-10)

    def test_negative_pressure_closure_and_bad_histories_are_not_success(self):
        for change in (
            {"apertures_m": [0., 0., 0.]}, {"times_s": [0., 1., 1.]},
            {"times_s": [1., 2., 3.]}, {"apertures_m": [1e-4]},
            {"inlet_pressures_pa": [-1., -1., -1.]}, {"divisions": True},
            {"initial_pressure": lambda x: np.zeros(x.shape[1])},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                solve_prescribed_fracture(**{**self.options(), **change})
        options = self.options()
        options.update(apertures_m=[.0001, .0002, .0003],
                       inlet_pressures_pa=[0.] * 3, outlet_pressures_pa=[0.] * 3,
                       initial_pressure=lambda x: np.zeros(x.shape[1]))
        with self.assertRaisesRegex(RuntimeError, "cavitation/wetting"):
            solve_prescribed_fracture(**options)
        for value in (0., -1., float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FractureFluid(value, 1e-8, 1000., 9.81)


if __name__ == "__main__":
    unittest.main()
