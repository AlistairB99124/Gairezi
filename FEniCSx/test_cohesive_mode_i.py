"""Analytical interface-law checks; not verification of coupled crack growth."""
import unittest

import numpy as np

from FEniCSx.cohesive_mode_i import LinearModeICohesiveLaw, ModeIHistory


class CohesiveModeITests(unittest.TestCase):
    def setUp(self):
        self.law = LinearModeICohesiveLaw(2e6, 400.)

    def test_critical_opening_and_activation_without_initial_gap(self):
        self.assertAlmostEqual(self.law.critical_opening_m, .0004)
        result = self.law.response(0., ModeIHistory())
        self.assertEqual(result.tensile_traction_pa, 2e6)
        self.assertEqual(result.stored_energy_j_m2, 0.)
        self.assertEqual(result.dissipated_energy_j_m2, 0.)
        self.assertLess(result.tangent_pa_per_m, 0.)

    def test_monotonic_work_equals_stored_plus_dissipated_energy(self):
        critical = self.law.critical_opening_m
        openings = np.linspace(0., critical, 101)
        history = ModeIHistory()
        tractions = []
        work = 0.
        for index, opening in enumerate(openings):
            result = self.law.response(opening, history)
            tractions.append(result.tensile_traction_pa)
            if index:
                work += .5 * (tractions[-2] + tractions[-1]) * (opening - openings[index - 1])
            self.assertAlmostEqual(
                work, result.stored_energy_j_m2 + result.dissipated_energy_j_m2, places=9)
            self.assertGreaterEqual(
                result.trial_history.maximum_opening_m, history.maximum_opening_m)
            history = result.trial_history
        self.assertAlmostEqual(work, self.law.fracture_energy_j_m2, places=9)
        self.assertEqual(result.stored_energy_j_m2, 0.)

    def test_unloading_reloading_does_not_heal_or_dissipate_more(self):
        maximum = .4 * self.law.critical_opening_m
        peak = self.law.response(maximum, ModeIHistory())
        history = peak.trial_history
        for fraction in (1., .8, .2, 0., .3, .9, 1.):
            with self.subTest(fraction=fraction):
                result = self.law.response(maximum * fraction, history)
                self.assertEqual(result.trial_history, history)
                self.assertAlmostEqual(
                    result.tensile_traction_pa, peak.tensile_traction_pa * fraction)
                self.assertEqual(result.dissipated_energy_j_m2, peak.dissipated_energy_j_m2)
        extended = self.law.response(.6 * self.law.critical_opening_m, history)
        self.assertGreater(extended.dissipated_energy_j_m2, peak.dissipated_energy_j_m2)

    def test_fully_separated_interface_never_recovers_tensile_strength(self):
        broken = self.law.response(2 * self.law.critical_opening_m, ModeIHistory())
        for opening in (0., .1 * self.law.critical_opening_m, 3 * self.law.critical_opening_m):
            result = self.law.response(opening, broken.trial_history)
            self.assertEqual(result.tensile_traction_pa, 0.)
            self.assertEqual(result.tangent_pa_per_m, 0.)
            self.assertEqual(result.stored_energy_j_m2, 0.)
            self.assertEqual(result.dissipated_energy_j_m2, 400.)
            self.assertGreaterEqual(
                result.trial_history.maximum_opening_m, broken.trial_history.maximum_opening_m)

    def test_trial_rejection_leaves_committed_history_unchanged(self):
        committed = ModeIHistory(.1 * self.law.critical_opening_m)
        rejected = self.law.response(.8 * self.law.critical_opening_m, committed)
        retried = self.law.response(.2 * self.law.critical_opening_m, committed)
        self.assertAlmostEqual(committed.maximum_opening_m, .1 * self.law.critical_opening_m)
        self.assertLess(retried.trial_history.maximum_opening_m, rejected.trial_history.maximum_opening_m)
        self.assertAlmostEqual(retried.dissipated_energy_j_m2, .2 * 400.)

    def test_tangent_and_stored_energy_gradient_on_smooth_branches(self):
        critical = self.law.critical_opening_m
        step = critical * 1e-6
        for previous, opening in ((.1 * critical, .3 * critical), (.7 * critical, .4 * critical)):
            history = ModeIHistory(previous)
            minus = self.law.response(opening - step, history)
            plus = self.law.response(opening + step, history)
            result = self.law.response(opening, history)
            self.assertAlmostEqual(
                (plus.tensile_traction_pa - minus.tensile_traction_pa) /
                (2 * step) / result.tangent_pa_per_m, 1., places=9)
            work_derivative = (
                plus.stored_energy_j_m2 + plus.dissipated_energy_j_m2
                - minus.stored_energy_j_m2 - minus.dissipated_energy_j_m2
            ) / (2 * step)
            self.assertAlmostEqual(work_derivative / result.tensile_traction_pa, 1., places=9)

    def test_units_scaling_changes_opening_but_not_fracture_work(self):
        stronger = LinearModeICohesiveLaw(4e6, 400.)
        self.assertEqual(stronger.critical_opening_m, self.law.critical_opening_m / 2)
        for law in (self.law, stronger):
            result = law.response(.5 * law.critical_opening_m, ModeIHistory())
            self.assertEqual(result.dissipated_energy_j_m2, 200.)
            self.assertEqual(result.tensile_traction_pa, .5 * law.tensile_strength_pa)

    def test_closed_unloading_cycle_has_zero_additional_work(self):
        maximum = .7 * self.law.critical_opening_m
        peak = self.law.response(maximum, ModeIHistory())
        openings = np.concatenate((np.linspace(maximum, 0., 101),
                                   np.linspace(0., maximum, 101)[1:]))
        work = 0.
        previous = peak
        previous_opening = maximum
        for opening in openings:
            result = self.law.response(opening, peak.trial_history)
            work += .5 * (result.tensile_traction_pa + previous.tensile_traction_pa) * (
                opening - previous_opening)
            previous, previous_opening = result, opening
        self.assertAlmostEqual(work, 0., places=10)
        self.assertEqual(result.stored_energy_j_m2, peak.stored_energy_j_m2)
        self.assertEqual(result.dissipated_energy_j_m2, peak.dissipated_energy_j_m2)

    def test_numpy_scalar_inputs_are_normalized_without_reduced_precision_arithmetic(self):
        law = LinearModeICohesiveLaw(np.float32(2e6), np.float32(400.))
        history = ModeIHistory(np.float32(.001))
        self.assertIs(type(law.tensile_strength_pa), float)
        self.assertIs(type(law.fracture_energy_j_m2), float)
        self.assertIs(type(history.maximum_opening_m), float)
        self.assertEqual(law.critical_opening_m, self.law.critical_opening_m)

    def test_invalid_parameters_and_compression_are_explicit(self):
        for value in (True, -1., 0., float("nan"), float("inf"), "2700", 1j):
            for strength, energy in ((value, 2700.), (2e6, value)):
                with self.subTest(strength=strength, energy=energy), self.assertRaises(ValueError):
                    LinearModeICohesiveLaw(strength, energy)
        for value in (True, -1., float("nan"), float("inf"), "0", 1j):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.law.response(value, ModeIHistory())
            with self.subTest(value=value), self.assertRaises(ValueError):
                ModeIHistory(value)
        with self.assertRaisesRegex(ValueError, "contact"):
            self.law.response(-1e-6, ModeIHistory())
        with self.assertRaisesRegex(ValueError, "ModeIHistory"):
            self.law.response(0., None)
        for strength, energy in ((1e-308, 1e308), (1e308, 1e-308)):
            with self.subTest(strength=strength, energy=energy), self.assertRaises(ValueError):
                LinearModeICohesiveLaw(strength, energy)


if __name__ == "__main__":
    unittest.main()
