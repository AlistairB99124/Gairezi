"""Intact-start intent and opening-limit semantics, independent of native solvers."""
from dataclasses import replace
import json
import math
import unittest

from FEniCSx.horizontal_fracture_contract import (
    DamageSelection, HorizontalFractureRequest, OpeningObservation, assess_opening,
    assess_opening_history,
    EquilibriumOpeningObservation, assess_equilibrium_opening,
)
from FEniCSx.fracture_faces import PairedCrackFaces
from Elmer.model_config import ConfigurationValidationError


class DamageSelectionTests(unittest.TestCase):
    def test_requires_both_an_element_and_an_edge(self):
        for elements, edges, paths in (
            ((), (), {"affected_element_ids", "crack_start_edge_ids"}),
            ((0,), (), {"crack_start_edge_ids"}),
            ((), (0,), {"affected_element_ids"}),
        ):
            with self.subTest(elements=elements, edges=edges):
                with self.assertRaises(ConfigurationValidationError) as caught:
                    DamageSelection(elements, edges)
                self.assertEqual({issue.path for issue in caught.exception.issues}, paths)
                self.assertTrue(all("at least one" in issue.message
                                    for issue in caught.exception.issues))

    def test_one_of_each_and_multiple_ids_roundtrip(self):
        for selection in (DamageSelection((0,), (0,)),
                          DamageSelection((10, 20), (3, 5))):
            payload = json.loads(json.dumps(selection.to_dict()))
            self.assertEqual(DamageSelection.from_dict(payload), selection)

    def test_deselecting_last_edge_blocks_submission_packet(self):
        payload = DamageSelection((10,), (3,)).to_dict()
        payload["crack_start_edge_ids"].clear()
        with self.assertRaisesRegex(ConfigurationValidationError, "at least one crack-start edge"):
            DamageSelection.from_dict(payload)

    def test_missing_fields_or_legacy_point_cannot_satisfy_selection(self):
        for payload in ({}, {"affected_element_ids": [1]},
                        {"crack_start_edge_ids": [2]},
                        {"selected_location_m": {"y": 1, "z": -5}}):
            with self.subTest(payload=payload), self.assertRaises(ConfigurationValidationError):
                DamageSelection.from_dict(payload)

    def test_rejects_malformed_and_duplicate_ids(self):
        for name in ("affected_element_ids", "crack_start_edge_ids"):
            for value in (None, "1", 1, [True], [-1], [1.5], ["1"], [1, 1]):
                payload = DamageSelection((1,), (2,)).to_dict()
                payload[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ConfigurationValidationError):
                        DamageSelection.from_dict(payload)


class HorizontalFractureContractTests(unittest.TestCase):
    def setUp(self):
        self.request = HorizontalFractureRequest(12.0, -15.0, 0.6)

    def observation(self, time, opening, initiated=True):
        return OpeningObservation(time, opening, (1.0, 12.0, -15.0), initiated)

    def test_equilibrium_peak_and_final_opening_have_no_fabricated_time(self):
        observations = (
            EquilibriumOpeningObservation("intact_loaded", 0., (0., 0., 0.), False),
            EquilibriumOpeningObservation("accepted_event_1", .003, (1., 2., 3.), True),
            EquilibriumOpeningObservation("final_equilibrium", .001, (4., 5., 6.), True),
        )
        report = assess_equilibrium_opening(.002, observations, completed=True)
        self.assertTrue(report["completed"])
        self.assertEqual(report["opening_limit_status"], "limit_exceeded")
        self.assertEqual(report["peak_opening_m"], .003)
        self.assertEqual(report["peak_opening_state_id"], "accepted_event_1")
        self.assertEqual(report["final_opening_m"], .001)
        self.assertEqual(report["final_state_id"], "final_equilibrium")
        self.assertEqual(report["peak_opening_location_m"], [1., 2., 3.])
        self.assertNotIn("peak_opening_time_s", report)
        self.assertIn("accepted_event_1", report["warnings"][0]["message"])
        self.assertNotIn("physical time", report["warnings"][0]["message"])
        self.assertIn("not a solver error", report["warnings"][0]["message"])
        self.assertFalse(report["engineering_validated"])
        incomplete = assess_equilibrium_opening(.002, observations, completed=False)
        self.assertEqual(incomplete["opening_limit_status"], "not_assessed")
        self.assertIsNone(incomplete["opening_criterion_satisfied"])
        self.assertEqual(len(incomplete["warnings"]), 1)

    def test_equilibrium_exact_thresholds_and_no_fracture(self):
        for opening, status, warnings in ((.001999999, "below_limit", 0),
                                          (.002, "limit_reached", 1),
                                          (.002000001, "limit_exceeded", 1),
                                          (0., "below_limit", 0)):
            observation = EquilibriumOpeningObservation("final", opening, (0., 0., 0.), opening > 0)
            report = assess_equilibrium_opening(.002, (observation,), completed=True)
            self.assertEqual(report["opening_limit_status"], status)
            self.assertEqual(len(report["warnings"]), warnings)
            self.assertTrue(report["completed"])

    def test_equilibrium_invalid_and_reversed_histories_fail(self):
        initial = EquilibriumOpeningObservation("same", 0., (0., 0., 0.), False)
        opened = EquilibriumOpeningObservation("event", .001, (0., 0., 0.), True)
        for history in ((), (initial, initial), (opened, initial), (self.observation(0, 0),)):
            with self.assertRaises(ValueError):
                assess_equilibrium_opening(.002, history, completed=True)
        for limit in (True, 0., float("nan")):
            with self.assertRaises(ValueError):
                assess_equilibrium_opening(limit, (initial,), completed=True)
        with self.assertRaises(ValueError):
            assess_equilibrium_opening(.002, (initial,), completed=1)
        for args in (("", 0., (0., 0., 0.), False),
                     ("s", -.001, (0., 0., 0.), True),
                     ("s", .001, (0., 0., 0.), False),
                     ("s", 0., (0., float("nan"), 0.), False)):
            with self.assertRaises(ValueError):
                EquilibriumOpeningObservation(*args)

    def test_painted_measured_face_opening_warns_without_horizontal_request(self):
        faces = PairedCrackFaces(
            [[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]] * 2,
            [[0, 1, 2]], [[3, 4, 5]], [[1., 0., 0.]],
        )
        observations = []
        for time, opening in enumerate((0., .001, .002, .003, 0.)):
            measurement = faces.measure([[0., 0., 0.]] * 3 + [[opening, .01, 0.]] * 3)
            observations.append(OpeningObservation(
                float(time), measurement["maximum_opening_m"],
                tuple(measurement["peak_location_m"]), time > 0,
            ))
        report = assess_opening_history(.002, tuple(observations), completed=True)
        self.assertEqual(report["scenario_kind"], "painted_tensile_loss")
        self.assertTrue(report["completed"])
        self.assertEqual(report["opening_limit_status"], "limit_exceeded")
        self.assertEqual(report["peak_opening_m"], .003)
        self.assertEqual(report["peak_opening_time_s"], 3.)
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("not a solver error", report["warnings"][0]["message"])
        incomplete = assess_opening_history(.002, tuple(observations), completed=False)
        self.assertEqual(incomplete["opening_limit_status"], "not_assessed")
        self.assertIsNone(incomplete["opening_criterion_satisfied"])

    def test_generic_opening_limit_requires_explicit_valid_inputs(self):
        for limit in (0., -1., True, float("nan")):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                assess_opening_history(limit, (self.observation(0, .001),), completed=True)
        with self.assertRaisesRegex(ValueError, "scenario_kind"):
            assess_opening_history(.002, (self.observation(0, .001),), completed=True, scenario_kind="")

    def test_roundtrip_preserves_intact_wet_scenario_without_a_seed(self):
        payload = self.request.to_dict()
        self.assertEqual(self.request, HorizontalFractureRequest.from_dict(
            json.loads(json.dumps(payload, allow_nan=False))))
        self.assertEqual(payload["initial_condition"], "intact")
        self.assertEqual(payload["hydraulic_model"], "transient_fracture_flow")
        self.assertFalse(payload["bulk_seepage"])
        self.assertEqual(payload["opening_limit_m"], 0.002)
        for key in ("seed_thickness_m", "opening_m", "depth_m", "length_m"):
            self.assertNotIn(key, payload)

    def test_no_initiation_is_a_reportable_completed_result(self):
        report = assess_opening(self.request, (self.observation(0, 0, False),
                                              self.observation(60, 0, False)), completed=True)
        self.assertFalse(report["fracture_initiated"])
        self.assertEqual(report["opening_limit_status"], "below_limit")
        self.assertTrue(report["opening_criterion_satisfied"])
        self.assertEqual(report["warnings"], [])
        self.assertFalse(report["engineering_validated"])

    def test_reaching_limit_warns_without_failing_completed_solve(self):
        for opening, expected, passed in (
            (0.001999999, "below_limit", True),
            (0.002, "limit_reached", False),
            (0.002000001, "limit_exceeded", False),
        ):
            with self.subTest(opening=opening):
                history = (self.observation(60, opening),)
                report = assess_opening(self.request, history, completed=True)
                self.assertEqual(report["opening_limit_status"], expected)
                self.assertEqual(report["opening_criterion_satisfied"], passed)
                self.assertTrue(report["completed"])
                self.assertEqual(len(report["warnings"]), 0 if passed else 1)
                if not passed:
                    warning = report["warnings"][0]
                    self.assertEqual(warning["path"], "opening_limit_m")
                    self.assertIn("not a solver error", warning["message"])
                changed = replace(self.request, opening_limit_m=0.003)
                changed_report = assess_opening(changed, history, completed=True)
                self.assertTrue(changed_report["opening_criterion_satisfied"])
                self.assertEqual(changed_report["warnings"], [])

    def test_reclosure_does_not_hide_peak_limit_exceedance(self):
        report = assess_opening(self.request, (self.observation(10, .003),
                                              self.observation(20, 0)), completed=True)
        self.assertEqual(report["peak_opening_m"], .003)
        self.assertEqual(report["peak_opening_time_s"], 10)
        self.assertFalse(report["opening_criterion_satisfied"])
        self.assertEqual(len(report["warnings"]), 1)

    def test_partial_run_has_no_success_verdict_even_if_below_limit(self):
        for opening in (0, .003):
            report = assess_opening(self.request, (self.observation(10, opening),),
                                    completed=False)
            self.assertEqual(report["opening_limit_status"], "not_assessed")
            self.assertIsNone(report["opening_criterion_satisfied"])
            self.assertEqual(report["observed_limit_reached"], opening >= .002)
            self.assertFalse(report["completed"])
            self.assertEqual(len(report["warnings"]), int(opening >= .002))

    def test_invalid_or_missing_friction_and_limits_are_rejected(self):
        for field in ("friction_coefficient", "opening_limit_m"):
            for value in (None, True, "0.6", -1, math.nan, math.inf):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.request, **{field: value})
        with self.assertRaises(ValueError):
            replace(self.request, opening_limit_m=0)
        self.assertEqual(replace(self.request, friction_coefficient=0).friction_coefficient, 0)
        payload = self.request.to_dict()
        del payload["friction_coefficient"]
        with self.assertRaisesRegex(ValueError, "missing"):
            HorizontalFractureRequest.from_dict(payload)

    def test_invalid_coordinates_are_rejected_without_overflow(self):
        for value in (True, None, "1", math.nan, math.inf, 10**400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.request, selected_z_m=value)
            with self.subTest(location=value), self.assertRaises(ValueError):
                OpeningObservation(0, 0, (0, 0, value), False)

    def test_legacy_or_changed_physics_cannot_be_silently_consumed(self):
        for key, value in (("seed_thickness_m", .002), ("initial_condition", "damaged"),
                           ("hydraulic_model", "fixed_pressure"), ("bulk_seepage", True),
                           ("schema_version", True)):
            payload = {**self.request.to_dict(), key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                HorizontalFractureRequest.from_dict(payload)

    def test_inconsistent_opening_histories_are_rejected(self):
        with self.assertRaises(ValueError):
            self.observation(1, .001, False)
        with self.assertRaises(ValueError):
            self.observation(-1, 0)
        with self.assertRaises(ValueError):
            self.observation(1, math.nan)
        for history in ((), (self.observation(2, 0), self.observation(1, 0)),
                        (self.observation(1, 0), self.observation(1, 0)),
                        (self.observation(1, 0), self.observation(2, 0, False))):
            with self.subTest(history=history), self.assertRaises(ValueError):
                assess_opening(self.request, history, completed=True)


if __name__ == "__main__":
    unittest.main()
