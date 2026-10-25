"""Finite-seed sensitivity classification must not imply convergence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Aster.finite_seed_comparison import attempt, compare
from Aster.run_case import file_hash


class FiniteSeedComparisonTests(unittest.TestCase):
    def pair(self, changes):
        inputs = {
            "shared_seed": {"sha256": "same"}, "material": {"E": 35e9},
            "centre_m": [1., 2., 3.], "times": [0., .5, 1., 1.5, 2.], "line_search": True,
        }
        row = {
            "normal_divisions": 32, "normal_grading_power": 2,
            "lateral_divisions": 8, "lateral_width_m": 1., "tetrahedral": False,
            "equilibrium_integral_m3": .02,
        }
        second = {**row, **changes}
        return [(inputs, row), (copy.deepcopy(inputs), second)]

    def test_discretisation_classification_and_no_convergence_claim(self):
        cases = (
            ({}, "repeat"),
            ({"normal_divisions": 64}, "normal_resolution"),
            ({"lateral_divisions": 16}, "lateral_resolution"),
            ({"lateral_divisions": 16, "lateral_width_m": 2.}, "extent_at_fixed_lateral_spacing"),
            ({"tetrahedral": True}, "element_family"),
            ({"normal_divisions": 64, "tetrahedral": True}, "multiple_discretisation_changes"),
            ({"lateral_divisions": 32, "lateral_width_m": 2.}, "multiple_discretisation_changes"),
        )
        for changes, kind in cases:
            with self.subTest(kind=kind), patch(
                "Aster.finite_seed_comparison.attempt", side_effect=self.pair(changes),
            ):
                result = compare(Path("first"), Path("second"))
                self.assertEqual(result["comparison_kind"], kind)
                self.assertFalse(result["mesh_convergence_established"])
                self.assertFalse(result["engineering_validated"])

    def test_physical_and_history_changes_rejected(self):
        for key in ("shared_seed", "material", "centre_m", "times", "line_search"):
            pair = self.pair({})
            pair[1][0][key] = {} if key == "material" else None
            with self.subTest(key=key), patch(
                "Aster.finite_seed_comparison.attempt", side_effect=pair,
            ), self.assertRaisesRegex(ValueError, key):
                compare(Path("first"), Path("second"))

    def test_platform_provenance_paths_are_reported_not_physical_parameters(self):
        pair = self.pair({})
        for index, path in enumerate(("C:\\model\\defaults.json", "/mnt/c/model/defaults.json")):
            pair[index][0]["material"] = {
                "parameters": {"E": 35e9},
                "provenance": {"fracture_energy": path, "width_basis": "same physical assumption"},
            }
        with patch("Aster.finite_seed_comparison.attempt", side_effect=pair):
            result = compare(Path("first"), Path("second"))
        self.assertNotEqual(*result["material_provenance_paths_by_attempt"])
        pair[1][0]["material"]["parameters"]["E"] = 30e9
        with patch("Aster.finite_seed_comparison.attempt", side_effect=pair), self.assertRaisesRegex(
            ValueError, "material",
        ):
            compare(Path("first"), Path("second"))

    def test_relative_change_uses_equilibrium_not_final_state(self):
        with patch("Aster.finite_seed_comparison.attempt", side_effect=self.pair(
            {"normal_divisions": 64, "equilibrium_integral_m3": .023},
        )):
            result = compare(Path("first"), Path("second"))
        self.assertAlmostEqual(result["relative_equilibrium_integral_change"], .15)

    def test_attempt_rejects_changed_inputs_and_persisted_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = {"finite_sector": True, "specimen_divisions": 32, "centre_m": [0., 0., 0.],
                      "lateral_divisions": 8, "shared_seed": {}, "times": []}
            input_path = directory / "inputs.json"
            input_path.write_text(json.dumps(inputs))
            evidence = {"status": "seed_placement_and_irreversibility_passed",
                        "inputs_sha256": file_hash(input_path), "checks": {}, "specimen_metrics": {}}
            report = directory / "benchmark_report.json"
            report.write_text(json.dumps(evidence))
            input_path.write_text(json.dumps({**inputs, "changed": True}))
            with self.assertRaisesRegex(ValueError, "unchanged"):
                attempt(directory)
            input_path.write_text(json.dumps(inputs))
            (directory / "native_seed.json").write_text("{}")
            with patch("Aster.finite_seed_comparison.history_checks", return_value={}), patch(
                "Aster.finite_seed_comparison.specimen_metrics", return_value={"changed": True},
            ), self.assertRaisesRegex(ValueError, "metrics differ"):
                attempt(directory)
