"""Regression tests for the shared dam configuration contract."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from Elmer.model_config import (
    CONFIG_PATH,
    ConfigurationValidationError,
    load_config,
    validate_config,
)


class ModelConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def assert_invalid_at(self, payload: object, path: str) -> None:
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config(payload)
        self.assertIn(path, {issue.path for issue in context.exception.issues})

    def test_default_configuration_is_valid_from_file_and_object(self):
        from_file = load_config(CONFIG_PATH)
        from_object = load_config(copy.deepcopy(self.payload))
        self.assertEqual(from_file, from_object)

    def test_validation_does_not_rewrite_configuration_values(self):
        original = copy.deepcopy(self.payload)
        validated = validate_config(self.payload)
        self.assertIs(validated, self.payload)
        self.assertEqual(self.payload, original)

    def test_file_loader_accepts_explicit_configuration_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(self.payload), encoding="utf-8")
            self.assertEqual(load_config(path), self.payload)

    def test_rejects_missing_and_unknown_fields_with_paths(self):
        del self.payload["loads"]["overflow_head"]
        self.payload["loads"]["surprise"] = 1
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config(self.payload)
        paths = {issue.path for issue in context.exception.issues}
        self.assertIn("loads.overflow_head", paths)
        self.assertIn("loads.surprise", paths)

    def test_rejects_boolean_as_a_number(self):
        self.payload["geometry"]["global_element_size_m"] = True
        self.assert_invalid_at(self.payload, "geometry.global_element_size_m")

    def test_rejects_nonfinite_numbers(self):
        self.payload["loads"]["gravity"] = float("nan")
        self.assert_invalid_at(self.payload, "loads.gravity")

    def test_checks_radius_width_midpoint_and_plinth_coverage(self):
        self.payload["geometry"]["wall"]["radial_width_m"] = 3.0
        self.payload["geometry"]["plinth"]["upstream_radius_m"] = 79.0
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config(self.payload)
        paths = {issue.path for issue in context.exception.issues}
        self.assertIn("geometry.wall.radial_width_m", paths)
        self.assertIn("geometry.plinth.upstream_radius_m", paths)

    def test_contours_must_have_matching_increasing_stations_and_bedrock_below(self):
        self.payload["geometry"]["bedrock"]["contours"][1]["chainage"] = 38
        self.payload["geometry"]["plinth"]["contours"][2]["chainage"] = 37
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config(self.payload)
        paths = {issue.path for issue in context.exception.issues}
        self.assertIn("geometry.plinth.contours", paths)
        self.assertIn("geometry.bedrock.contours", paths)

        self.payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.payload["geometry"]["bedrock"]["contours"][0]["height"] = 1
        self.assert_invalid_at(self.payload, "geometry.bedrock.contours[0].height")

    def test_support_requires_expected_material_model_and_xyz_springs(self):
        del self.payload["geometry"]["plinth"]["support"]["spring_n_per_m3"]["y"]
        self.payload["geometry"]["bedrock"]["support"]["model"] = "unsupported"
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config(self.payload)
        paths = {issue.path for issue in context.exception.issues}
        self.assertIn("geometry.plinth.support.spring_n_per_m3.y", paths)
        self.assertIn("geometry.bedrock.support.model", paths)

    def test_output_flags_must_be_booleans(self):
        self.payload["outputs"]["max_displacement"] = 1
        self.assert_invalid_at(self.payload, "outputs.max_displacement")

    def test_phase_field_controls_are_explicit_and_validated(self):
        crack = self.payload["crack_simulations"][0]
        crack["phase_field_parameters"]["model"] = "AT1"
        crack["phase_field_parameters"]["residual_stiffness"] = 1.0
        crack["loading"]["load_steps"] = 0
        crack["solver"]["field_output_interval"] = 0
        crack["solver"]["staggered_tolerance"] = 1.0
        crack["solver"]["maximum_refinement_levels"] = -1
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config(self.payload)
        paths = {issue.path for issue in context.exception.issues}
        prefix = "crack_simulations[0]"
        self.assertIn(f"{prefix}.phase_field_parameters.model", paths)
        self.assertIn(f"{prefix}.phase_field_parameters.residual_stiffness", paths)
        self.assertIn(f"{prefix}.loading.load_steps", paths)
        self.assertIn(f"{prefix}.solver.field_output_interval", paths)
        self.assertIn(f"{prefix}.solver.staggered_tolerance", paths)
        self.assertIn(f"{prefix}.solver.maximum_refinement_levels", paths)

    def test_crack_solver_rejects_unknown_fields(self):
        self.payload["crack_simulations"][0]["solver"]["silent_failure"] = True
        self.assert_invalid_at(self.payload, "crack_simulations[0].solver.silent_failure")

    def test_water_reference_level_must_match_its_components(self):
        self.payload["loads"]["overflow_water_level"] = 35
        self.assert_invalid_at(self.payload, "loads.overflow_water_level")

    def test_rejects_non_object_configuration(self):
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config([])
        self.assertEqual(context.exception.issues[0].path, "")


if __name__ == "__main__":
    unittest.main()
