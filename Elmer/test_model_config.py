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
from Elmer.regions.bedrock_contours import load_contours, wall_extent


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

    def test_supplied_bedrock_boundary_stiffness_is_validated(self):
        self.assertEqual(self.payload["bedrock_analysis"]["spring_n_per_m3"],
                         {"x": 2e10, "y": 2e10, "z": 53333333000.})
        for bad in (0, -1, True, "20000000000", float("inf")):
            with self.subTest(value=bad):
                changed = copy.deepcopy(self.payload)
                changed["bedrock_analysis"]["spring_n_per_m3"]["x"] = bad
                self.assert_invalid_at(changed, "bedrock_analysis.spring_n_per_m3.x")

    def test_global_element_size_is_a_suggestion_not_a_step_grid_constraint(self):
        self.payload["geometry"]["global_element_size_m"] = 0.75
        self.payload["geometry"]["local_element_size_m"] = 0.25
        validate_config(self.payload)

    def test_wall_tetrahedral_target_is_an_optional_integer_of_at_least_1000(self):
        from Elmer.mesh_sizing import wall_tetrahedral_target

        geometry = self.payload["geometry"]
        self.assertEqual(geometry["target_wall_tetrahedral_elements"], 150000)
        self.assertEqual(wall_tetrahedral_target(geometry), 150000)
        geometry["target_wall_tetrahedral_elements"] = 300000.0
        self.assertIs(validate_config(self.payload), self.payload)
        self.assertEqual(wall_tetrahedral_target(geometry), 300000)
        for value in (999, 1500.5, True, None, "big", float("inf")):
            payload = copy.deepcopy(self.payload)
            payload["geometry"]["target_wall_tetrahedral_elements"] = value
            self.assert_invalid_at(payload, "geometry.target_wall_tetrahedral_elements")
            with self.assertRaises(ValueError):
                wall_tetrahedral_target(payload["geometry"])
        del geometry["target_wall_tetrahedral_elements"]
        self.assertIs(validate_config(self.payload), self.payload)
        self.assertEqual(wall_tetrahedral_target(geometry), 150000)

    def test_concrete_compressive_strength_is_positive_and_optional_for_historical_jobs(self):
        self.assertEqual(self.payload["material"]["concrete"]["compressive_strength_pa"], 30e6)
        for value in (0, -1, True, None, float("nan"), float("inf")):
            payload = copy.deepcopy(self.payload)
            payload["material"]["concrete"]["compressive_strength_pa"] = value
            self.assert_invalid_at(payload, "material.concrete.compressive_strength_pa")
        del self.payload["material"]["concrete"]["compressive_strength_pa"]
        self.assertIs(validate_config(self.payload), self.payload)

    def test_cohesive_fracture_energy_is_positive_and_optional_for_historical_jobs(self):
        self.assertEqual(self.payload["material"]["concrete"]["cohesive_fracture_energy_j_m2"], 100.)
        for value in (0, -1, True, None, float("nan"), float("inf")):
            payload = copy.deepcopy(self.payload)
            payload["material"]["concrete"]["cohesive_fracture_energy_j_m2"] = value
            self.assert_invalid_at(payload, "material.concrete.cohesive_fracture_energy_j_m2")
        del self.payload["material"]["concrete"]["cohesive_fracture_energy_j_m2"]
        self.assertIs(validate_config(self.payload), self.payload)

    def test_bedrock_domain_is_optional_and_validated(self):
        self.payload.pop("bedrock_analysis", None)
        domain = self.payload["geometry"]["bedrock"]["domain"]
        self.assertEqual(domain, {"stream_margin_m": 58., "bank_margin_m": 58., "bottom_z_m": -87.})
        domain["stream_margin_m"] = -1.
        domain["bank_margin_m"] = True
        domain["bottom_z_m"] = -29.
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config(self.payload)
        paths = {issue.path for issue in context.exception.issues}
        self.assertTrue({
            "geometry.bedrock.domain.stream_margin_m",
            "geometry.bedrock.domain.bank_margin_m",
            "geometry.bedrock.domain.bottom_z_m",
        } <= paths)
        del self.payload["geometry"]["bedrock"]["domain"]
        self.assertIs(validate_config(self.payload), self.payload)

    def test_explicit_bedrock_analysis_mode_is_validated_without_changing_cli_default(self):
        self.payload.pop("bedrock_analysis", None)
        from Elmer.model_config import bedrock_analysis_mode

        self.assertEqual(bedrock_analysis_mode(self.payload), "full_volume")
        for mode in ("full_volume", "elastic_layer"):
            candidate = copy.deepcopy(self.payload)
            candidate["bedrock_analysis"] = {"mode": mode}
            self.assertIs(validate_config(candidate), candidate)
            self.assertEqual(bedrock_analysis_mode(candidate), mode)
        for value in (None, {}, {"mode": "rigid_seat"}, {"mode": "elastic_layer", "other": 1}):
            candidate = copy.deepcopy(self.payload)
            candidate["bedrock_analysis"] = value
            with self.subTest(value=value), self.assertRaises(ConfigurationValidationError):
                validate_config(candidate)

    def test_elastic_layer_requires_a_bonded_concrete_foundation_and_bottom(self):
        for mutate in (
            lambda candidate: candidate["geometry"].pop("foundation"),
            lambda candidate: candidate["geometry"]["bedrock"]["support"].update(model="spring"),
            lambda candidate: candidate["geometry"]["bedrock"].pop("domain"),
        ):
            candidate = copy.deepcopy(self.payload)
            mutate(candidate)
            candidate["bedrock_analysis"] = {"mode": "elastic_layer"}
            self.assert_invalid_at(candidate, "bedrock_analysis.mode")
    def test_validation_does_not_rewrite_configuration_values(self):
        original = copy.deepcopy(self.payload)
        validated = validate_config(self.payload)
        self.assertIs(validated, self.payload)
        self.assertEqual(self.payload, original)

    def test_draft_bedrock_section_accepts_placeholder_but_validates_points(self):
        bedrock = self.payload["geometry"]["bedrock"]
        bedrock["bedrock_top_cross_section"] = [{"chainage": 6, "height": 0}]
        self.assertIs(validate_config(self.payload), self.payload)
        bedrock["bedrock_top_cross_section"][0]["height"] = float("nan")
        self.assert_invalid_at(
            self.payload, "geometry.bedrock.bedrock_top_cross_section[0].height",
        )

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

    def test_bedrock_density_accepts_zero_but_concrete_density_must_be_positive(self):
        bedrock = self.payload["material"]["bedrock"]
        for density in (0, 0.0, 2700):
            bedrock["density"] = density
            self.assertIs(validate_config(self.payload), self.payload)
        for density in (-1.0, float("nan"), float("inf"), True):
            bedrock["density"] = density
            self.assert_invalid_at(self.payload, "material.bedrock.density")
        bedrock["density"] = 0
        self.payload["material"]["concrete"]["density"] = 0
        self.assert_invalid_at(self.payload, "material.concrete.density")

    def test_bedrock_outer_support_is_optional_and_validated(self):
        from Elmer.model_config import rock_outer_support

        self.assertEqual(rock_outer_support(self.payload)["model"], "rollers")
        self.assertEqual(rock_outer_support(self.payload)["spring_n_per_m3"],
                         {"x": 1e12, "y": 1e12, "z": 1e12})
        legacy = copy.deepcopy(self.payload)
        del legacy["material"]["bedrock"]["support"]
        self.assertIs(validate_config(legacy), legacy)
        self.assertEqual(rock_outer_support(legacy), {"model": "fixed_bottom", "spring_n_per_m3": None})
        for key, value, path in (
            ("model", "pinned", "material.bedrock.support.model"),
            ("material", "concrete", "material.bedrock.support.material"),
        ):
            payload = copy.deepcopy(self.payload)
            payload["material"]["bedrock"]["support"][key] = value
            self.assert_invalid_at(payload, path)
        for bad in (0, -1.0, float("nan"), True):
            payload = copy.deepcopy(self.payload)
            payload["material"]["bedrock"]["support"]["spring_n_per_m3"]["y"] = bad
            self.assert_invalid_at(payload, "material.bedrock.support.spring_n_per_m3.y")
        payload = copy.deepcopy(self.payload)
        payload["material"]["bedrock"]["support"]["model"] = "fixed_bottom"
        payload["material"]["bedrock"]["support"]["spring_n_per_m3"]["y"] = 0
        self.assertIs(validate_config(payload), payload)

    def test_checks_radius_width_and_midpoint(self):
        self.payload["geometry"]["wall"]["radial_width_m"] = 3.0
        self.assert_invalid_at(self.payload, "geometry.wall.radial_width_m")

    def test_plinth_section_is_no_longer_accepted(self):
        self.payload["geometry"]["plinth"] = {}
        self.assert_invalid_at(self.payload, "geometry.plinth")

    def test_contours_must_increase_and_dip_below_the_crest(self):
        self.payload["geometry"]["bedrock"]["contours"][2]["chainage"] = 6
        self.assert_invalid_at(self.payload, "geometry.bedrock.contours")

        self.payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for row in self.payload["geometry"]["bedrock"]["contours"]:
            row["height"] = max(row["height"], 0)
        self.assert_invalid_at(self.payload, "geometry.bedrock.contours")

    def test_wall_spans_where_bedrock_lies_below_the_crest(self):
        self.assertIs(validate_config(self.payload), self.payload)
        rows = self.payload["geometry"]["bedrock"]["contours"]
        start, end = wall_extent(rows)
        contours = load_contours(CONFIG_PATH)
        self.assertEqual((contours[0].chainage_m, contours[-1].chainage_m), (start, end))
        self.assertEqual(
            [point.chainage_m for point in contours[1:-1]],
            [row["chainage"] for row in rows if start < row["chainage"] < end],
        )
        self.assertEqual(contours[0].bedrock_z_m, 0)
        self.assertEqual(contours[-1].bedrock_z_m, 0)
        self.assertEqual(min(point.bedrock_z_m for point in contours), -29)

    def test_support_requires_expected_material_model_and_xyz_springs(self):
        del self.payload["geometry"]["bedrock"]["support"]["spring_n_per_m3"]["y"]
        self.payload["geometry"]["bedrock"]["support"]["model"] = "unsupported"
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config(self.payload)
        paths = {issue.path for issue in context.exception.issues}
        self.assertIn("geometry.bedrock.support.spring_n_per_m3.y", paths)
        self.assertIn("geometry.bedrock.support.model", paths)

    def test_bedrock_allows_explicit_bond(self):
        self.payload.pop("bedrock_analysis", None)
        self.payload["geometry"].pop("foundation", None)
        self.payload["geometry"]["wall"].pop("step_end_taper_ratio", None)
        self.payload["geometry"]["bedrock"]["support"]["model"] = "bonded"
        validate_config(self.payload)
        self.payload["geometry"]["bedrock"]["support"]["model"] = "distributed_elastic_granite"
        validate_config(self.payload)

    def test_retired_painted_concrete_properties_are_rejected(self):
        for key, value in (("selected_tensile_strength_pa", 1e6), ("selected_youngs_modulus", 15e9)):
            payload = copy.deepcopy(self.payload)
            payload["material"]["concrete"][key] = value
            self.assert_invalid_at(payload, f"material.concrete.{key}")

    def test_max_damage_count_is_optional_null_or_an_integer_of_at_least_two(self):
        self.payload["outputs"]["cracking_risk"] = False
        self.assertIsNone(self.payload["damage_selection"]["max_damage_count"])
        for selection in ({}, {"max_damage_count": None}, {"max_damage_count": 2},
                          {"max_damage_count": 500}, {"max_damage_count": 40.0}):
            self.payload["damage_selection"] = selection
            validate_config(self.payload)
        del self.payload["damage_selection"]
        validate_config(self.payload)
        for value in (0, 1, -3, 2.5, True, "10", float("nan"), float("inf")):
            self.payload["damage_selection"] = {"max_damage_count": value}
            self.assert_invalid_at(self.payload, "damage_selection.max_damage_count")
        for selection in (None, [], {"max_damage_count": 4, "rings": 2}):
            self.payload["damage_selection"] = selection
            self.assertRaises(ConfigurationValidationError, validate_config, self.payload)

    def test_maximum_overload_factor_is_optional_null_or_between_one_and_ten(self):
        self.payload["outputs"]["cracking_risk"] = False
        for factor in (None, 1, 1.0, 2.5, 10):
            self.payload["damage_selection"] = {"max_damage_count": None, "maximum_overload_factor": factor}
            validate_config(self.payload)
        for factor in (0.99, 0, -1, 10.5, True, "2", float("nan"), float("inf")):
            self.payload["damage_selection"] = {"maximum_overload_factor": factor}
            self.assert_invalid_at(self.payload, "damage_selection.maximum_overload_factor")

    def test_crack_base_is_an_optional_boolean_requiring_an_uncapped_count(self):
        self.payload["outputs"]["cracking_risk"] = False
        for selection in ({"crack_base": True}, {"crack_base": False},
                          {"crack_base": True, "max_damage_count": None},
                          {"crack_base": False, "max_damage_count": 12}):
            self.payload["damage_selection"] = selection
            validate_config(self.payload)
        for value in (1, 0, "true", None):
            self.payload["damage_selection"] = {"crack_base": value}
            self.assert_invalid_at(self.payload, "damage_selection.crack_base")
        self.payload["damage_selection"] = {"crack_base": True, "max_damage_count": 12}
        self.assert_invalid_at(self.payload, "damage_selection.crack_base")

    def test_crack_water_pressure_is_an_explicit_optional_boolean(self):
        self.payload["outputs"]["cracking_risk"] = False
        for value in (True, False):
            self.payload["damage_selection"] = {"crack_water_pressure": value}
            validate_config(self.payload)
        for value in (None, 0, 1, "true", {}, []):
            self.payload["damage_selection"] = {"crack_water_pressure": value}
            self.assert_invalid_at(self.payload, "damage_selection.crack_water_pressure")

    def test_support_degradation_is_optional_and_validated(self):
        self.payload["outputs"]["cracking_risk"] = False
        for degradation in (None, {"interface_scale": 0.01}, {"interface_scale": 1},
                            {"abutment": {"scale": 0.1, "fraction": 0.25, "end": "positive"}},
                            {"interface_scale": 0.5,
                             "abutment": {"scale": 0.1, "fraction": 0.5, "end": "negative"}}):
            self.payload["damage_selection"] = {"support_degradation": degradation}
            validate_config(self.payload)
        for scale in (0, -1, 1.5, True, "0.1", float("nan")):
            self.payload["damage_selection"] = {"support_degradation": {"interface_scale": scale}}
            self.assert_invalid_at(self.payload, "damage_selection.support_degradation.interface_scale")
        for abutment, issue in ((
                {"scale": 0, "fraction": 0.25, "end": "positive"}, "scale"), (
                {"scale": 1.5, "fraction": 0.25, "end": "positive"}, "scale"), (
                {"scale": 0.1, "fraction": 0.75, "end": "positive"}, "fraction"), (
                {"scale": 0.1, "fraction": 0.25, "end": "sideways"}, "end")):
            self.payload["damage_selection"] = {"support_degradation": {"abutment": abutment}}
            self.assert_invalid_at(
                self.payload, f"damage_selection.support_degradation.abutment.{issue}")
        self.payload["damage_selection"] = {"support_degradation": {"unknown": 1}}
        self.assertRaises(ConfigurationValidationError, validate_config, self.payload)

    def test_enabled_is_an_optional_boolean(self):
        self.payload["outputs"]["cracking_risk"] = False
        for selection in ({"enabled": True}, {"enabled": False},
                          {"enabled": False, "max_damage_count": 12}):
            self.payload["damage_selection"] = selection
            validate_config(self.payload)
        for value in (1, 0, "true", None):
            self.payload["damage_selection"] = {"enabled": value}
            self.assert_invalid_at(self.payload, "damage_selection.enabled")

    def test_crack_thickness_is_an_optional_validated_block(self):
        self.payload["outputs"]["cracking_risk"] = False
        for thickness in (None, {"elements": 1, "from": "upstream"},
                          {"elements": 8, "from": "downstream"}):
            self.payload["damage_selection"] = {"crack_thickness": thickness}
            validate_config(self.payload)
        for elements in (0, -1, 1.5, True, "2", None):
            self.payload["damage_selection"] = {"crack_thickness": {"elements": elements, "from": "upstream"}}
            self.assert_invalid_at(self.payload, "damage_selection.crack_thickness.elements")
        self.payload["damage_selection"] = {"crack_thickness": {"elements": 2, "from": "sideways"}}
        self.assert_invalid_at(self.payload, "damage_selection.crack_thickness.from")
        self.payload["damage_selection"] = {"crack_thickness": {"elements": 2, "from": "upstream"},
                                            "max_damage_count": 12}
        self.assert_invalid_at(self.payload, "damage_selection.crack_thickness")
        for block in ({"elements": 2}, {"from": "upstream"}, {"elements": 2, "from": "upstream", "x": 1}, []):
            self.payload["damage_selection"] = {"crack_thickness": block}
            self.assertRaises(ConfigurationValidationError, validate_config, self.payload)

    def test_obsolete_interface_overrides_are_rejected(self):
        self.assertNotIn("interfaces", self.payload)
        self.assertIs(validate_config(self.payload), self.payload)
        self.payload["interfaces"] = {
            "wall_bedrock": {"model": "isotropic_linear_spring", "stiffness_n_per_m3": 7e11},
        }
        self.assert_invalid_at(self.payload, "interfaces")

    def test_support_springs_accept_zero_but_reject_invalid_directional_values(self):
        for region in ("bedrock",):
            for axis in ("x", "y", "z"):
                springs = self.payload["geometry"][region]["support"]["spring_n_per_m3"]
                original = springs[axis]
                springs[axis] = 0.
                validate_config(self.payload)
                for value in (-1., True, float("inf"), float("nan")):
                    springs[axis] = value
                    with self.subTest(region=region, axis=axis, value=value):
                        self.assert_invalid_at(
                            self.payload, f"geometry.{region}.support.spring_n_per_m3.{axis}",
                        )
                springs[axis] = original

    def test_output_flags_must_be_booleans(self):
        self.payload["outputs"]["max_displacement"] = 1
        self.assert_invalid_at(self.payload, "outputs.max_displacement")

    def test_custom_section_is_optional_and_sanitized(self):
        payload = copy.deepcopy(self.payload)
        payload["custom"] = {
            "notes": "Engineer notes.\nLine two.", "enabled": True,
            "factor": 1.5, "flags": ["a", "b"], "empty": None,
        }
        self.assertIs(validate_config(payload), payload)
        for key, value in (("Bad Key", "x"), ("notes", "<script>x</script>"),
                           ("notes", "a < b"), ("notes", "x\x00y"), ("notes", "a" * 4001),
                           ("nested", {"a": 1}), ("deep", [[1]])):
            bad = copy.deepcopy(self.payload)
            bad["custom"] = {key: value}
            with self.subTest(key=key):
                path = f"custom.{key}" if key != "Bad Key" else "custom.Bad Key"
                self.assert_invalid_at(bad, path)
        bad = copy.deepcopy(self.payload)
        bad["custom"] = "not an object"
        self.assert_invalid_at(bad, "custom")

    def test_tiny_global_element_size_warns_but_validates(self):
        from Elmer.model_config import collect_warnings
        payload = copy.deepcopy(self.payload)
        payload["geometry"]["global_element_size_m"] = 0.1
        payload["geometry"]["local_element_size_m"] = 0.1
        self.assertIs(validate_config(payload), payload)
        warnings = {w.path: w.message for w in collect_warnings(payload)}
        self.assertIn("geometry.global_element_size_m", warnings)
        self.assertIn("target_wall_tetrahedral_elements", warnings["geometry.global_element_size_m"])
        # The shipped default (0.5) must stay warning-free.
        clean = {w.path for w in collect_warnings(self.payload)}
        self.assertNotIn("geometry.global_element_size_m", clean)

    def test_cracking_seed_settings_are_optional_and_validated(self):
        from Elmer.model_config import collect_warnings
        self.assertEqual(self.payload["cracking"], [])
        self.payload["damage_selection"].pop("damaged_element_properties", None)
        self.payload["damage_selection"].pop("damaged_joint_friction", None)
        self.payload["damage_selection"].pop("lift_joints", None)
        self.payload["cracking"] = [{
            "face": "upstream", "depth_m": 3.0, "opening_m": 0.002,
            "full_width": True, "buffer_m": 1.0, "length_m": None,
        }]
        self.assertEqual(self.payload["cracking"], [{
            "face": "upstream", "depth_m": 3.0, "opening_m": 0.002,
            "full_width": True, "buffer_m": 1.0, "length_m": None,
        }])
        self.assertIs(self.payload["outputs"]["cracking_risk"], True)
        for key, value in (("face", "crest"), ("depth_m", 0.0), ("depth_m", 4.5),
                           ("opening_m", 0.0005), ("opening_m", 0.02), ("buffer_m", -0.1),
                           ("buffer_m", 2.5), ("full_width", "yes"), ("length_m", 0.0),
                           ("length_m", True), ("unknown", 1)):
            payload = copy.deepcopy(self.payload)
            payload["cracking"][0][key] = value
            with self.subTest(key=key, value=value):
                self.assert_invalid_at(payload, f"cracking[0].{key}")
        payload = copy.deepcopy(self.payload)
        payload["cracking"][0]["full_width"] = False
        self.assert_invalid_at(payload, "cracking[0].length_m")
        payload["cracking"][0].update(length_m=10.0, buffer_m=0.0, face="downstream")
        self.assertIs(validate_config(payload), payload)

        payload["cracking"][0]["full_width"] = True
        self.assertIn("cracking[0].length_m", {w.path for w in collect_warnings(payload)})
        for cracks in ({"face": "upstream"}, [copy.deepcopy(self.payload["cracking"][0])] * 2):
            payload = copy.deepcopy(self.payload)
            payload["cracking"] = cracks
            with self.subTest(cracks=cracks):
                self.assert_invalid_at(payload, "cracking")
        payload = copy.deepcopy(self.payload)
        payload["outputs"]["cracking_risk"] = False
        self.assert_invalid_at(payload, "outputs.cracking_risk")
        # An empty array means no crack (baselines only) and needs cracking_risk false.
        payload["cracking"] = []
        self.assertIs(validate_config(payload), payload)
        payload["outputs"]["cracking_risk"] = True
        self.assert_invalid_at(payload, "outputs.cracking_risk")
        del self.payload["cracking"]
        self.assertIs(validate_config(self.payload), self.payload)

    def damaged_element_fixture(self) -> dict:
        """A configuration using the legacy degraded-element band instead of lift joints."""
        payload = copy.deepcopy(self.payload)
        selection = payload["damage_selection"]
        selection.pop("lift_joints", None)
        selection["damaged_element_properties"] = {**payload["material"]["concrete"],
                                                   "compressive_strength_pa": 30e6,
                                                   "cohesive_fracture_energy_j_m2": 100.0}
        return payload

    def test_damaged_element_properties_are_optional_and_physically_validated(self):
        payload = self.damaged_element_fixture()
        validate_config(payload)
        properties = payload["damage_selection"]["damaged_element_properties"]
        for key in properties:
            for value in (None, True, "bad", float("inf"), float("nan"), -1, 0):
                if key == "poisson_ratio" and value == 0:
                    continue
                with self.subTest(key=key, value=value):
                    changed = copy.deepcopy(payload)
                    changed["damage_selection"]["damaged_element_properties"][key] = value
                    self.assert_invalid_at(changed, f"damage_selection.damaged_element_properties.{key}")
        changed = copy.deepcopy(payload)
        changed["damage_selection"]["damaged_element_properties"]["poisson_ratio"] = .5
        self.assert_invalid_at(changed, "damage_selection.damaged_element_properties.poisson_ratio")

    def test_damaged_joint_friction_is_optional_and_validated(self):
        self.payload = self.damaged_element_fixture()
        self.payload["damage_selection"]["damaged_joint_friction"] = {"friction_coefficient": 1.0}
        validate_config(self.payload)
        self.payload["damage_selection"]["damaged_joint_friction"] = {"friction_coefficient": .75,
                                                                      "adhesion_pa": 0.0}
        validate_config(self.payload)
        for coefficient in (None, True, "bad", 0, 5e-4, 11):
            payload = copy.deepcopy(self.payload)
            payload["damage_selection"]["damaged_joint_friction"] = {"friction_coefficient": coefficient}
            self.assert_invalid_at(payload, "damage_selection.damaged_joint_friction.friction_coefficient")
        payload = copy.deepcopy(self.payload)
        payload["damage_selection"]["damaged_joint_friction"] = {"friction_coefficient": 1.0, "adhesion_pa": -1}
        self.assert_invalid_at(payload, "damage_selection.damaged_joint_friction.adhesion_pa")
        payload = copy.deepcopy(self.payload)
        payload["outputs"]["cracking_risk"] = False
        payload["damage_selection"].pop("damaged_element_properties")
        payload["damage_selection"]["damaged_joint_friction"] = {"friction_coefficient": 1.0}
        self.assert_invalid_at(payload, "damage_selection.damaged_joint_friction")

    def test_lift_joints_default_to_a_poorly_bonded_construction_joint(self):
        joints = self.payload["damage_selection"]["lift_joints"]
        concrete = self.payload["material"]["concrete"]
        # FERC/EPRI: a poorly prepared lift surface keeps about a tenth of the parent
        # tensile strength. Its fracture energy is lower still, so the JOINT_MECA_RUPT joint
        # (K_N = f_t^2 * 11 / (2 Gf)) stays at least as stiff as the elements beside it and
        # the full-separation opening 2 Gf / f_t does not exceed the concrete's.
        self.assertEqual(joints["tensile_strength_pa"], 0.1 * concrete["tensile_strength_pa"])
        self.assertEqual(joints["cohesive_fracture_energy_j_m2"], 0.05 * concrete["cohesive_fracture_energy_j_m2"])
        self.assertLessEqual(joints["cohesive_fracture_energy_j_m2"] / joints["tensile_strength_pa"],
                             concrete["cohesive_fracture_energy_j_m2"] / concrete["tensile_strength_pa"])
        self.assertEqual(joints["max_normal_deviation_deg"], 15.0)
        self.assertIs(joints["include_base_contact"], False)
        self.assertEqual(joints["law"], "JOINT_MECA_RUPT")
        self.assertEqual(joints["alpha"], 1.0)
        # The seat is not softened: a softer, tension-carrying seat takes the heel lift away
        # from the lift plane (Gairezi/taking_a_breath.md).
        self.assertNotIn("support_degradation", self.payload["damage_selection"])
        # The shipped default degrades the band's elements too (seeded with the concrete
        # values, to be tuned in the app) and weakens the lift planes between them.
        self.assertEqual(self.payload["damage_selection"]["damaged_element_properties"],
                         self.payload["material"]["concrete"])
        self.assertIs(validate_config(self.payload), self.payload)

    def test_cracked_preset_is_the_same_dam_with_a_sliding_lift_plane(self):
        from Elmer.model_config import collect_warnings
        cracked = json.loads((Path(__file__).resolve().parents[1] / "cracked_config.json").read_text(encoding="utf-8"))
        validate_config(cracked)
        # Only the damage scenario may differ: there is one dam, not two copies of it.
        self.assertEqual({key for key in set(cracked) | set(self.payload) if cracked.get(key) != self.payload.get(key)},
                         {"damage_selection"})
        selection = cracked["damage_selection"]
        self.assertEqual(set(selection), {"enabled", "crack_water_pressure", "max_damage_count",
                                          "maximum_overload_factor", "lift_joints", "vertical_joint"})
        self.assertIs(selection["enabled"], True)
        self.assertIs(selection["vertical_joint"]["enabled"], True)
        self.assertEqual(selection["vertical_joint"]["critical_opening_m"], 0.002)
        # Water must be able to enter the plane, and nothing else is softened.
        self.assertIs(selection["crack_water_pressure"], True)
        joints = selection["lift_joints"]
        self.assertEqual((joints["law"], joints["friction_coefficient"], joints["uplift"]),
                         ("JOINT_MECA_FROT", 0.7, "linear_upstream_to_downstream"))
        self.assertIn("plane_elevation_m", joints)
        concrete = cracked["material"]["concrete"]
        # Poorly bonded: about 1 % of the concrete's tensile strength; the cone's apex stays there.
        self.assertLessEqual(joints["tensile_strength_pa"], 0.01 * concrete["tensile_strength_pa"])
        self.assertEqual({warning.path for warning in collect_warnings(cracked, three_body=True)}
                         - {"geometry.wall.height_m"}, set())

    def test_lift_joints_are_validated(self):
        path = "damage_selection.lift_joints"
        minimal = {"tensile_strength_pa": 2e5, "cohesive_fracture_energy_j_m2": 10.0}
        payload = copy.deepcopy(self.payload)
        payload["damage_selection"]["lift_joints"] = minimal
        validate_config(payload)
        for key in ("tensile_strength_pa", "cohesive_fracture_energy_j_m2"):
            for value in (None, True, "bad", float("inf"), -1, 0):
                with self.subTest(key=key, value=value):
                    changed = copy.deepcopy(payload)
                    changed["damage_selection"]["lift_joints"][key] = value
                    self.assert_invalid_at(changed, f"{path}.{key}")
            changed = copy.deepcopy(payload)
            del changed["damage_selection"]["lift_joints"][key]
            self.assert_invalid_at(changed, f"{path}.{key}")
        for key, bad in (("max_normal_deviation_deg", 0), ("max_normal_deviation_deg", 90),
                         ("max_normal_deviation_deg", "15"), ("include_base_contact", 1),
                         ("law", "JOINT_MECA_ENDO"), ("law", "CZM_LIN_REG"),
                         ("alpha", -0.1), ("alpha", 2.1), ("pena_contact", 0), ("unknown", 1),
                         ("plane_elevation_m", "low"), ("plane_elevation_m", float("nan")),
                         ("uplift", "full"), ("uplift", True)):
            with self.subTest(key=key, value=bad):
                changed = copy.deepcopy(payload)
                changed["damage_selection"]["lift_joints"][key] = bad
                self.assert_invalid_at(changed, f"{path}.{key}")
        for key, good in (("max_normal_deviation_deg", 89.9), ("alpha", 0), ("alpha", 2),
                          ("include_base_contact", False), ("pena_contact", 3.0), ("uplift", "none")):
            with self.subTest(key=key, value=good):
                changed = copy.deepcopy(payload)
                changed["damage_selection"]["lift_joints"][key] = good
                validate_config(changed)
        # include_base_contact is informational: the seat is a spring layer and softening it
        # is no longer a precondition (a softer seat takes the heel lift away from the plane).
        changed = copy.deepcopy(payload)
        changed["damage_selection"]["lift_joints"]["include_base_contact"] = True
        changed["damage_selection"].pop("support_degradation", None)
        validate_config(changed)
        # Mohr-Coulomb sliding law: needs its friction coefficient, which is otherwise rejected.
        frictional = copy.deepcopy(payload)
        frictional["damage_selection"]["lift_joints"]["law"] = "JOINT_MECA_FROT"
        self.assert_invalid_at(frictional, f"{path}.friction_coefficient")
        for value in (0, 1e-4, 10.1, "0.7", True):
            with self.subTest(friction_coefficient=value):
                changed = copy.deepcopy(frictional)
                changed["damage_selection"]["lift_joints"]["friction_coefficient"] = value
                self.assert_invalid_at(changed, f"{path}.friction_coefficient")
        frictional["damage_selection"]["lift_joints"]["friction_coefficient"] = 0.7
        validate_config(frictional)
        changed = copy.deepcopy(payload)
        changed["damage_selection"]["lift_joints"]["friction_coefficient"] = 0.7
        self.assert_invalid_at(changed, f"{path}.friction_coefficient")
        # A pour plane at one elevation replaces the peak/base band.
        placed = copy.deepcopy(payload)
        placed["damage_selection"]["lift_joints"]["plane_elevation_m"] = -15.0
        placed["damage_selection"]["crack_base"] = True
        self.assert_invalid_at(placed, f"{path}.plane_elevation_m")
        placed["damage_selection"]["crack_base"] = False
        validate_config(placed)
        # Uplift inside the plane rides the coupled crack-water solve.
        wet = copy.deepcopy(payload)
        wet["damage_selection"]["lift_joints"]["uplift"] = "linear_upstream_to_downstream"
        wet["damage_selection"]["crack_water_pressure"] = False
        self.assert_invalid_at(wet, f"{path}.uplift")
        wet["damage_selection"]["crack_water_pressure"] = True
        validate_config(wet)
        # Lift joints combine with degraded elements: the band's elements take the degraded
        # material and the lift planes between them the weak joint law.
        combined = self.damaged_element_fixture()
        combined["damage_selection"]["lift_joints"] = minimal
        validate_config(combined)
        combined["damage_selection"]["damaged_joint_friction"] = {"friction_coefficient": 1.0}
        validate_config(combined)
        # The lift planes come from the uncapped band, so a capped count is rejected up front.
        capped = copy.deepcopy(payload)
        capped["damage_selection"]["max_damage_count"] = 12
        capped["damage_selection"].pop("crack_base", None)
        self.assert_invalid_at(capped, path)
        # Lift joints are enough to justify cracking_risk without a manual crack.
        changed = copy.deepcopy(payload)
        changed["outputs"]["cracking_risk"] = True
        changed["cracking"] = []
        validate_config(changed)

    def test_damage_selection_warns_about_choices_that_defeat_the_weak_joint(self):
        from Elmer.model_config import collect_warnings

        def paths(config):
            return {warning.path for warning in collect_warnings(config)}

        base = copy.deepcopy(self.payload)
        concrete = base["material"]["concrete"]
        concrete.update(tensile_strength_pa=2e6, cohesive_fracture_energy_j_m2=100.0)
        selection = base["damage_selection"]
        selection.pop("support_degradation", None)
        selection.pop("damaged_element_properties", None)
        selection["lift_joints"] = {"tensile_strength_pa": 2e5, "cohesive_fracture_energy_j_m2": 5.0}
        seat = "damage_selection.support_degradation.interface_scale"
        lift_energy = "damage_selection.lift_joints.cohesive_fracture_energy_j_m2"
        element_energy = "damage_selection.damaged_element_properties.cohesive_fracture_energy_j_m2"
        self.assertFalse({seat, lift_energy, element_energy} & paths(base))
        # A softened, tension-carrying seat takes the heel lift away from the lift plane.
        softened = copy.deepcopy(base)
        softened["damage_selection"]["support_degradation"] = {"interface_scale": 0.1}
        self.assertIn(seat, paths(softened))
        softened["damage_selection"]["support_degradation"] = {"interface_scale": 1.0}
        self.assertNotIn(seat, paths(softened))
        softened["damage_selection"].pop("lift_joints")
        softened["damage_selection"]["support_degradation"] = {"interface_scale": 0.1}
        self.assertNotIn(seat, paths(softened))
        # Lower strength with the same energy: 2 Gf / ft grows from 0.1 mm to 1 mm.
        ductile = copy.deepcopy(base)
        ductile["damage_selection"]["lift_joints"]["cohesive_fracture_energy_j_m2"] = 100.0
        self.assertIn(lift_energy, paths(ductile))
        ductile["damage_selection"]["lift_joints"].update(law="JOINT_MECA_FROT", friction_coefficient=0.7)
        self.assertNotIn(lift_energy, paths(ductile))
        ductile["damage_selection"]["damaged_element_properties"] = dict(
            concrete, tensile_strength_pa=5e5, cohesive_fracture_energy_j_m2=150.0)
        self.assertIn(element_energy, paths(ductile))
        ductile["damage_selection"]["damaged_element_properties"]["cohesive_fracture_energy_j_m2"] = 20.0
        self.assertNotIn(element_energy, paths(ductile))

    def test_phase_field_controls_are_explicit_and_validated(self):
        self.add_legacy_crack_fixture()
        crack = self.payload["crack_simulations"][0]
        crack["phase_field_parameters"]["model"] = "AT3"
        crack["phase_field_parameters"]["residual_stiffness"] = 1.0
        crack["loading"]["load_steps"] = 0
        crack["solver"]["field_output_interval"] = 0
        crack["solver"]["staggered_tolerance"] = 1.0
        crack["solver"]["maximum_refinement_levels"] = -1
        crack["solver"]["maximum_refined_cells"] = 0
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
        self.assertIn(f"{prefix}.solver.maximum_refined_cells", paths)

    def test_crack_solver_rejects_unknown_fields(self):
        self.add_legacy_crack_fixture()
        self.payload["crack_simulations"][0]["solver"]["silent_failure"] = True
        self.assert_invalid_at(self.payload, "crack_simulations[0].solver.silent_failure")

    def add_legacy_crack_fixture(self):
        self.payload["crack_simulations"] = [{
            "name": "scenario_1", "type": "phase_field", "shape": "rectangular",
            "central_crack_pos_m": {"x": 76.7430082340184, "y": 13.946708830164285, "z": -25.0},
            "size_m": {"x": 3, "y": 5, "z": 0.02},
            "orientation": {"normal_x": 0, "normal_y": 0, "normal_z": 1},
            "phase_field_parameters": {
                "model": "AT2", "energy_split": "volumetric_deviatoric", "length_scale_m": 0.1,
                "critical_energy_release_rate_J_m2": 2700.0, "residual_stiffness": 1e-8,
            },
            "loading": {"internal_pressure_pa": 0, "load_steps": 10},
            "solver": json.loads((CONFIG_PATH.parent / "FEniCSx" / "fracture_defaults.json").read_text())["solver"],
        }]
        validate_config(self.payload)

    def test_water_reference_level_must_match_its_components(self):
        self.payload["loads"]["overflow_water_level"] = 35
        self.assert_invalid_at(self.payload, "loads.overflow_water_level")

    def test_rejects_non_object_configuration(self):
        with self.assertRaises(ConfigurationValidationError) as context:
            validate_config([])
        self.assertEqual(context.exception.issues[0].path, "")


if __name__ == "__main__":
    unittest.main()
