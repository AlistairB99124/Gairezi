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
        self.payload["geometry"].pop("foundation", None)
        self.payload["geometry"]["wall"].pop("step_end_taper_ratio", None)
        self.payload["geometry"]["bedrock"]["support"]["model"] = "bonded"
        validate_config(self.payload)
        self.payload["geometry"]["bedrock"]["support"]["model"] = "distributed_elastic_granite"
        validate_config(self.payload)

    def test_painted_tensile_strength_is_optional_finite_and_nonnegative(self):
        for value in (0., 1e6, 2e6):
            self.payload["material"]["concrete"]["selected_tensile_strength_pa"] = value
            validate_config(self.payload)
        for value in (-1., True, float("nan"), float("inf")):
            self.payload["material"]["concrete"]["selected_tensile_strength_pa"] = value
            self.assert_invalid_at(self.payload, "material.concrete.selected_tensile_strength_pa")

    def test_damaged_youngs_modulus_is_optional_positive_and_not_stiffer(self):
        concrete = self.payload["material"]["concrete"]
        for value in (1e3, 15e9, concrete["youngs_modulus"]):
            concrete["selected_youngs_modulus"] = value
            validate_config(self.payload)
        for value in (0., -1., True, float("nan"), concrete["youngs_modulus"] * 2):
            concrete["selected_youngs_modulus"] = value
            self.assert_invalid_at(self.payload, "material.concrete.selected_youngs_modulus")

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

    def test_cracking_seed_settings_are_optional_and_validated(self):
        from Elmer.model_config import collect_warnings
        self.assertEqual(self.payload["cracking"], [{
            "face": "upstream", "depth_m": 3.0, "opening_m": 0.002,
            "full_width": True, "buffer_m": 1.0, "length_m": None,
        }])
        self.assertIs(self.payload["outputs"]["cracking_risk"], True)
        for key, value in (("face", "crest"), ("depth_m", 0.0), ("depth_m", 4.0),
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
