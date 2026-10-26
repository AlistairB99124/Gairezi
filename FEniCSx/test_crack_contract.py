"""Windows/WSL shared physical UI seed tests with no DOLFINx dependency."""
import copy
from dataclasses import asdict
import json
import math
import unittest

import numpy as np

from FEniCSx.automatic_crack import DEFAULTS, locate_crack
from FEniCSx.crack_contract import (
    apply_fracture_debug, default_fracture_debug, freeze_selected_seed, load_schedule, validate_selected_seed,
)
from Elmer.model_config import load_config


class SharedSeedTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config["geometry"]["wall"].pop("thickness_steps", None)
        self.y, self.z = 78 * math.cos(108.5 / 78), -27.5
        radii = np.array([76., 77., 78., 79., 80.])
        theta = 108.5 / 78
        self.points = np.column_stack((radii * math.sin(theta), radii * math.cos(theta),
                                       np.full(5, self.z)))
        stress = np.zeros((5, 6))
        stress[:, 0] = np.arange(1, 6) * 1e5
        self.fields = self.points, stress, np.ones(5, dtype=int)

    def seed(self):
        spec, options, selection = locate_crack(self.config, (self.y, self.z), self.fields)
        return spec, options, selection, freeze_selected_seed(spec, selection)

    def test_ui_placement_is_callable_without_dolfinx_and_frozen_exactly(self):
        spec, options, selection, contract = self.seed()
        payload = validate_selected_seed(json.loads(json.dumps(contract)))
        self.assertEqual(payload["selected_location_m"], {"y": self.y, "z": self.z})
        self.assertEqual(payload["seed_face"], "upstream")
        self.assertEqual(payload["geometry"]["annular_sector"], list(spec.annular_sector))
        self.assertEqual(payload["geometry"]["region_ids"], [1])
        self.assertEqual(payload["geometry"]["elevation_m"], self.z)
        self.assertEqual(payload["loading"]["fracture_energy_j_m2"], spec.critical_energy_release_rate_j_m2)
        np.testing.assert_allclose(payload["loading"]["load_factors"], np.linspace(0, 1, 11))
        self.assertEqual(options.maximum_refined_cells, 4_000_000)
        self.assertEqual(len(contract["sha256"]), 64)
        json.dumps(asdict(spec), allow_nan=False)

    def test_same_input_has_same_hash_and_changed_ui_location_changes_hash(self):
        spec, _, selection, contract = self.seed()
        self.assertEqual(contract, freeze_selected_seed(spec, selection))
        other_spec, _, other_selection = locate_crack(self.config, (self.y, self.z + .1), self.fields)
        self.assertNotEqual(contract["sha256"], freeze_selected_seed(other_spec, other_selection)["sha256"])

    def test_changing_sector_loading_or_click_after_freezing_is_rejected(self):
        _, _, _, contract = self.seed()
        for section, key, value in (
            ("geometry", "annular_sector", [75, 80, 1, 2]),
            ("loading", "internal_pressure_pa", 1e6),
            ("selected_location_m", "z", -26),
        ):
            changed = copy.deepcopy(contract)
            changed["payload"][section][key] = value
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "hash"):
                validate_selected_seed(changed)

    def test_source_location_and_specification_cannot_disagree(self):
        spec, _, selection, _ = self.seed()
        selection["selected_location_m"]["z"] += 1
        with self.assertRaisesRegex(ValueError, "elevation"):
            freeze_selected_seed(spec, selection)

    def test_final_substeps_share_the_actual_solver_schedule(self):
        spec, _, selection, _ = self.seed()
        contract = freeze_selected_seed(spec, selection, final_load_substeps=4)
        factors = validate_selected_seed(contract)["loading"]["load_factors"]
        self.assertEqual(factors, [0., *load_schedule(spec.load_steps, 4)])
        np.testing.assert_allclose(factors[-4:], [.925, .95, .975, 1.])
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                freeze_selected_seed(spec, selection, final_load_substeps=invalid)


class FractureDebugTests(unittest.TestCase):
    """fracture_debug overrides of fracture_defaults.json (process-of-elimination runs)."""

    def setUp(self):
        self.shared = SharedSeedTests("test_ui_placement_is_callable_without_dolfinx_and_frozen_exactly")
        self.shared.setUp()
        self.config = self.shared.config
        self.config["fracture_debug"] = None
        self.defaults = json.loads(DEFAULTS.read_text(encoding="utf-8"))
        self.aster = json.loads((DEFAULTS.parents[1] / "Aster" / "fracture_defaults.json").read_text())

    def test_null_and_absent_blocks_both_use_the_solver_defaults(self):
        null_spec, null_options, _ = locate_crack(self.config, (self.shared.y, self.shared.z), self.shared.fields)
        absent = {key: value for key, value in self.config.items() if key != "fracture_debug"}
        spec, options, _ = locate_crack(absent, (self.shared.y, self.shared.z), self.shared.fields)
        self.assertEqual((spec, options), (null_spec, null_options))
        self.assertEqual((spec.model, spec.critical_energy_release_rate_j_m2),
                         (self.defaults["model"], self.defaults["critical_energy_release_rate_j_m2"]))
        self.assertIsNone(load_config({**load_config(), "fracture_debug": None})["fracture_debug"])
        self.assertEqual(apply_fracture_debug(self.defaults, {"fracture_debug": None}), (self.defaults, 1.0))

    def test_authoritative_config_defaults_to_fridays_working_values(self):
        self.config["fracture_debug"] = load_config()["fracture_debug"]
        spec, _, report = locate_crack(self.config, (self.shared.y, self.shared.z), self.shared.fields)
        self.assertEqual((spec.model, spec.critical_energy_release_rate_j_m2, spec.length_scale_m),
                         ("AT2", 2700.0, 0.25))
        self.assertEqual(spec.internal_pressure_pa, 0.0)
        self.assertEqual(report["crack_water_pressure"]["factor"], 0.0)

    def test_default_block_reproduces_the_shipped_defaults(self):
        baseline, baseline_options, baseline_report = locate_crack(
            self.config, (self.shared.y, self.shared.z), self.shared.fields)
        debug = default_fracture_debug(self.defaults, self.aster)
        self.assertIsNone(debug["length_scale_m"])
        self.assertEqual(debug["crack_water_pressure_factor"], 1.0)
        self.assertEqual(debug["aster"], {"band_width_m": 0.5, "softening_p": 5.0})
        self.assertEqual(load_config({**load_config(), "fracture_debug": debug})["fracture_debug"], debug)
        spec, options, report = locate_crack({**self.config, "fracture_debug": debug},
                                             (self.shared.y, self.shared.z), self.shared.fields)
        self.assertEqual(spec, baseline)
        self.assertEqual(options, baseline_options)
        self.assertEqual(report["crack_water_pressure"]["pressure_pa"],
                         baseline_report["crack_water_pressure"]["pressure_pa"])

    def test_overrides_reach_the_seed_specification_and_solver_options(self):
        self.config["fracture_debug"] = {
            "model": "AT2", "critical_energy_release_rate_j_m2": 2700.0, "length_scale_m": 0.25,
            "residual_stiffness": 1e-6, "crack_water_pressure_factor": 0.0, "load_steps": 20,
            "solver": {"maximum_staggered_iterations": 80, "staggered_tolerance": 1e-4},
        }
        spec, options, report = locate_crack(self.config, (self.shared.y, self.shared.z), self.shared.fields)
        self.assertEqual((spec.model, spec.critical_energy_release_rate_j_m2, spec.length_scale_m),
                         ("AT2", 2700.0, 0.25))
        self.assertEqual((spec.residual_stiffness, spec.load_steps, spec.internal_pressure_pa), (1e-6, 20, 0.0))
        self.assertEqual((options.maximum_staggered_iterations, options.staggered_tolerance), (80, 1e-4))
        self.assertEqual(options.mechanics_maximum_iterations, self.defaults["solver"]["mechanics_maximum_iterations"])
        calibration = report["length_scale_calibration"]
        self.assertTrue(calibration["basis"].startswith("fixed"))
        # Friday's settings: AT2, Gc 2700 J/m2, l 0.25 m imply about 6.3 MPa.
        self.assertAlmostEqual(calibration["implied_homogeneous_tensile_strength_pa"] / 1e6, 6.3, delta=0.05)
        self.assertEqual(report["fracture_debug"], self.config["fracture_debug"])
        frozen = freeze_selected_seed(spec, report)["payload"]["loading"]
        self.assertEqual(frozen["internal_pressure_pa"], 0.0)
        self.assertEqual(len(frozen["load_factors"]), 21)

    def test_partial_crack_water_scales_the_hydrostatic_head(self):
        _, _, full = locate_crack(self.config, (self.shared.y, self.shared.z), self.shared.fields)
        self.config["fracture_debug"] = {"crack_water_pressure_factor": 0.25}
        spec, _, report = locate_crack(self.config, (self.shared.y, self.shared.z), self.shared.fields)
        self.assertAlmostEqual(spec.internal_pressure_pa, 0.25 * full["crack_water_pressure"]["pressure_pa"])
        self.assertEqual(report["crack_water_pressure"]["factor"], 0.25)

    def test_invalid_overrides_are_rejected(self):
        for debug in (
            {"model": "AT3"}, {"critical_energy_release_rate_j_m2": 0}, {"length_scale_m": -1},
            {"crack_water_pressure_factor": 1.5}, {"load_steps": 0}, {"load_steps": 2.5},
            {"solver": {"unknown": 1}}, {"solver": {"maximum_refined_cells": 0}}, {"other": 1},
        ):
            with self.subTest(debug=debug):
                with self.assertRaises(ValueError):
                    apply_fracture_debug(self.defaults, {"fracture_debug": debug})
                with self.assertRaises(ValueError):
                    load_config({**load_config(), "fracture_debug": debug})
        with self.assertRaises(ValueError):
            load_config({**load_config(), "fracture_debug": {"aster": {"softening_p": 0.5}}})


if __name__ == "__main__":
    unittest.main()
