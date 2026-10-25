"""Windows/WSL shared physical UI seed tests with no DOLFINx dependency."""
import copy
from dataclasses import asdict
import json
import math
import unittest

import numpy as np

from FEniCSx.automatic_crack import locate_crack
from FEniCSx.crack_contract import freeze_selected_seed, load_schedule, validate_selected_seed
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


if __name__ == "__main__":
    unittest.main()
