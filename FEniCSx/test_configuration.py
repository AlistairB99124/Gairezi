"""Configuration regression tests; no solver dependencies required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from configuration import load_scenarios

HERE = Path(__file__).resolve().parent


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = [s for s in json.loads((HERE / "cracking.json").read_text())
                          if s["kind"] in ("elastic_benchmark", "predefined_crack")]

    def load(self, scenarios):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cracking.json"
            path.write_text(json.dumps(scenarios))
            return load_scenarios(path)

    def test_default_configuration(self):
        self.assertEqual(len(self.load(self.scenarios)), 2)

    def test_invalid_benchmark_values(self):
        for key, value in (("id", "../escape"), ("enabled", 1), ("mesh_divisions", True),
                           ("axial_strain", float("nan")), ("kind", "unknown")):
            with self.subTest(key=key):
                scenarios = copy.deepcopy(self.scenarios)
                scenarios[0][key] = value
                with self.assertRaises(ValueError):
                    self.load(scenarios)

    def test_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load([self.scenarios[0], self.scenarios[0]])

    def test_unknown_parameters_rejected(self):
        self.scenarios[0]["typo"] = 1
        with self.assertRaises(ValueError):
            self.load(self.scenarios)

    def test_invalid_crack_values(self):
        for key, value in (("depth_m", -1), ("initial_opening_mm", float("inf")),
                           ("z_m", True), ("orientation", "diagonal")):
            with self.subTest(key=key):
                scenarios = copy.deepcopy(self.scenarios)
                scenarios[1]["crack"][key] = value
                with self.assertRaises(ValueError):
                    self.load(scenarios)

    def test_coulomb_requires_coefficient(self):
        self.scenarios[1]["contact"]["model"] = "coulomb"
        with self.assertRaises(ValueError):
            self.load(self.scenarios)
        self.scenarios[1]["contact"]["friction_coefficient"] = 0.5
        self.load(self.scenarios)

    def test_vertical_crack(self):
        crack = self.scenarios[1]["crack"]
        crack.update(orientation="vertical", z_lo_m=-25.0, z_hi_m=-20.0)
        del crack["z_m"]
        del crack["width_m"]
        self.load(self.scenarios)
        crack["z_hi_m"] = -30.0
        with self.assertRaises(ValueError):
            self.load(self.scenarios)

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            self.load([])

    def test_default_dam_configuration(self):
        all_scenarios = json.loads((HERE / "cracking.json").read_text())
        self.assertEqual(len(self.load(all_scenarios)), 4)

    def test_invalid_dam_parity_options(self):
        dam = next(s for s in json.loads((HERE / "cracking.json").read_text()) if s["kind"] == "dam_elastic")
        for key, value in (("reference", "../outside.pvtu"), ("maximum_relative_volume_change", -1)):
            with self.subTest(key=key):
                invalid = copy.deepcopy(dam)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    self.load([invalid])
        dam["parity"]["probes"][0]["point_m"] = [0, float("nan"), 1]
        with self.assertRaises(ValueError):
            self.load([dam])


if __name__ == "__main__":
    unittest.main()
