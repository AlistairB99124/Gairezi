"""A stage rerun may change fracture_debug but must reuse a physically identical baseline."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from three_body_model import physical_configuration, require_verified_baseline

ROOT = Path(__file__).resolve().parents[1]


class RerunConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.submitted = json.loads((ROOT / "config.json").read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        snapshot = json.dumps(self.submitted, indent=2).encode()
        (self.directory / "config.json").write_bytes(snapshot)
        self.sha256 = hashlib.sha256(snapshot).hexdigest()
        self.source = {"sha256": {"three_body.msh": "a", "config.json": "b", "diagnostic.json": "c"}}
        gate = {"passed": True, "relative_l2_displacement_difference": 1e-6,
                "relative_nodal_stress_difference": 1e-6}
        documents = {
            "manifest.json": {"status": "completed_not_engineering_validated",
                              "scenario_name": "Baseline_NoPhase", "configuration_sha256": self.sha256},
            "baseline_report.json": {"configuration_sha256": self.sha256, "three_body_verification": gate,
                                     "relative_force_balance_error": 1e-9,
                                     "stress_recovery": {"peak_measure": "recovered_nodal"}},
            "three_body_source.json": self.source,
        }
        for name, document in documents.items():
            (self.directory / name).write_text(json.dumps(document))

    def tearDown(self):
        self.temp.cleanup()

    def structure(self, config):
        return SimpleNamespace(three_body_config=config, three_body_source=self.source)

    def test_fracture_debug_change_reuses_baseline_snapshot_hash(self):
        rerun = copy.deepcopy(self.submitted)
        rerun["fracture_debug"] = None
        evidence = require_verified_baseline(self.structure(rerun), self.directory)
        self.assertEqual(evidence["configuration_sha256"], self.sha256)
        self.assertNotIn("fracture_debug", physical_configuration(rerun))

    def test_physical_change_rejects_baseline(self):
        changed = copy.deepcopy(self.submitted)
        changed["material"]["concrete"]["youngs_modulus"] *= 2
        with self.assertRaisesRegex(ValueError, "does not match this fracture job"):
            require_verified_baseline(self.structure(changed), self.directory)

    def test_tampered_snapshot_rejects_baseline(self):
        (self.directory / "config.json").write_text(json.dumps(self.submitted))
        with self.assertRaisesRegex(ValueError, "does not match this fracture job"):
            require_verified_baseline(self.structure(self.submitted), self.directory)


if __name__ == "__main__":
    unittest.main()
