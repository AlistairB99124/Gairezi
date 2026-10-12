"""Integration checks for isolated phase-field job setup and failure reporting."""
import json
from pathlib import Path
import tempfile
import unittest

from run import LOAD_CASE, run_fracture


class FractureRunnerTests(unittest.TestCase):
    def test_invalid_crack_location_produces_a_failed_isolated_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = json.loads(LOAD_CASE.read_text())
            config["crack_simulations"][0]["central_crack_pos_m"] = {
                "x": 0.0,
                "y": 87.0,
                "z": -28.0,
            }
            config_path = output / "invalid-config.json"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "does not intersect the concrete wall"):
                run_fracture(config_path, output, {"test_environment": True})
            jobs = list((output / "dam_phase_field").iterdir())
            self.assertEqual(len(jobs), 1)
            job = jobs[0]
            manifest = json.loads((job / "manifest.json").read_text())
            placement = json.loads((job / "crack_placement.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(manifest["engineering_validated"])
            self.assertIn("mesh_bounds", manifest["error"])
            self.assertFalse(placement[0]["valid"])
            self.assertEqual(json.loads((job / "config.json").read_text()), config)
            self.assertTrue((job / "import_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
