"""Real elastic/export checks and scenario-runner failure regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from elastic_benchmark import solve
import run as workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = [s for s in json.loads((workflow.HERE / "cracking.json").read_text())
                          if s["kind"] in ("elastic_benchmark", "predefined_crack")]
        self.material = json.loads(workflow.LOAD_CASE.read_text())["material"]

    def test_elastic_solution_and_exported_stresses(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = solve(self.scenarios[0], self.material, output)
            self.assertLess(summary["relative_l2_displacement_error"], 1e-8)
            self.assertLess(summary["relative_energy_error"], 1e-8)
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(output / "stress_p0_000000.vtu"))
            reader.Update()
            tensor = vtk_to_numpy(reader.GetOutput().GetCellData().GetArray("stress_pa"))
            expected = np.zeros_like(tensor)
            expected[:, 0] = self.material["youngs_modulus"] * self.scenarios[0]["axial_strain"]
            np.testing.assert_allclose(tensor, expected, atol=0.01, rtol=1e-8)
            reader.SetFileName(str(output / "displacement_p0_000000.vtu"))
            reader.Update()
            displacement = reader.GetOutput()
            self.assertIsNotNone(displacement.GetPointData().GetArray("displacement_m"))

    def test_enabled_crack_fails_before_outputs(self):
        self.scenarios[1]["enabled"] = True
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            config = here / "cracking.json"
            config.write_text(json.dumps(self.scenarios))
            with patch.object(workflow, "HERE", here), patch.object(
                workflow, "environment", return_value={}
            ), patch("sys.argv", ["run.py", "run", "--config", str(config)]):
                with self.assertRaisesRegex(RuntimeError, "contact is not implemented"):
                    workflow.main()
            self.assertFalse((here / "results").exists())

    def test_failed_solve_records_failure_without_replacing_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            config = here / "cracking.json"
            config.write_text(json.dumps(self.scenarios))
            parent = here / "results" / "elastic_benchmark"
            parent.mkdir(parents=True)
            latest = parent / "latest.json"
            latest.write_text('{"run_directory": "previous-success"}')
            with patch.object(workflow, "HERE", here), patch.object(
                workflow, "environment", return_value={}
            ), patch("elastic_benchmark.solve", side_effect=RuntimeError("test solver failure")), patch(
                "sys.argv", ["run.py", "run", "--config", str(config)]
            ):
                with self.assertRaisesRegex(RuntimeError, "test solver failure"):
                    workflow.main()
            self.assertEqual(json.loads(latest.read_text())["run_directory"], "previous-success")
            manifests = list(parent.glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(json.loads(manifests[0].read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
