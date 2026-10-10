"""Analytical normal-contact checks, release, failure and exported time series."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from configuration import load_scenarios
from contact_benchmark import solve
import run as workflow

HERE = Path(__file__).resolve().parent


class ContactBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.scenario = next(s for s in json.loads((HERE / "cracking.json").read_text())
                             if s["kind"] == "contact_benchmark")
        self.material = {"youngs_modulus": 35e9}

    def test_open_close_pressure_release_and_mesh_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = solve(self.scenario, self.material, Path(directory))
        self.assertEqual(summary["passed_steps"], 16)
        for divisions in self.scenario["mesh_divisions"]:
            rows = [r for r in summary["steps"] if r["mesh_divisions_per_bar"] == divisions]
            self.assertAlmostEqual(rows[0]["gap_m"], 0.00015, places=13)
            self.assertEqual(rows[0]["contact_pressure_pa"], 0)
            self.assertAlmostEqual(rows[4]["contact_pressure_pa"], 700000, places=5)
            self.assertAlmostEqual(rows[5]["contact_pressure_pa"], 1400000, places=5)
            self.assertAlmostEqual(rows[5]["gap_m"], 0, places=13)
            np.testing.assert_allclose(rows[5]["outer_reactions_n"], [14000, -14000], atol=1e-6)
            self.assertEqual(rows[6]["contact_pressure_pa"], 0)
            self.assertFalse(rows[6]["active_contact"])
            self.assertEqual(rows[6]["active_set_iterations"], 2)
            self.assertAlmostEqual(rows[7]["gap_m"], 0.00015, places=13)
            self.assertTrue(all(row["contact_force_n"] >= 0 for row in rows))
        coarse, fine = summary["steps"][:8], summary["steps"][8:]
        np.testing.assert_allclose([r["contact_pressure_pa"] for r in coarse],
                                   [r["contact_pressure_pa"] for r in fine], atol=1e-6)

    def test_zero_gap_keeps_coincident_lips_independent(self):
        self.scenario.update(initial_gap_m=0.0, outer_closures_m=[-1e-5, 1e-5, -1e-5])
        with tempfile.TemporaryDirectory() as directory:
            summary = solve(self.scenario, self.material, Path(directory))
        for row in summary["steps"]:
            if row["outer_closure_m"] < 0:
                self.assertAlmostEqual(row["gap_m"], 1e-5, places=13)
                self.assertEqual(row["contact_force_n"], 0)
            else:
                self.assertAlmostEqual(row["contact_pressure_pa"], 140000, places=5)

    def test_changed_lengths_area_and_modulus(self):
        self.scenario.update(bar_lengths_m=[0.5, 0.5], cross_section_m2=0.02, mesh_divisions=[1, 5])
        with tempfile.TemporaryDirectory() as directory:
            summary = solve(self.scenario, {"youngs_modulus": 1e9}, Path(directory))
        for row in summary["steps"]:
            if row["step"] == 5:
                self.assertAlmostEqual(row["contact_pressure_pa"], 100000, places=6)
                self.assertAlmostEqual(row["contact_force_n"], 2000, places=6)

    def test_iteration_limit_does_not_produce_success(self):
        self.scenario["verification"]["maximum_active_set_iterations"] = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "active set did not converge"):
                solve(self.scenario, self.material, Path(directory))

    def test_exported_time_series_contains_compressive_stress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            solve(self.scenario, self.material, path)
            series = ET.parse(path / "mesh_2" / "stress.pvd").findall(".//DataSet")
            self.assertEqual(len(series), 8)
            grid_path = path / "mesh_2" / series[5].attrib["file"]
            reader = vtk.vtkXMLPUnstructuredGridReader()
            reader.SetFileName(str(grid_path))
            reader.Update()
            self.assertEqual(reader.GetErrorCode(), 0)
            grid = reader.GetOutput()
            self.assertEqual(grid.GetNumberOfCells(), 4)
            stress = vtk_to_numpy(grid.GetCellData().GetArray("axial_stress_pa"))
            np.testing.assert_allclose(stress, -1400000, atol=1e-5)

    def test_runner_persists_contact_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = path / "cracking.json"
            config.write_text(json.dumps([self.scenario]))
            with patch.object(workflow, "HERE", path), patch.object(
                workflow, "environment", return_value={}
            ), patch("sys.argv", ["run.py", "run", "--config", str(config)]):
                self.assertEqual(workflow.main(), 0)
            parent = path / "results" / self.scenario["id"]
            latest = json.loads((parent / "latest.json").read_text())
            self.assertEqual(latest["status"], "passed_reduced_contact_benchmark")
            manifest = json.loads((parent / latest["run_directory"] / "manifest.json").read_text())
            self.assertFalse(manifest["engineering_validated"])

    def test_invalid_contact_configuration(self):
        for name, value in (
            ("mesh_divisions", [2, 2]), ("bar_lengths_m", [1, -1]),
            ("outer_closures_m", [0, 0.0002]), ("cross_section_m2", True),
            ("initial_gap_m", -1), ("outer_closures_m", [0, float("nan"), -1])
        ):
            with self.subTest(name=name):
                scenario = copy.deepcopy(self.scenario)
                scenario[name] = value
                with tempfile.TemporaryDirectory() as directory:
                    config = Path(directory) / "cracking.json"
                    config.write_text(json.dumps([scenario]))
                    with self.assertRaises(ValueError):
                        load_scenarios(config)


if __name__ == "__main__":
    unittest.main()
