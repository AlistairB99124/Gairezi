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

    def test_water_pressure_releases_contact_opens_and_recloses_zero_initial_gap(self):
        self.scenario.update(initial_gap_m=0., outer_closures_m=[.0001] * 5)
        pressures = [0., .7e6, 1.4e6, 2.1e6, 0.]
        with tempfile.TemporaryDirectory() as directory:
            summary = solve(self.scenario, self.material, Path(directory), face_pressures_pa=pressures)
        self.assertFalse(summary["solver_ready"])
        self.assertFalse(summary["engineering_validated"])
        for divisions in self.scenario["mesh_divisions"]:
            rows = [r for r in summary["steps"] if r["mesh_divisions_per_bar"] == divisions]
            np.testing.assert_allclose([r["contact_pressure_pa"] for r in rows],
                                       [1.4e6, .7e6, 0., 0., 1.4e6], atol=1e-6)
            np.testing.assert_allclose([r["gap_m"] for r in rows], [0., 0., 0., 5e-5, 0.], atol=1e-13)
            for row, pressure in zip(rows, pressures):
                total_force = max(1.4e6, pressure) * self.scenario["cross_section_m2"]
                np.testing.assert_allclose(row["outer_reactions_n"], [total_force, -total_force], atol=1e-6)
                self.assertAlmostEqual(sum(row["water_opening_forces_n"]), 0., places=12)
            self.assertFalse(rows[3]["active_contact"])
            self.assertTrue(rows[4]["active_contact"])

    def test_pressure_only_opening_obeys_bar_compliance_not_a_seed_gap(self):
        self.scenario.update(initial_gap_m=0., outer_closures_m=[0., 0., 0.])
        for young, area, lengths in ((35e9, .01, [1., 1.5]), (1e9, .02, [.5, .5])):
            self.scenario.update(cross_section_m2=area, bar_lengths_m=lengths)
            with tempfile.TemporaryDirectory() as directory:
                report = solve(self.scenario, {"youngs_modulus": young}, Path(directory),
                               face_pressures_pa=[0., 1e5, 0.])
            for row in report["steps"]:
                self.assertAlmostEqual(row["gap_m"], row["prescribed_face_pressure_pa"] * sum(lengths) / young,
                                       places=13)
                self.assertAlmostEqual(row["contact_force_n"], 0., places=6)

    def test_invalid_face_pressure_history_fails_before_output(self):
        for pressures in ([1.], [True] * 8, [-1.] * 8, [float("nan")] * 8, [[1.]] * 8):
            with self.subTest(pressures=pressures), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                with self.assertRaisesRegex(ValueError, "Face pressures"):
                    solve(self.scenario, self.material, path, face_pressures_pa=pressures)
                self.assertEqual(list(path.iterdir()), [])

    def test_coulomb_stick_slide_reverse_and_nonnegative_dissipation(self):
        self.scenario.update(initial_gap_m=0., outer_closures_m=[.0001] * 6)
        shifts = [0., .00005, .0002, .00025, .0002, -.0002]
        with tempfile.TemporaryDirectory() as directory:
            report = solve(self.scenario, self.material, Path(directory), tangential_shifts_m=shifts,
                           friction_coefficient=.5, shear_modulus_pa=14e9)
            for divisions in self.scenario["mesh_divisions"]:
                rows = [r for r in report["steps"] if r["mesh_divisions_per_bar"] == divisions]
                np.testing.assert_allclose([r["tangential_force_n"] for r in rows],
                                           [0., -2800., -7000., -7000., -4200., 7000.], atol=1e-6)
                np.testing.assert_allclose([r["plastic_slip_m"] for r in rows],
                                           [0., 0., .000075, .000125, .000125, -.000075], atol=1e-13)
                np.testing.assert_allclose([r["friction_dissipation_j"] for r in rows],
                                           [0., 0., .525, .35, 0., 1.4], atol=1e-9)
                self.assertEqual([r["sliding"] for r in rows], [False, False, True, True, False, True])
            series_path = Path(directory) / "mesh_2" / "shear_stress.pvd"
            series = ET.parse(series_path).findall(".//DataSet")
            self.assertEqual(len(series), len(shifts))
            reader = vtk.vtkXMLPUnstructuredGridReader()
            reader.SetFileName(str(series_path.parent / series[2].attrib["file"]))
            reader.Update()
            self.assertEqual(reader.GetErrorCode(), 0)
            np.testing.assert_allclose(
                vtk_to_numpy(reader.GetOutput().GetCellData().GetArray("shear_stress_pa")),
                700000., atol=1e-6,
            )

    def test_water_reduces_solid_friction_and_open_faces_cannot_carry_shear(self):
        self.scenario.update(initial_gap_m=0., outer_closures_m=[.0001] * 4)
        with tempfile.TemporaryDirectory() as directory:
            report = solve(
                self.scenario, self.material, Path(directory), face_pressures_pa=[0., .7e6, 2.1e6, 0.],
                tangential_shifts_m=[.0001] * 4, friction_coefficient=.5, shear_modulus_pa=14e9,
            )
        for divisions in self.scenario["mesh_divisions"]:
            rows = [r for r in report["steps"] if r["mesh_divisions_per_bar"] == divisions]
            np.testing.assert_allclose([r["tangential_force_n"] for r in rows], [-5600., -3500., 0., 0.], atol=1e-6)
            np.testing.assert_allclose([r["friction_limit_n"] for r in rows], [7000., 3500., 0., 7000.], atol=1e-6)
            self.assertGreater(rows[2]["gap_m"], 0.)
            self.assertEqual(rows[2]["friction_dissipation_j"], 0.)
            self.assertAlmostEqual(rows[3]["gap_m"], 0., places=13)

    def test_explicit_zero_friction_is_not_an_inferred_default(self):
        self.scenario.update(initial_gap_m=0., outer_closures_m=[.0001] * 3)
        with tempfile.TemporaryDirectory() as directory:
            report = solve(self.scenario, self.material, Path(directory),
                           tangential_shifts_m=[.001, -.001, 0.], friction_coefficient=0.,
                           shear_modulus_pa=14e9)
        for row in report["steps"]:
            self.assertAlmostEqual(row["tangential_force_n"], 0., places=12)
            self.assertAlmostEqual(row["tangential_jump_m"], row["tangential_outer_shift_m"], places=13)
            self.assertAlmostEqual(row["friction_dissipation_j"], 0., places=12)

    def test_invalid_or_partial_friction_inputs_are_rejected(self):
        valid = dict(tangential_shifts_m=[0.] * 8, friction_coefficient=.5, shear_modulus_pa=14e9)
        for change in (
            {"friction_coefficient": None}, {"shear_modulus_pa": None}, {"friction_coefficient": True},
            {"friction_coefficient": -1.}, {"shear_modulus_pa": 0.}, {"shear_modulus_pa": float("inf")},
            {"tangential_shifts_m": [0.]}, {"tangential_shifts_m": [True] * 8},
            {"tangential_shifts_m": None},
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    solve(self.scenario, self.material, Path(directory), **{**valid, **change})

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
