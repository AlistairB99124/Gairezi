"""Tests for the wall-geometry sweep (no solver needed)."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cracking_sweeps import run_sweep as rs  # noqa: E402

CONFIG = rs.read_json(rs.HERE / "config.json")
FOUR_METRES = {"geometry": {"wall": {"radial_width_m": 4.0, "centerline_radius_m": 78.0, "downstream_radius_m": 76.0}}}
STEPPED = {"geometry": {"wall": {"radial_width_m": 4.0, "centerline_radius_m": 78.0, "downstream_radius_m": 76.0,
                                 "thickness_steps": [{"top_z_m": -20.0, "radial_width_m": 4.5},
                                                     {"top_z_m": -25.0, "radial_width_m": 5.0}]}}}
MORTAR = {"density": 2200.0, "youngs_modulus": 3.5e8, "poisson_ratio": 0.2,
          "bond_spring_n_per_m3": {"x": 1.25e9, "y": 1.25e9, "z": 1.25e9}}
FILLED = {**STEPPED, "material": {"mortar": MORTAR}}


class ScenarioTests(unittest.TestCase):
    def test_an_overwrite_replaces_only_what_it_names(self):
        base = {"a": {"b": 1, "c": [1, 2], "d": {"e": 5}}, "f": 2}
        merged = rs.merge(base, {"a": {"c": [], "d": {"g": 7}}, "h": {"i": 1}})
        self.assertEqual(merged, {"a": {"b": 1, "c": [], "d": {"e": 5, "g": 7}}, "f": 2, "h": {"i": 1}})
        self.assertEqual(base, {"a": {"b": 1, "c": [1, 2], "d": {"e": 5}}, "f": 2})  # the base is not altered
        self.assertEqual(rs.merge({"a": {"b": 1}}, {"a": None}), {"a": None})

    def test_each_scenario_is_the_config_with_its_overwrite(self):
        scenarios = rs.define_scenarios(CONFIG, [FOUR_METRES, STEPPED])
        self.assertEqual([scenario["number"] for scenario in scenarios], [1, 2])
        self.assertEqual([scenario["id"] for scenario in scenarios], ["scenario_01", "scenario_02"])
        first = scenarios[0]["config"]
        self.assertEqual(first["geometry"]["wall"]["radial_width_m"], 4.0)
        self.assertEqual(first["geometry"]["wall"]["upstream_radius_m"], CONFIG["geometry"]["wall"]["upstream_radius_m"])
        # Everything the overwrite does not name is the base configuration's.
        untouched = deepcopy(first)
        untouched["geometry"]["wall"] = CONFIG["geometry"]["wall"]
        self.assertEqual(untouched, CONFIG)
        self.assertEqual(scenarios[0]["geometry"], {
            "width_m": 4.0, "foundation_depth_m": CONFIG["geometry"]["foundation"]["depth_m"],
            "foundation_side_margin_m": CONFIG["geometry"]["foundation"]["side_margin_m"],
            "upstream_radius_m": 80.0, "steps": [], "mortar_fill": None})
        self.assertEqual(scenarios[1]["geometry"]["steps"], [{"width_m": 4.5, "top_z_m": -20.0},
                                                             {"width_m": 5.0, "top_z_m": -25.0}])
        self.assertEqual(rs.describe(scenarios[1]["geometry"]), "wall 4 m, 4.5 m below z = -20 m, 5 m below z = -25 m")

    def test_the_shipped_scenarios_are_the_seven_walls_of_the_sheet_and_the_stepped_ones_with_a_fill(self):
        overwrites = rs.read_json(rs.HERE / "scenarios.json")
        scenarios = rs.define_scenarios(CONFIG, overwrites)
        stepped = ["wall 4 m, 4.5 m below z = -20 m, 5 m below z = -25 m",
                   "wall 4 m, 4.5 m below z = -20 m", "wall 4 m, 5 m below z = -25 m"]
        self.assertEqual([rs.describe(scenario["geometry"]) for scenario in scenarios], [
            "wall 4 m", "wall 4.5 m", "wall 5 m", "wall 5.5 m", *stepped,
            *(f"{wall}, mortar fill on the steps at 1 in 10" for wall in stepped)])
        # Runs 8 to 10 are runs 5 to 7 with the fill and nothing else changed.
        for plain, filled in zip(overwrites[4:7], overwrites[7:10]):
            self.assertEqual({key: value for key, value in filled.items() if key != "material"}, plain)
            self.assertEqual(list(filled["material"]), ["mortar"])
        for plain, filled in zip(scenarios[4:7], scenarios[7:10]):
            without = deepcopy(filled["config"])
            del without["material"]["mortar"]
            self.assertEqual(without, plain["config"])

    def test_a_mortar_fill_is_part_of_the_scenario_only_where_the_wall_has_steps(self):
        plain, filled, uniform = rs.define_scenarios(CONFIG, [STEPPED, FILLED, {**FOUR_METRES, "material": {"mortar": MORTAR}}])
        self.assertIsNone(plain["geometry"]["mortar_fill"])
        self.assertEqual(filled["geometry"]["mortar_fill"], {"slope_run_per_rise": 0.1})
        self.assertEqual(filled["config"]["material"]["mortar"], MORTAR)
        self.assertEqual(filled["config"]["material"]["concrete"], CONFIG["material"]["concrete"])
        self.assertEqual(filled["geometry"]["steps"], plain["geometry"]["steps"])
        # material.mortar has no effect on a wall without steps: there is no fill to report.
        self.assertIsNone(uniform["geometry"]["mortar_fill"])
        steeper = rs.define_scenarios(CONFIG, [{**STEPPED, "material": {"mortar": {**MORTAR, "slope_run_per_rise": 0.25}}}])[0]
        self.assertEqual(rs.describe(steeper["geometry"]),
                         "wall 4 m, 4.5 m below z = -20 m, 5 m below z = -25 m, mortar fill on the steps at 1 in 4")

    def test_a_scenario_the_pipeline_cannot_run_is_refused_with_its_number(self):
        cases = {
            "scenario 2: .*thickness_steps.*not a supported field": [FOUR_METRES, {"geometry": {"thickness_steps": []}}],
            "scenario 1: must be an object": [[]],
            "scenario 1: .*radial_width_m": [{"geometry": {"wall": {"radial_width_m": -1}}}],
            "scenario 1: the sheet has columns for 2 thickness steps; this wall has 3": [{"geometry": {"wall": {
                **STEPPED["geometry"]["wall"],
                "thickness_steps": STEPPED["geometry"]["wall"]["thickness_steps"] + [{"top_z_m": -27.0, "radial_width_m": 5.5}]}}}],
            "scenario 1: .*elastic-layer bedrock mode": [{"bedrock_analysis": {"mode": "full_volume"}}],
            "non-empty array": [],
        }
        for message, overwrites in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(rs.SweepError, message):
                rs.define_scenarios(CONFIG, overwrites)
        with self.assertRaisesRegex(rs.SweepError, "non-empty array"):
            rs.define_scenarios(CONFIG, {"geometry": {}})
        with self.assertRaisesRegex(rs.SweepError, "^config.json: "):
            rs.define_scenarios({**CONFIG, "unknown": 1}, [FOUR_METRES])


def write_stage_outputs(directory: Path, joint_status: str = "completed", water: bool = True,
                        mortar: bool = False) -> None:
    """The files the three stages leave behind, with recognisable figures."""
    directory.mkdir(parents=True, exist_ok=True)
    rs.write_json(directory / "elmer_report.json", {
        "stress_analysis": {"max_tensile_principal_stress_pa": {"value_pa": 2.0e6},
                            "max_compressive_principal_stress_pa": {"value_pa": -3.3e6}},
        "stress_bodies": {"wall": {"maximum_displacement_m": 0.0104}}})
    baseline = directory / "aster_baseline"
    baseline.mkdir(exist_ok=True)
    rs.write_json(baseline / "run_manifest.json", {"status": "completed"})
    rs.write_json(baseline / "comparison_report.json", {"status": "requires_verification", "regions": {"wall": {
        "aster_tension_pa": {"value_pa": 1.6e6}, "aster_compression_pa": {"value_pa": -3.2e6},
        "aster_maximum_displacement_m": 0.0103}}})
    # Two unit right-angled tetrahedra: 1/6 m3 each.
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 0, 0], [4, 0, 0], [3, 2, 0], [3, 0, 3]], dtype=float)
    arrays = {"nodes": nodes, "tetra4_concrete": np.array([[0, 1, 2, 3], [4, 5, 6, 7]])}
    if mortar:
        # A fill of half the first tetrahedron's size, beside the concrete: 1/12 m3.
        arrays["nodes"] = np.vstack([nodes, [[10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 0.5]]])
        arrays["tetra4_mortar"] = np.array([[8, 9, 10, 11]])
    np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
    case = lambda opening, tension: {  # noqa: E731
        "status": "completed", "levels": [{"fill_fraction": 0.75, "largest_opening_m": 1.0},
                                          {"fill_fraction": 1.0, "largest_opening_m": opening, "opened_share": 0.2,
                                           "maximum_wall_displacement_m": 0.0113}],
        "final_wall_stress": {"maximum_principal_tensile_stress_pa": tension,
                              "maximum_principal_compressive_stress_pa": -3.4e6}}
    cases = {"dry": case(8.8e-4, 1.75e6)}
    if water:
        cases["water_in_joint"] = case(1.13e-3, 1.78e6)
    joint = directory / "joint_opening"
    joint.mkdir(exist_ok=True)
    rs.write_json(joint / "joint_opening_report.json", {
        "status": joint_status, "message": None if joint_status == "completed" else "The water in joint case ended as 'failed': refused",
        "cases": cases})


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_stage_outputs(self.directory)

    def test_baseline_figures_are_read_in_the_units_of_the_sheet(self):
        self.assertEqual(rs.elmer_figures(self.directory / "elmer_report.json"),
                         {"tension_mpa": 2.0, "compression_mpa": -3.3, "displacement_mm": 10.4})
        self.assertEqual(rs.aster_figures(self.directory / "aster_baseline"),
                         {"tension_mpa": 1.6, "compression_mpa": -3.2, "displacement_mm": 10.3})
        self.assertAlmostEqual(rs.concrete_volume_m3(self.directory / "aster_baseline"), 1 / 6 + 1.0)
        self.assertIsNone(rs.mortar_volume_m3(self.directory / "aster_baseline"))
        # The figures are left where the contact-joint stage looks for the elastic baseline.
        written = rs.read_json(self.directory / "aster_baseline" / "aster_baseline_report.json")
        self.assertEqual((written["maximum_principal_tensile_stress_pa"], written["maximum_principal_compressive_stress_pa"],
                          written["maximum_wall_displacement_m"], written["baseline_verified"], written["comparison_status"]),
                         (1.6e6, -3.2e6, 0.0103, False, "requires_verification"))

    def test_a_mortar_fill_is_measured_apart_from_the_concrete(self):
        write_stage_outputs(self.directory / "filled", mortar=True)
        baseline = self.directory / "filled" / "aster_baseline"
        self.assertAlmostEqual(rs.mortar_volume_m3(baseline), 1 / 12)
        self.assertAlmostEqual(rs.concrete_volume_m3(baseline), 1 / 6 + 1.0)

    def test_figures_that_are_missing_or_not_finite_are_an_error(self):
        baseline = self.directory / "aster_baseline"
        comparison = rs.read_json(baseline / "comparison_report.json")
        comparison["regions"]["wall"]["aster_maximum_displacement_m"] = None
        rs.write_json(baseline / "comparison_report.json", comparison)
        with self.assertRaisesRegex(ValueError, "no finite stress and displacement"):
            rs.aster_figures(baseline)
        self.assertFalse((baseline / "aster_baseline_report.json").exists())
        # A solve that did not complete is not read, whatever files it left.
        rs.write_json(baseline / "run_manifest.json", {"status": "failed"})
        with self.assertRaisesRegex(ValueError, r"did not complete \(failed\)"):
            rs.aster_figures(baseline)
        rs.write_json(self.directory / "elmer_report.json", {"stress_analysis": {}})
        with self.assertRaises(KeyError):
            rs.elmer_figures(self.directory / "elmer_report.json")

    def test_cracking_figures_are_those_at_full_reservoir(self):
        figures, failure = rs.joint_figures(self.directory / "joint_opening")
        self.assertIsNone(failure)
        self.assertEqual(list(figures), ["dry", "water"])
        self.assertAlmostEqual(figures["dry"]["opening_mm"], 0.88)
        self.assertAlmostEqual(figures["water"]["opening_mm"], 1.13)
        self.assertAlmostEqual(figures["water"]["tension_mpa"], 1.78)
        self.assertAlmostEqual(figures["dry"]["compression_mpa"], -3.4)
        self.assertAlmostEqual(figures["dry"]["displacement_mm"], 11.3)

    def test_a_case_that_did_not_complete_leaves_the_other_and_says_why(self):
        write_stage_outputs(self.directory, joint_status="failed", water=False)
        figures, failure = rs.joint_figures(self.directory / "joint_opening")
        self.assertEqual(list(figures), ["dry"])
        self.assertIn("water in joint case ended as 'failed'", failure)
        # A case that stopped before full reservoir is not reported, whatever its status says.
        report = rs.read_json(self.directory / "joint_opening" / "joint_opening_report.json")
        report["cases"]["dry"]["levels"].pop()
        rs.write_json(self.directory / "joint_opening" / "joint_opening_report.json", report)
        self.assertEqual(rs.joint_figures(self.directory / "joint_opening")[0], {})


class FakeStages:
    """Stands in for the solver processes: records each call and writes that stage's files."""

    def __init__(self, fail: dict | None = None, joint_status: str = "completed", water: bool = True,
                 wrong_fill: bool = False):
        self.calls, self.fail, self.joint_status, self.water = [], fail or {}, joint_status, water
        self.wrong_fill = wrong_fill

    def __call__(self, module, arguments, log, allowed_codes=(0,)):
        directory = log.parent
        self.calls.append((directory.name, module, list(arguments), allowed_codes))
        log.write_text(module, encoding="utf-8")
        if self.fail.get(directory.name) == module:
            raise RuntimeError(f"{module} exited with code 1; see {log}")
        scratch = directory / "_all"
        # The mesh has a fill exactly when the scenario's own configuration asks for one.
        fill = rs.sheet_geometry(rs.read_json(directory / "config.json"))["mortar_fill"] is not None
        write_stage_outputs(scratch, self.joint_status, self.water, mortar=fill != self.wrong_fill)
        if module == "Elmer.three_body_solver":
            (scratch / "elmer_report.json").replace(directory / "elmer_report.json")
        elif module == "Aster.run_three_body_pipeline":
            (scratch / "aster_baseline").replace(directory / "aster_baseline")
        else:
            (scratch / "joint_opening").replace(directory / "joint_opening")
        shutil.rmtree(scratch)
        return 0


class RunTests(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.scenarios = rs.define_scenarios(CONFIG, [FOUR_METRES, STEPPED])
        self.enterContext(mock.patch("builtins.print"))

    def test_a_scenario_runs_the_three_stages_of_an_app_job_on_its_own_config(self):
        stages = FakeStages()
        result = rs.run_scenario(self.run_dir, self.scenarios[1], stages)
        directory = self.run_dir / "scenario_02"
        self.assertEqual(rs.read_json(directory / "config.json"), self.scenarios[1]["config"])
        self.assertEqual([(name, module, codes) for name, module, _, codes in stages.calls], [
            ("scenario_02", "Elmer.three_body_solver", (0,)),
            ("scenario_02", "Aster.run_three_body_pipeline", (0, 2)),
            ("scenario_02", "Aster.dam_joint_baseline", (0, 2))])
        elmer, baseline, joint = (call[2] for call in stages.calls)
        self.assertEqual(elmer, ["--config-root", str(directory), "--output-dir", str(directory / "elmer"),
                                 "--report-path", str(directory / "elmer_report.json")])
        self.assertEqual(baseline[:4], ["--source-mesh", str(directory / "elmer" / "model" / "three_body.msh"),
                                        "--output-dir", str(directory / "aster_baseline")])
        self.assertEqual(joint[:7], ["app", "--baseline-dir", str(directory / "aster_baseline"),
                                     "--output-dir", str(directory / "joint_opening"), "--time-limit-s", "7200"])
        self.assertEqual((result["status"], result["failed_stage"], result["error"]), ("completed", None, None))
        self.assertEqual(result["elmer"]["tension_mpa"], 2.0)
        self.assertEqual(result["aster"]["displacement_mm"], 10.3)
        self.assertAlmostEqual(result["concrete_volume_m3"], 1 / 6 + 1.0)
        self.assertIsNone(result["mortar_volume_m3"])
        self.assertAlmostEqual(result["cracking"]["water"]["opening_mm"], 1.13)
        self.assertEqual(set(result["seconds"]), {"elmer", "aster_baseline", "joint_opening"})
        self.assertEqual(rs.read_json(directory / "result.json"), json.loads(json.dumps(result)))

    def test_a_scenario_with_a_mortar_fill_reports_its_volume_apart_from_the_concrete(self):
        filled = rs.define_scenarios(CONFIG, [FILLED])[0]
        result = rs.run_scenario(self.run_dir, filled, FakeStages())
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["mortar_volume_m3"], 1 / 12)
        self.assertAlmostEqual(result["concrete_volume_m3"], 1 / 6 + 1.0)
        self.assertEqual(rs.read_json(self.run_dir / "scenario_01" / "config.json")["material"]["mortar"], MORTAR)

    def test_a_mesh_that_disagrees_with_the_scenario_about_the_fill_fails_the_baseline_stage(self):
        for name, scenario in (("fill_missing", rs.define_scenarios(CONFIG, [FILLED])[0]), ("fill_unasked", self.scenarios[1])):
            run = self.run_dir / name
            run.mkdir()
            result = rs.run_scenario(run, scenario, FakeStages(wrong_fill=True))
            self.assertEqual((result["status"], result["failed_stage"]), ("failed", "aster_baseline"), name)
            self.assertIn("disagree on whether the wall has a mortar fill", result["error"])
            self.assertEqual(result["cracking"], {})

    def test_a_failed_stage_ends_the_scenario_and_keeps_what_was_solved(self):
        stages = FakeStages(fail={"scenario_01": "Aster.run_three_body_pipeline"})
        result = rs.run_scenario(self.run_dir, self.scenarios[0], stages)
        self.assertEqual([call[1] for call in stages.calls], ["Elmer.three_body_solver", "Aster.run_three_body_pipeline"])
        self.assertEqual((result["status"], result["failed_stage"]), ("failed", "aster_baseline"))
        self.assertIn("RuntimeError: Aster.run_three_body_pipeline exited with code 1", result["error"])
        self.assertEqual(result["elmer"]["tension_mpa"], 2.0)
        self.assertEqual((result["aster"], result["concrete_volume_m3"], result["cracking"]), (None, None, {}))

    def test_a_joint_case_that_did_not_complete_fails_the_scenario_but_keeps_the_dry_figures(self):
        result = rs.run_scenario(self.run_dir, self.scenarios[0], FakeStages(joint_status="failed", water=False))
        self.assertEqual((result["status"], result["failed_stage"]), ("failed", "joint_opening"))
        self.assertIn("water in joint case ended as 'failed'", result["error"])
        self.assertEqual(list(result["cracking"]), ["dry"])
        self.assertEqual(result["aster"]["tension_mpa"], 1.6)

    def test_the_sweep_goes_on_after_a_failed_scenario_and_saves_after_each(self):
        stages = FakeStages(fail={"scenario_01": "Elmer.three_body_solver"})
        state = rs.execute(self.run_dir, self.scenarios, None, stages)
        self.assertEqual((state["status"], state["failed_scenarios"]), ("completed_with_failures", [1]))
        self.assertEqual([result["status"] for result in state["results"]], ["failed", "completed"])
        self.assertEqual(rs.read_json(self.run_dir / "results.json"), json.loads(json.dumps(state)))
        rows = (self.run_dir / "results.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 3)
        self.assertIn("cracking_water_opening_mm", rows[0])
        self.assertTrue(rows[2].startswith('2,"wall 4 m, 4.5 m below z = -20 m, 5 m below z = -25 m",completed'))
        log = (self.run_dir / "progress.log").read_text(encoding="utf-8")
        self.assertIn("Scenario 1: Elmer baseline FAILED", log)
        self.assertIn("Scenario 2: Code_Aster contact joint (dry, then water in the joint) finished", log)

    def test_an_interrupted_sweep_keeps_the_finished_scenarios(self):
        stages = FakeStages()

        def interrupted(module, arguments, log, allowed_codes=(0,)):
            if log.parent.name == "scenario_02":
                raise KeyboardInterrupt
            return stages(module, arguments, log, allowed_codes)

        with self.assertRaises(KeyboardInterrupt):
            rs.execute(self.run_dir, self.scenarios, None, interrupted)
        state = rs.read_json(self.run_dir / "results.json")
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual([result["number"] for result in state["results"]], [1])


def make_workbook(path: Path, rows: int = 2) -> None:
    """A workbook with the header layout of FEA_Matrix.xlsx and ``rows`` rows of earlier figures."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = rs.SHEET_NAME
    for cell, text in rs.EXPECTED_HEADERS.items():
        sheet[cell] = text
    for row in range(rs.FIRST_ROW, rs.FIRST_ROW + rows):
        for column in range(1, rs.LAST_COLUMN + 1):
            sheet.cell(row, column).value = 99
        sheet.cell(row, 1).value = row - rs.FIRST_ROW + 1
    sheet["J3"].number_format = "0.00"
    sheet["U12"] = "Cracking columns: an earlier note."
    sheet["A20"] = "kept"
    workbook.save(path)


class SheetTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.sheet = self.directory / "FEA_Matrix.xlsx"
        self.run_dir = self.directory / "20260101T000000Z-test"
        self.run_dir.mkdir()
        self.enterContext(mock.patch("builtins.print"))
        self.scenarios = rs.define_scenarios(CONFIG, [FOUR_METRES, STEPPED, FOUR_METRES])

    def results(self, **stages) -> list[dict]:
        return rs.execute(self.run_dir, self.scenarios, None, FakeStages(**stages))["results"]

    def cells(self, *names):
        sheet = openpyxl.load_workbook(self.sheet)[rs.SHEET_NAME]
        return [sheet[name].value for name in names]

    def test_each_run_is_written_to_its_row(self):
        make_workbook(self.sheet)
        outcome = rs.update_sheet(self.sheet, self.results(), self.run_dir)
        self.assertEqual(outcome["rows_written"], [1, 2, 3])
        self.assertEqual(self.cells("A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3", "I3"),
                         [1, 4.0, 1.0, 0.5, "-", "-", "-", "-", 80.0])
        self.assertEqual(self.cells("A4", "B4", "E4", "F4", "G4", "H4"), [2, 4.0, 4.5, -20.0, 5.0, -25.0])
        self.assertEqual(self.cells("J3", "K3", "L3", "M3", "N3", "O3", "P3", "Q3", "R3"),
                         [2.0, 1.6, "=AVERAGE(J3:K3)", -3.3, -3.2, "=AVERAGE(M3:N3)", 10.4, 10.3, "=AVERAGE(P3:Q3)"])
        self.assertEqual(self.cells("S3", "T3", "S5", "T5"), [1, "=S3-$S$3", 1, "=S5-$S$3"])
        self.assertEqual(self.cells("U4", "V4", "W4", "X4", "Y4", "Z4", "AA4", "AB4"),
                         [0.88, 1.13, 1.75, 1.78, -3.4, -3.4, 11.3, 11.3])
        # A new row takes the formatting of the first row; cells outside the table are kept.
        sheet = openpyxl.load_workbook(self.sheet)[rs.SHEET_NAME]
        self.assertEqual(sheet["J5"].number_format, "0.00")
        self.assertEqual(sheet["A20"].value, "kept")
        # The earlier note is replaced by one below the table that names the sweep.
        self.assertIsNone(sheet["U12"].value)
        self.assertEqual(outcome["note_row"], 22)
        self.assertIn("Runs 1, 2, 3 written by cracking_sweeps.run_sweep, sweep 20260101T000000Z-test.",
                      sheet[f"U{outcome['note_row']}"].value)
        self.assertIn("not an engineering assessment", sheet[f"U{outcome['note_row']}"].value)

    def test_the_mortar_fill_volume_has_its_own_column(self):
        make_workbook(self.sheet)
        self.assertEqual(self.cells("AC1", "AC2"), [None, None])
        scenarios = rs.define_scenarios(CONFIG, [FOUR_METRES, STEPPED, FILLED])
        results = rs.execute(self.run_dir, scenarios, None, FakeStages())["results"]
        outcome = rs.update_sheet(self.sheet, results, self.run_dir)
        # The column is added to a workbook that does not have it; "-" marks a wall without a fill.
        self.assertEqual(self.cells("AC1", "AC2", "AC3", "AC4", "AC5"), ["Mortar fill", "Volume (m^3)", "-", "-", 0.1])
        # The fill is not concrete: the Volume column is the same with and without it.
        self.assertEqual(self.cells("S4", "S5", "B5", "E5", "G5"), [1, 1, 4.0, 4.5, 5.0])
        sheet = openpyxl.load_workbook(self.sheet)[rs.SHEET_NAME]
        self.assertIn("Mortar fill = soft fill sloping the wall's steps", sheet[f"U{outcome['note_row']}"].value)
        self.assertIn("mortar_volume_m3", (self.run_dir / "results.csv").read_text(encoding="utf-8").splitlines()[0])
        # A heading already in the workbook is left as it is.
        workbook = openpyxl.load_workbook(self.sheet)
        workbook[rs.SHEET_NAME]["AC1"] = "Fill"
        workbook.save(self.sheet)
        rs.update_sheet(self.sheet, results, self.run_dir)
        self.assertEqual(self.cells("AC1", "AC2"), ["Fill", "Volume (m^3)"])

    def test_a_fill_whose_run_stopped_early_has_an_empty_cell_and_earlier_results_still_write(self):
        make_workbook(self.sheet)
        scenarios = rs.define_scenarios(CONFIG, [FILLED])
        results = rs.execute(self.run_dir, scenarios, None, FakeStages(fail={"scenario_01": "Elmer.three_body_solver"}))["results"]
        rs.update_sheet(self.sheet, results, self.run_dir)
        self.assertEqual(self.cells("A3", "AC3", "J3"), [1, None, None])
        # Results saved before the fill existed have no such entries; their walls had no fill.
        earlier = self.directory / "earlier"
        earlier.mkdir()
        old = rs.execute(earlier, rs.define_scenarios(CONFIG, [FOUR_METRES]), None, FakeStages())["results"]
        for result in old:
            del result["mortar_volume_m3"], result["geometry"]["mortar_fill"]
        outcome = rs.update_sheet(self.sheet, old, earlier)
        self.assertEqual(self.cells("AC3", "J3"), ["-", 2.0])
        sheet = openpyxl.load_workbook(self.sheet)[rs.SHEET_NAME]
        self.assertNotIn("Mortar fill", sheet[f"U{outcome['note_row']}"].value)

    def test_the_workbook_as_it_was_is_kept_in_the_sweep_directory(self):
        make_workbook(self.sheet)
        before = self.sheet.read_bytes()
        outcome = rs.update_sheet(self.sheet, self.results(), self.run_dir)
        self.assertEqual(Path(outcome["backup"]), self.run_dir / "FEA_Matrix.before.xlsx")
        self.assertEqual(Path(outcome["backup"]).read_bytes(), before)
        rs.update_sheet(self.sheet, self.results_again(), self.run_dir)
        self.assertEqual(Path(outcome["backup"]).read_bytes(), before)  # a second write does not replace it

    def results_again(self) -> list[dict]:
        return rs.read_json(self.run_dir / "results.json")["results"]

    def test_a_run_that_failed_leaves_no_earlier_figure_in_its_row(self):
        make_workbook(self.sheet, rows=3)
        results = self.results(fail={"scenario_01": "Aster.run_three_body_pipeline"}, joint_status="failed", water=False)
        outcome = rs.update_sheet(self.sheet, results, self.run_dir)
        # Run 1 stopped at the Code_Aster baseline: its Elmer figures only.
        self.assertEqual(self.cells("A3", "B3", "J3", "K3", "L3", "S3", "T3", "U3", "AB3"),
                         [1, 4.0, 2.0, None, None, None, None, None, None])
        # Run 2 lost its water-in-joint case: baselines and the dry case.
        self.assertEqual(self.cells("K4", "L4", "S4", "U4", "V4", "W4", "X4"),
                         [1.6, "=AVERAGE(J4:K4)", 1, 0.88, None, 1.75, None])
        sheet = openpyxl.load_workbook(self.sheet)[rs.SHEET_NAME]
        self.assertIn("runs 1, 2, 3 did not complete, their empty cells have no result",
                      sheet[f"U{outcome['note_row']}"].value)

    def test_only_the_rows_of_the_runs_given_are_touched(self):
        make_workbook(self.sheet, rows=3)
        results = [result for result in self.results() if result["number"] == 2]
        rs.update_sheet(self.sheet, results, self.run_dir)
        self.assertEqual(self.cells("B3", "J3", "U3", "B5", "AB5"), [99, 99, 99, 99, 99])
        self.assertEqual(self.cells("A4", "B4", "E4", "J4"), [2, 4.0, 4.5, 2.0])

    def test_a_workbook_with_another_layout_is_refused_and_left_alone(self):
        make_workbook(self.sheet)
        workbook = openpyxl.load_workbook(self.sheet)
        workbook[rs.SHEET_NAME]["K2"] = "Something else"
        workbook.save(self.sheet)
        before = self.sheet.read_bytes()
        with self.assertRaisesRegex(rs.SweepError, "K2 is 'Something else', expected 'Aster'"):
            rs.update_sheet(self.sheet, self.results(), self.run_dir)
        with self.assertRaisesRegex(rs.SweepError, "K2 is 'Something else'"):
            rs.check_sheet(self.sheet)
        self.assertEqual(self.sheet.read_bytes(), before)
        workbook[rs.SHEET_NAME].title = "Other"
        workbook.save(self.sheet)
        with self.assertRaisesRegex(rs.SweepError, "no sheet named 'Sweep'"):
            rs.check_sheet(self.sheet)
        with self.assertRaisesRegex(rs.SweepError, "does not exist"):
            rs.check_sheet(self.directory / "missing.xlsx")

    def test_the_sweep_writes_the_sheet_when_it_ends_and_reports_a_sheet_it_cannot_write(self):
        make_workbook(self.sheet)
        state = rs.execute(self.run_dir, self.scenarios[:1], self.sheet, FakeStages())
        self.assertEqual(state["sheet"]["rows_written"], [1])
        self.assertEqual(self.cells("J3", "U3"), [2.0, 0.88])
        other = self.directory / "second"
        other.mkdir()
        state = rs.execute(other, self.scenarios[:1], self.directory / "missing.xlsx", FakeStages())
        self.assertEqual(state["status"], "completed")
        self.assertIn("does not exist", state["sheet"]["error"])
        self.assertIn("--update-sheet", (other / "progress.log").read_text(encoding="utf-8"))

    @unittest.skipUnless((rs.HERE / "FEA_Matrix.xlsx").is_file(), "the project workbook is not present")
    def test_the_project_workbook_has_the_layout_the_sweep_writes(self):
        rs._load_sheet(rs.HERE / "FEA_Matrix.xlsx")


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.printed = self.enterContext(mock.patch("builtins.print"))
        self.sheet = self.directory / "FEA_Matrix.xlsx"
        make_workbook(self.sheet)
        self.scenarios = self.directory / "scenarios.json"
        rs.write_json(self.scenarios, [FOUR_METRES, STEPPED])
        self.common = ["--scenarios", str(self.scenarios), "--sheet", str(self.sheet),
                       "--output-root", str(self.directory / "runs")]

    def output(self) -> str:
        return "\n".join(str(call.args[0]) for call in self.printed.call_args_list if call.args)

    def test_without_execute_only_the_plan_is_written(self):
        with mock.patch.object(rs, "run_module") as runner:
            self.assertEqual(rs.main(self.common), 0)
        runner.assert_not_called()
        (run,) = (self.directory / "runs").iterdir()
        plan = rs.read_json(run / "plan.json")
        self.assertEqual([scenario["number"] for scenario in plan["scenarios"]], [1, 2])
        self.assertEqual(plan["scenarios"][1]["config"]["geometry"]["wall"]["thickness_steps"],
                         STEPPED["geometry"]["wall"]["thickness_steps"])
        self.assertEqual(rs.read_json(run / "scenarios.json"), [FOUR_METRES, STEPPED])
        self.assertIn("Plan only: no solver was started", self.output())
        self.assertFalse((run / "results.json").exists())

    def test_execute_runs_the_chosen_scenarios_and_updates_the_sheet(self):
        stages = FakeStages()
        with mock.patch.object(rs, "run_module", stages):
            self.assertEqual(rs.main([*self.common, "--execute", "--only", "2"]), 0)
        self.assertEqual({call[0] for call in stages.calls}, {"scenario_02"})
        sheet = openpyxl.load_workbook(self.sheet)[rs.SHEET_NAME]
        self.assertEqual((sheet["A4"].value, sheet["E4"].value, sheet["U4"].value, sheet["J3"].value), (2, 4.5, 0.88, 99))
        self.assertIn("2: wall 4 m, 4.5 m below z = -20 m, 5 m below z = -25 m: completed; opening dry / water 0.880 / 1.130 mm",
                      self.output())

    def test_a_failed_scenario_gives_exit_code_2(self):
        stages = FakeStages(fail={"scenario_01": "Elmer.three_body_solver"})
        with mock.patch.object(rs, "run_module", stages):
            self.assertEqual(rs.main([*self.common, "--execute"]), 2)
        self.assertIn("1: wall 4 m: failed at elmer; opening dry / water - / - mm", self.output())

    def test_nothing_is_started_when_the_sweep_or_the_sheet_is_not_usable(self):
        with mock.patch.object(rs, "run_module") as runner:
            self.assertEqual(rs.main([*self.common, "--execute", "--only", "3"]), 1)
            rs.write_json(self.scenarios, [{"geometry": {"thickness_steps": []}}])
            self.assertEqual(rs.main([*self.common, "--execute"]), 1)
            rs.write_json(self.scenarios, [FOUR_METRES])
            self.assertEqual(rs.main([*self.common[:2], "--sheet", str(self.directory / "missing.xlsx"),
                                      *self.common[4:], "--execute"]), 1)
        runner.assert_not_called()
        self.assertFalse((self.directory / "runs").exists())

    def test_a_finished_sweep_can_be_written_to_the_sheet_later(self):
        run = self.directory / "runs" / "20260101T000000Z-later"
        run.mkdir(parents=True)
        scenarios = rs.define_scenarios(CONFIG, [FOUR_METRES])
        rs.execute(run, scenarios, None, FakeStages())
        with mock.patch.object(rs, "run_module") as runner:
            self.assertEqual(rs.main(["--sheet", str(self.sheet), "--update-sheet", str(run)]), 0)
        runner.assert_not_called()
        sheet = openpyxl.load_workbook(self.sheet)[rs.SHEET_NAME]
        self.assertEqual((sheet["J3"].value, sheet["V3"].value), (2.0, 1.13))


if __name__ == "__main__":
    unittest.main()
