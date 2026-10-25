"""Check nonlinear benchmark acceptance gates without requiring native Aster."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from Aster.damage_benchmark import CASE, SETTINGS, run, verify, visualisation_warnings
from Elmer.model_config import load_config


def synthetic_history():
    histories = {}
    for label in ("INTACT", "SEEDED", "LOCAL", "FREE", "NEAR", "NEARLOC"):
        rows = []
        for time in SETTINGS["times"]:
            if time < 1:
                damage = .2 * time
                values = [damage, damage]
                if label == "SEEDED":
                    values = [max(.1, damage), max(.1, damage)]
                if label == "LOCAL":
                    values = [max(.1, damage), damage]
                stress = 1000 * time
            else:
                damage = .2 if time <= 2 else .2 + .01 * (time - 2)
                values = [damage, damage]
                stress = 1000 * (1 - .5 * (time - 1)) if time <= 2 else 500 + 200 * (time - 2)
            if label == "NEAR":
                values = [SETTINGS["near_complete_damage"]] * 2
            elif label == "NEARLOC":
                values[0] = SETTINGS["near_complete_damage"]
            rows.append({"time": time, "damage_values": values, "stress_zz_mean_pa": stress,
                         "maximum_lateral_stress_pa": 0.,
                         "damage_point_ids": [[0, 0, 0], [0, 1, 0]]})
        histories[label] = rows
    return {"expected_target_stress_zz_pa": 1000., "histories": histories}


class DamageBenchmarkTests(unittest.TestCase):
    def test_nodal_recovery_overshoot_is_reported_without_clipping_or_failing_physics(self):
        series = {"NEARLOC": {
            "damage_field": "damage_recovered_nodal", "damage_association": "nodes",
            "damage_recovery": "native_VARI_NOEU_unclipped",
            "snapshots": [{"damage_min": -.014, "damage_max": .999}],
        }}
        warnings = visualisation_warnings(series)
        self.assertEqual(len(warnings), 1)
        self.assertIn("NEARLOC", warnings[0])
        self.assertIn("not clipped", warnings[0])
        self.assertIn("native_seed.json", visualisation_warnings(series, "native_seed.json")[0])
        self.assertEqual(series["NEARLOC"]["snapshots"][0]["damage_min"], -.014)
        series["NEARLOC"]["snapshots"][0]["damage_min"] = 0.
        self.assertEqual(visualisation_warnings(series), [])

    def test_invalid_nodal_recovery_metadata_and_ranges_are_rejected(self):
        series = {"NEAR": {
            "damage_field": "damage_recovered_nodal", "damage_association": "nodes",
            "damage_recovery": "native_VARI_NOEU_unclipped",
            "snapshots": [{"damage_min": 0., "damage_max": .999}],
        }}
        for change in (
            lambda s: s["NEAR"].update(damage_association="integration_points"),
            lambda s: s["NEAR"].update(damage_recovery="clipped"),
            lambda s: s["NEAR"].update(snapshots=[]),
            lambda s: s["NEAR"]["snapshots"][0].update(damage_min=float("nan")),
            lambda s: s["NEAR"]["snapshots"][0].update(damage_min=1.),
        ):
            corrupted = copy.deepcopy(series)
            change(corrupted)
            with self.subTest(change=change), self.assertRaises(ValueError):
                visualisation_warnings(corrupted)

    def test_template_is_valid_python_and_uses_native_nonlinear_model(self):
        text = CASE.read_text()
        compile(text, str(CASE), "exec")
        self.assertIn('MODELISATION="3D_GRAD_VARI"', text)
        self.assertIn('RELATION="ENDO_FISS_EXP"', text)
        self.assertIn('LINE_QUAD=', text)
        self.assertIn('GROUP_MA="SEED"', text)

    def test_complete_nonlinear_histories_pass(self):
        checks = verify(synthetic_history())
        self.assertEqual(set(checks), {"INTACT", "SEEDED", "LOCAL", "FREE", "NEAR", "NEARLOC"})
        self.assertEqual(checks["SEEDED"]["increments"], 30)
        self.assertEqual(checks["INTACT"]["target_stress_relative_error"], 0)
        self.assertIsNone(checks["LOCAL"]["target_stress_relative_error"])

    def test_missing_increments_damage_healing_wrong_target_and_unloading_are_rejected(self):
        baseline = synthetic_history()
        changes = (
            lambda r: r["histories"]["INTACT"].pop(),
            lambda r: r["histories"]["LOCAL"][0].update(damage_values=[.1, .1]),
            lambda r: r["histories"]["SEEDED"][1].update(damage_values=[0., 0.]),
            lambda r: r["histories"]["INTACT"][10].update(damage_values=[.3, .3]),
            lambda r: r["histories"]["INTACT"][10].update(stress_zz_mean_pa=800),
            lambda r: r["histories"]["INTACT"][20].update(stress_zz_mean_pa=900),
            lambda r: r["histories"]["INTACT"][25].update(damage_values=[float("nan"), .2]),
            lambda r: r.update(expected_target_stress_zz_pa=float("nan")),
            lambda r: r["histories"]["FREE"][10].update(maximum_lateral_stress_pa=1000),
            lambda r: r["histories"]["FREE"][20].update(stress_zz_mean_pa=1100),
            lambda r: r["histories"]["LOCAL"][5].update(damage_point_ids=[[0, 1, 0], [0, 0, 0]]),
            lambda r: r["histories"]["NEAR"][0].update(damage_values=[0., 0.]),
            lambda r: r["histories"]["NEAR"][20].update(damage_values=[.99, .99]),
            lambda r: r["histories"]["NEARLOC"][0].update(damage_values=[.999, .999]),
        )
        for index, alter in enumerate(changes):
            report = copy.deepcopy(baseline)
            alter(report)
            with self.subTest(index=index), self.assertRaises(ValueError):
                verify(report)

    def test_already_damaged_without_further_growth_is_rejected(self):
        report = synthetic_history()
        for row in report["histories"]["LOCAL"][21:]:
            row["damage_values"] = [.2, .2]
        with self.assertRaisesRegex(ValueError, "did not grow"):
            verify(report)

    def test_nonfinite_stress_and_unphysical_damage_are_rejected(self):
        for values in ([1.1, .2], [-.1, .2]):
            report = synthetic_history()
            report["histories"]["INTACT"][25]["damage_values"] = values
            with self.assertRaisesRegex(ValueError, "outside"):
                verify(report)
        report = synthetic_history()
        report["histories"]["LOCAL"][25]["stress_zz_mean_pa"] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite stress"):
            verify(report)

    def test_native_exception_even_with_zero_exit_never_reports_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(load_config()))
            with patch("Aster.damage_benchmark.detect_installation", return_value={
                "available": True, "launcher_path": "fake",
            }), patch("Aster.damage_benchmark.run_solver", return_value=
                      subprocess.CompletedProcess([], 0, "<EXCEPTION> failure", "")):
                with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
                    run(config_path, root)
            reports = list(root.glob("damage-*/benchmark_report.json"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(json.loads(reports[0].read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
