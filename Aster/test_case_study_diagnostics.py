import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from Aster import case_study_diagnostics as diag


class DiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.study = Path(self.temporary.name)
        self.run = self.study / "cracking" / "aster_crack_submodel" / "saved"
        self.run.mkdir(parents=True)
        self.output = self.study / "diagnostics"
        (self.run / "mesh.med").write_bytes(b"frozen mesh")
        (self.run / "quadratic_discrete.npz").write_bytes(b"frozen operator")
        (self.run / "config.json").write_text("{}", encoding="utf-8")
        diag.write(self.run / "inputs.json", {
            "repository_root": "old root", "loads": {"crack_pressure_pa": 0.0},
            "times": [0.0, 1.0], "shared_seed": {"sha256": "frozen"},
            "interface_operator": diag.wm.INTERFACE_OPERATOR,
        })
        diag.write(self.run / "crack_report.json", {"region_stress_history": [{
            "time": 1.0, "wall": {"maximum_principal_stress_pa": 15e6,
                                "minimum_principal_stress_pa": -51e6},
        }]})
        diag.write(self.run / "manifest.json", {
            "status": "completed_not_engineering_validated",
            "configuration_sha256": diag.digest(self.run / "config.json"),
            "case_sha256": diag.digest(diag.wm.CASE),
            "quadratic_discrete_sha256": diag.digest(self.run / "quadratic_discrete.npz"),
        })
        baseline = self.study / "cracking" / "aster_elastic_baseline"
        baseline.mkdir()
        diag.write(baseline / "aster_baseline_report.json", {
            "maximum_principal_tensile_stress_pa": 1e6,
            "maximum_principal_compressive_stress_pa": -3e6,
        })

    def test_templates_compile_and_isolate_the_intended_change(self):
        for stage in ("elastic", "unseeded", "seeded"):
            text = diag.render_case(stage)
            compile(text, stage + ".comm", "exec")
            self.assertIn('LIAISON_MAIL', text)
            self.assertIn('REAC_PRECOND=1, RESI_RELA=1e-8', text)
            self.assertIn('tensile_peak_xyz_m', text)
            self.assertEqual('ETAT_INIT=_F(VARI=INITIAL)' in text, stage == "seeded")
            self.assertEqual('INITIAL, audit = initial_field(' in text, stage == "seeded")
            self.assertEqual('MODELISATION="3D_GRAD_VARI"' in text, stage != "elastic")
            self.assertEqual('RELATION="ENDO_FISS_EXP"' in text, stage != "elastic")
            self.assertEqual('_F(GROUP_MA="DAMAGE_BAND", MATER=DAMAGE)' in text, stage != "elastic")
        with self.assertRaises(ValueError):
            diag.replace_once("changed template", "expected", "")

    def test_legacy_replay_is_explicit_and_never_uses_quadratic_characteristics(self):
        text = diag.render_case("elastic", legacy_lumped=True)
        compile(text, "legacy.comm", "exec")
        self.assertIn('CARA="K_T_D_L"', text)
        self.assertNotIn('springs = characteristics(', text)
        self.assertNotIn('MESH = read_mesh(', text)
        inputs = diag.read(self.run / "inputs.json")
        inputs.pop("interface_operator")
        diag.write(self.run / "inputs.json", inputs)
        directory = diag.prepare(self.study, self.output, "elastic", 570, 8, 20480,
                                 legacy_lumped=True)
        self.assertNotIn("D 92", (directory / "diagnostic.export").read_text())
        self.assertEqual(diag.read(directory / "manifest.json")["interface_operator"], "legacy_lumped")

    def test_legacy_archive_rebuilds_corrected_mesh_by_default(self):
        inputs = diag.read(self.run / "inputs.json")
        inputs.pop("interface_operator")
        diag.write(self.run / "inputs.json", inputs)
        def rebuilt(run, directory):
            self.assertEqual(run, self.run)
            (directory / "mesh.med").write_bytes(b"corrected mesh")
            (directory / "quadratic_discrete.npz").write_bytes(b"corrected operator")
            return {"interface_integration": {"law": diag.wm.INTERFACE_OPERATOR}}
        with mock.patch.object(diag, "corrected_mesh", side_effect=rebuilt) as rebuild:
            directory = diag.prepare(self.study, self.output, "elastic", 570, 8, 20480)
        rebuild.assert_called_once()
        self.assertIn(diag.wm.wsl_path(directory / "mesh.med"),
                      (directory / "diagnostic.export").read_text())
        self.assertEqual(diag.read(directory / "inputs.json")["interface_operator"],
                         diag.wm.INTERFACE_OPERATOR)

    def test_corrupt_corrected_operator_and_legacy_downgrade_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be replayed"):
            diag.prepare(self.study, self.output, "elastic", 570, 8, 20480, legacy_lumped=True)
        (self.run / "quadratic_discrete.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            diag.prepare(self.study, self.output, "elastic", 570, 8, 20480)

    def test_preparation_uses_saved_mesh_and_inputs_without_changing_sources(self):
        before = {p.name: diag.digest(p) for p in self.run.iterdir()}
        directory = diag.prepare(self.study, self.output, "elastic", 570, 8, 20480)
        inputs = diag.read(directory / "inputs.json")
        self.assertEqual(inputs["shared_seed"], {"sha256": "frozen"})
        self.assertEqual(inputs["times"], [0.0, 1.0])
        self.assertIn(diag.wm.wsl_path(self.run / "mesh.med"),
                      (directory / "diagnostic.export").read_text())
        self.assertEqual(before, {p.name: diag.digest(p) for p in self.run.iterdir()})
        self.assertEqual(diag.read(directory / "manifest.json")["status"], "prepared")

    def test_ambiguity_and_changed_configuration_are_rejected(self):
        (self.run / "config.json").write_text('{"changed":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            diag.source_run(self.study)

    def test_no_seed_reference_rejects_seed_face_pressure(self):
        inputs = diag.read(self.run / "inputs.json")
        inputs["loads"]["crack_pressure_pa"] = 100.0
        diag.write(self.run / "inputs.json", inputs)
        with self.assertRaisesRegex(ValueError, "zero archived"):
            diag.prepare(self.study, self.output, "unseeded", 570, 8, 20480)

    def summary(self, stage="elastic"):
        return {"stage": stage, "history": [
            {"time": t, "regions": {"wall": {
                "maximum_principal_stress_pa": t * 1.5e6,
                "minimum_principal_stress_pa": t * -3e6,
            }}} for t in (0.0, 1.0)
        ]}

    def test_validation_does_not_accept_missing_steps_nonfinite_or_empty_med(self):
        directory = diag.prepare(self.study, self.output, "elastic", 570, 8, 20480)
        with self.assertRaisesRegex(ValueError, "no summary.json.*solver.live.log"):
            diag.validate_summary(directory, [0.0, 1.0], "elastic")
        diag.write(directory / "summary.json", self.summary())
        with self.assertRaisesRegex(ValueError, "MED"):
            diag.validate_summary(directory, [0.0, 1.0], "elastic")
        (directory / "result.med").write_bytes(b"result")
        diag.validate_summary(directory, [0.0, 1.0], "elastic")
        with self.assertRaisesRegex(ValueError, "every archived"):
            diag.validate_summary(directory, [0.0, 0.5, 1.0], "elastic")
        data = self.summary()
        data["history"][-1]["regions"]["wall"]["maximum_principal_stress_pa"] = None
        diag.write(directory / "summary.json", data)
        with self.assertRaisesRegex(ValueError, "invalid stress"):
            diag.validate_summary(directory, [0.0, 1.0], "elastic")

    def test_failed_solver_is_recorded_and_lock_released(self):
        directory = diag.prepare(self.study, self.output, "elastic", 570, 8, 20480)
        with mock.patch.object(diag.wm, "find_linux_installation", return_value="/opt/aster"), \
                mock.patch.object(diag, "run_solver", side_effect=subprocess.TimeoutExpired("solver", 570)), \
                mock.patch.object(diag.subprocess, "run") as cleanup:
            with self.assertRaises(subprocess.TimeoutExpired):
                diag.solve(directory)
        self.assertFalse((self.output / "solver.lock").exists())
        self.assertEqual(diag.read(directory / "manifest.json")["status"], "failed")
        self.assertEqual(cleanup.call_count, 1)

    def test_solver_waits_for_isolated_group_before_cleanup_and_validation(self):
        directory = diag.prepare(self.study, self.output, "elastic", 570, 8, 20480)
        events = []

        def run(command, run_dir, timeout, *, live_log):
            start = command.index("/usr/bin/setsid")
            self.assertEqual(command[start:start + 5],
                             ["/usr/bin/setsid", "--fork", "--wait", "/bin/sh", "-c"])
            self.assertEqual(run_dir, directory)
            self.assertEqual(timeout, 600)
            self.assertEqual(live_log, directory / "solver.live.log")
            diag.write(directory / "summary.json", self.summary())
            (directory / "result.med").write_bytes(b"result")
            events.append("solver_finished")
            return subprocess.CompletedProcess(command, 0, "finished\n", "")

        def cleanup(*args, **kwargs):
            events.append("cleanup")

        with mock.patch.object(diag.wm, "find_linux_installation", return_value="/opt/aster"), \
                mock.patch.object(diag, "run_solver", side_effect=run), \
                mock.patch.object(diag.subprocess, "run", side_effect=cleanup):
            diag.solve(directory)
        self.assertEqual(events, ["solver_finished", "cleanup", "cleanup"])
        self.assertEqual(diag.read(directory / "manifest.json")["status"], "completed")
        self.assertEqual((directory / "solver.log").read_text(), "finished\n")
        self.assertFalse((self.output / "solver.lock").exists())

    def test_nonzero_solver_exit_or_missing_summary_preserves_failed_scratch(self):
        for returncode, output, error in (
                (7, "solver failed\n", r"failed \(7\)"),
                (0, "", "no summary.json.*solver.live.log")):
            with self.subTest(returncode=returncode):
                directory = diag.prepare(self.study, self.output, "elastic", 570, 8, 20480)
                completed = subprocess.CompletedProcess(["solver"], returncode, output, "")
                with mock.patch.object(diag.wm, "find_linux_installation", return_value="/opt/aster"), \
                        mock.patch.object(diag, "run_solver", return_value=completed), \
                        mock.patch.object(diag.subprocess, "run") as cleanup:
                    with self.assertRaisesRegex((RuntimeError, ValueError), error):
                        diag.solve(directory)
                self.assertEqual(cleanup.call_count, 1)
                manifest = diag.read(directory / "manifest.json")
                self.assertEqual(manifest["status"], "failed")
                self.assertIn("linux_work_directory", manifest)
                self.assertRegex(manifest["error"], error)
                self.assertEqual((directory / "solver.log").read_text(), output)
                self.assertFalse((self.output / "solver.lock").exists())

    def test_shared_solver_lock_blocks_other_stages(self):
        directory = diag.prepare(self.study, self.output, "elastic", 570, 8, 20480)
        lock = self.output / "solver.lock"
        lock.write_text("existing owner", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "concurrently"), mock.patch.object(diag, "_solve") as solve:
            diag.solve(directory)
        solve.assert_not_called()
        self.assertEqual(lock.read_text(), "existing owner")

    def test_corrected_output_subdirectories_share_the_study_solver_lock(self):
        directory = diag.prepare(self.study, self.output / "corrected", "elastic", 570, 8, 20480)
        lock = self.output / "solver.lock"
        lock.write_text("existing owner", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "concurrently"), mock.patch.object(diag, "_solve") as solve:
            diag.solve(directory)
        solve.assert_not_called()
        self.assertEqual(lock.read_text(), "existing owner")

    def test_comparison_reports_missing_and_completed_stages_without_inventing_values(self):
        directory = diag.prepare(self.study, self.output, "elastic", 570, 8, 20480)
        manifest = diag.read(directory / "manifest.json")
        manifest["status"] = "completed"
        diag.write(directory / "manifest.json", manifest)
        diag.write(directory / "summary.json", self.summary())
        (directory / "result.med").write_bytes(b"result")
        result = diag.compare(self.study, self.output)
        self.assertEqual([row["status"] for row in result["diagnostics"]],
                         ["completed", "not_run", "not_run"])
        self.assertEqual(result["diagnostics"][0]["final"]["regions"]["wall"]
                         ["maximum_principal_stress_pa"], 1.5e6)
        self.assertNotIn("final", result["diagnostics"][1])

    def test_comparison_rejects_mixed_interface_policies(self):
        for stage, operator in (("elastic", diag.wm.INTERFACE_OPERATOR), ("seeded", "legacy_lumped")):
            directory = diag.prepare(self.study, self.output, stage, 570, 8, 20480)
            manifest = diag.read(directory / "manifest.json")
            manifest.update(status="completed", interface_operator=operator)
            diag.write(directory / "manifest.json", manifest)
            diag.write(directory / "summary.json", self.summary(stage))
            (directory / "result.med").write_bytes(b"result")
        with self.assertRaisesRegex(ValueError, "mix legacy lumped and corrected"):
            diag.compare(self.study, self.output)


if __name__ == "__main__":
    unittest.main()
