"""Integration checks for isolated phase-field job setup and failure reporting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from run import LOAD_CASE, main, run_baseline, run_fracture
from test_crack_field import configured_test_case
import test_engineering_report


class FractureRunnerTests(unittest.TestCase):
    def test_automatic_run_records_generated_scenario_without_changing_dam_snapshot(self):
        fixture = test_engineering_report.EngineeringReportTests()
        fixture.setUp()
        baseline = {"max_principal_stress": {
            "value_pa": 2e6, "centroid_xyz_m": [79, 0, -25], "element_id": 1, "region_id": 2,
        }}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = output / "submitted.json"
            config.write_bytes(LOAD_CASE.read_bytes())
            with (
                patch("elmer_baseline.import_baseline", return_value=baseline) as importer,
                patch("crack_field.refine_for_phase_field", side_effect=lambda imported, specifications, options:
                      SimpleNamespace(domain=imported.domain, cell_tags=imported.cell_tags,
                                      facet_tags=imported.facet_tags, report=fixture.refinement)),
                patch("dam_phase_field.solve", return_value=SimpleNamespace(report=fixture.result)) as solve,
            ):
                run_fracture(config, output, {"test_environment": True}, output / "fake.pvtu")
            run = next((output / "dam_phase_field").iterdir())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed_not_engineering_validated")
            self.assertEqual(manifest["crack_scenario_names"], ["automatic_peak_tension"])
            self.assertEqual((run / "config.json").read_bytes(), config.read_bytes())
            generated = json.loads((run / "automatic_crack.json").read_text())
            self.assertEqual(generated["opening_m"], 0.002)
            self.assertEqual(generated["depth_m"], 3)
            self.assertTrue((run / "results.md").is_file())
            self.assertEqual(solve.call_args.args[2][0].shape, "annular_sector")
            self.assertEqual({kind for kind, _ in importer.call_args.args[0].cells}, {4})

    def test_baseline_cli_routes_the_exact_api_commands(self):
        for arguments, phase in (
            (["run-baseline"], False),
            (["run-fracture", "--baseline-phase"], True),
        ):
            with (
                self.subTest(phase=phase),
                patch("sys.argv", ["run.py", *arguments, "--dam-config", str(LOAD_CASE),
                                   "--output", "isolated-output"]),
                patch("run.environment", return_value={"test": True}),
                patch("run.run_baseline") as baseline,
                patch("run.load_scenarios", side_effect=AssertionError("Unrelated scenarios loaded")),
                patch("automatic_crack.generate_crack", side_effect=AssertionError("Crack generated")),
            ):
                self.assertEqual(main(), 0)
                self.assertEqual(baseline.call_args.kwargs, {"phase": phase})
                self.assertEqual(baseline.call_args.args[0], LOAD_CASE.resolve())

    def test_baseline_cli_rejects_conflicting_options(self):
        for arguments in (
            ["run-baseline", "--baseline-phase"],
            ["run", "--baseline-phase"],
            ["run-fracture", "--baseline-phase", "--elmer-result", "result.pvtu"],
        ):
            with self.subTest(arguments=arguments), patch("sys.argv", ["run.py", *arguments]):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_both_baselines_write_real_solver_reports_and_preserve_input(self):
        import test_dam_mesh
        import test_dam_phase_field

        structure = test_dam_mesh.DamMeshTests().cube()
        _, material_model, _, _ = test_dam_phase_field.DamPhaseFieldTests().model()
        for name in ("material", "loads", "foundation_support", "plinth_support"):
            setattr(structure, name, getattr(material_model, name))
        structure.boundaries = [
            (1 if index == 0 else 7, face)
            for index, (_, face) in enumerate(structure.boundaries)
        ]
        for phase in (False, True):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                snapshot = output / "submitted.json"
                snapshot.write_bytes(LOAD_CASE.read_bytes())
                original = snapshot.read_bytes()
                with (
                    patch("run.submitted_structure", return_value=structure),
                    patch("automatic_crack.generate_crack", side_effect=AssertionError("Crack generated")),
                ):
                    run_baseline(snapshot, output, {"test": True}, phase=phase)
                root = output / ("dam_phase_field" if phase else "dam_elastic_baseline")
                directory = next(root.iterdir())
                manifest = json.loads((directory / "manifest.json").read_text())
                name = "Baseline_Phase" if phase else "Baseline_NoPhase"
                self.assertEqual(manifest["scenario_name"], name)
                self.assertEqual(manifest["crack_scenario_names"], [name])
                self.assertEqual(manifest["status"], "completed_not_engineering_validated")
                report_name = "phase_field_report.json" if phase else "baseline_report.json"
                report = json.loads((directory / report_name).read_text())
                self.assertEqual(report["initial_cracks"], [])
                self.assertEqual(report["internal_crack_pressure_pa"], 0)
                self.assertIn("maximum_principal_tensile_stress_pa", report)
                self.assertIn("maximum_principal_compressive_stress_pa", report)
                self.assertTrue((directory / "results.md").is_file())
                self.assertEqual((directory / "config.json").read_bytes(), original)
                self.assertEqual(snapshot.read_bytes(), original)
                if phase:
                    self.assertEqual(report["initial_damage"], "zero everywhere; no seeded crack")
                    self.assertIn("mesh_resolution", report)

    def test_failed_baseline_records_original_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch("run.submitted_structure", side_effect=ValueError("invalid submitted mesh")):
                with self.assertRaisesRegex(ValueError, "invalid submitted mesh"):
                    run_baseline(LOAD_CASE, output, {"test": True})
            directory = next((output / "dam_elastic_baseline").iterdir())
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("invalid submitted mesh", manifest["error"])
            self.assertFalse((directory / "baseline_report.json").exists())

    def test_invalid_elmer_handover_fails_run_and_records_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config_path = output / "config.json"
            config_path.write_bytes(LOAD_CASE.read_bytes())
            baseline = output / "missing.pvtu"
            with self.assertRaises(FileNotFoundError):
                run_fracture(config_path, output, {"test_environment": True}, baseline)
            runs = list((output / "dam_phase_field").iterdir())
            self.assertEqual(len(runs), 1)
            manifest = json.loads((runs[0] / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("missing.pvtu", manifest["error"])
            self.assertFalse((runs[0] / "elmer_baseline_fields.npz").exists())

    def test_invalid_crack_location_produces_a_failed_isolated_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = configured_test_case()
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
