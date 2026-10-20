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
    def setUp(self):
        # Runner orchestration remains testable independently of the production model-review hold.
        hold = patch("run.assert_fracture_ready")
        hold.start()
        self.addCleanup(hold.stop)

    def test_parallel_cli_only_delegates_the_fracture_solve(self):
        with (
            patch("sys.argv", ["run.py", "run-fracture", "--fracture-processes", "2",
                               "--dam-config", str(LOAD_CASE)]),
            patch("run.environment", return_value={"test": True}),
            patch("run.run_fracture") as fracture,
        ):
            self.assertEqual(main(), 0)
            self.assertEqual(fracture.call_args.kwargs, {
                "processes": 2, "mechanics_formulation": "hybrid", "damage_acceleration": False,
                "final_load_substeps": 1,
            })

    def test_experimental_mechanics_cli_requires_explicit_opt_in(self):
        with (
            patch("sys.argv", ["run.py", "run-fracture", "--fracture-mechanics", "experimental_amor",
                               "--accelerate-damage", "--fracture-final-substeps", "4",
                               "--fracture-processes", "2", "--dam-config", str(LOAD_CASE)]),
            patch("run.environment", return_value={"test": True}),
            patch("run.run_fracture") as fracture,
        ):
            self.assertEqual(main(), 0)
            self.assertEqual(fracture.call_args.kwargs, {
                "processes": 2, "mechanics_formulation": "experimental_amor", "damage_acceleration": True,
                "final_load_substeps": 4,
            })

    def test_spectral_mechanics_cli_routes_only_an_explicit_cracking_request(self):
        with (
            patch("sys.argv", ["run.py", "run-fracture", "--fracture-mechanics", "experimental_spectral",
                               "--accelerate-damage", "--dam-config", str(LOAD_CASE)]),
            patch("run.environment", return_value={"test": True}),
            patch("run.run_fracture") as fracture,
        ):
            self.assertEqual(main(), 0)
            self.assertEqual(fracture.call_args.kwargs["mechanics_formulation"], "experimental_spectral")
            self.assertTrue(fracture.call_args.kwargs["damage_acceleration"])

    def test_user_located_run_records_selected_scenario_without_changing_dam_snapshot(self):
        import numpy as np
        from test_automatic_crack import location, synthetic_fields

        fixture = test_engineering_report.EngineeringReportTests()
        fixture.setUp()

        def import_baseline(structure, result, snapshot, directory):
            points, stress, regions = synthetic_fields()
            np.savez_compressed(directory / "fields.npz", coordinates_m=points, stress_pa=stress,
                                node_regions=regions)
            return {"field_archive": "fields.npz"}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = output / "submitted.json"
            config.write_bytes(LOAD_CASE.read_bytes())
            with self.assertRaisesRegex(ValueError, "No crack location was selected"):
                with patch("elmer_baseline.import_baseline", side_effect=import_baseline):
                    run_fracture(config, output / "unselected", {"test_environment": True}, output / "fake.pvtu")
            with (
                patch("elmer_baseline.import_baseline", side_effect=import_baseline) as importer,
                patch("crack_field.refine_for_phase_field", side_effect=lambda imported, specifications, options, *_:
                      SimpleNamespace(domain=imported.domain, cell_tags=imported.cell_tags,
                                      facet_tags=imported.facet_tags, report=fixture.refinement)),
                patch("dam_phase_field.solve", return_value=SimpleNamespace(report=fixture.result)) as solve,
            ):
                run_fracture(config, output, {"test_environment": True}, output / "fake.pvtu",
                             crack_location=location(97.988, -27.5))
            run = next((output / "dam_phase_field").iterdir())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed_not_engineering_validated")
            self.assertEqual(manifest["mechanics_formulation"], "hybrid")
            self.assertEqual(solve.call_args.kwargs["mechanics_formulation"], "hybrid")
            self.assertTrue(solve.call_args.kwargs["refined_baseline"])
            self.assertTrue(manifest["refined_baseline_required"])
            self.assertEqual(manifest["crack_scenario_names"], ["user_located_crack"])
            self.assertEqual((run / "config.json").read_bytes(), config.read_bytes())
            generated = json.loads((run / "crack_location.json").read_text())
            self.assertEqual(generated["opening_m"], 0.002)
            self.assertEqual(generated["depth_m"], 3)
            self.assertAlmostEqual(generated["selected_location_m"]["z"], -27.5)
            self.assertTrue((run / "results.md").is_file())
            self.assertEqual(solve.call_args.args[2][0].shape, "annular_sector")
            self.assertEqual(solve.call_args.args[2][0].centre_m[2], -27.5)
            self.assertEqual({kind for kind, _ in importer.call_args.args[0].cells}, {4})

    def test_crack_location_cli_is_forwarded_and_rejected_elsewhere(self):
        with (
            patch("sys.argv", ["run.py", "run-fracture", "--dam-config", str(LOAD_CASE),
                               "--elmer-result", "result.pvtu", "--crack-location", "24.5", "-27.5"]),
            patch("run.environment", return_value={"test": True}),
            patch("run.run_fracture") as fracture,
        ):
            self.assertEqual(main(), 0)
            self.assertEqual(fracture.call_args.kwargs["crack_location"], (24.5, -27.5))
        for arguments in (
            ["run-fracture", "--crack-location", "24.5", "-27.5"],
            ["run-baseline", "--elmer-result", "r.pvtu", "--crack-location", "24.5", "-27.5"],
            ["run-fracture", "--elmer-result", "r.pvtu", "--crack-location", "nan", "-27.5"],
            ["run-fracture", "--elmer-result", "r.pvtu", "--crack-location", "24.5"],
        ):
            with self.subTest(arguments=arguments), patch("sys.argv", ["run.py", *arguments]):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

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
                patch("automatic_crack.locate_crack", side_effect=AssertionError("Crack generated")),
            ):
                self.assertEqual(main(), 0)
                self.assertEqual(baseline.call_args.kwargs, {"phase": phase})
                self.assertEqual(baseline.call_args.args[0], LOAD_CASE.resolve())

    def test_baseline_cli_rejects_conflicting_options(self):
        for arguments in (
            ["run-baseline", "--baseline-phase"],
            ["run", "--baseline-phase"],
            ["run-fracture", "--baseline-phase", "--elmer-result", "result.pvtu"],
            ["run-baseline", "--fracture-processes", "2"],
            ["run-fracture", "--baseline-phase", "--fracture-processes", "2"],
            ["run-baseline", "--fracture-mechanics", "experimental_amor"],
            ["run-fracture", "--baseline-phase", "--fracture-mechanics", "experimental_amor"],
            ["run", "--fracture-mechanics", "experimental_amor"],
            ["run-fracture", "--accelerate-damage"],
            ["run-baseline", "--accelerate-damage"],
            ["run-fracture", "--fracture-final-substeps", "0"],
            ["run-baseline", "--fracture-final-substeps", "2"],
            ["run-fracture", "--baseline-phase", "--fracture-final-substeps", "2"],
            ["run-baseline", "--fracture-mechanics", "experimental_spectral"],
            ["run-fracture", "--baseline-phase", "--fracture-mechanics", "experimental_spectral"],
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
                    patch("automatic_crack.locate_crack", side_effect=AssertionError("Crack generated")),
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

    def test_canonical_job_rejects_lower_explicit_resolution_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = configured_test_case()
            config["crack_simulations"][0]["solver"]["minimum_elements_per_length_scale"] = .9
            snapshot = output / "submitted.json"
            snapshot.write_text(json.dumps(config))
            structure = SimpleNamespace(three_body_config=config, three_body_source={})
            with (
                patch("run.submitted_structure", return_value=structure),
                patch("run.assert_fracture_ready", return_value={"directory": "verified-baseline"}),
                patch("elmer_baseline.import_baseline", return_value={}),
                patch("dam_mesh.import_structure") as importer,
                patch("dam_phase_field.solve") as solver,
                self.assertRaisesRegex(ValueError, "requires at least 2"),
            ):
                run_fracture(snapshot, output, {"test": True}, output / "reference.pvtu")
            importer.assert_not_called()
            solver.assert_not_called()
            directory = next((output / "dam_phase_field").iterdir())
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("configured target is 0.9", manifest["error"])
            self.assertEqual((directory / "config.json").read_bytes(), snapshot.read_bytes())
            self.assertFalse((directory / "phase_field_report.json").exists())

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
