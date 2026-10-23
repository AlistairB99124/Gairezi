"""Regression checks for isolated three-body solver orchestration."""
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from Elmer import three_body_diagnostic as diagnostic
from Elmer import three_body_solver as solver


ROOT = Path(__file__).resolve().parents[1]


class AppJobWorkerTests(unittest.TestCase):
    def test_app_worker_uses_job_paths_and_concrete_only_extrema(self):
        from Elmer.regions import combined_structure
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            report_path = root / "result_report.json"
            config = {"material": {"concrete": {"tensile_strength_pa": 2e6}}}
            geometry = {
                "material": {"density_kg_m3": 2400}, "loads": {"gravity_z_m_s2": -9.81},
                "wall_geometry": {"max_thickness_m": 4}, "maximum_applied_pressure_pa": 3e5,
            }
            diagnostic_report = {
                "nodes": 100, "cells_by_body": {"1": 30, "2": 30},
                "foundation": {"present": True},
                "wall_foundation_connection": "bonded",
                "foundation_bedrock_connection": "bonded",
                "wall_bedrock_connection": "not_in_contact",
                "wall_connection": "spring_jointed_blocks",
                "wall_joints": [{"name": "wall_joint_1", "z_m": -10.0, "tags": [21, 22]}],
                "interface_stiffness_n_per_m3": {},
                "assumptions": ["rock rollers"],
                "rock_outer_support": {"model": "rollers", "faces": {}},
            }
            def prepare_job(config_root, model, **kwargs):
                self.assertEqual(config_root, root)
                self.assertEqual(kwargs, {"joints": "spring", "integrated": True})
                model.mkdir(parents=True)
                (model / "diagnostic.json").write_text(json.dumps(diagnostic_report))
            def export(result, **kwargs):
                reporting = output / "model" / "reporting"
                reporting.mkdir()
                summary = {
                    "bodies": {
                        name: {"peak_tension_mpa": tension, "peak_compression_mpa": compression,
                               "reporting_cells": count}
                        for name, tension, compression, count in (
                            ("wall", 2.4, 3.6, 30), ("bedrock", 99, 99, 5))
                    },
                    "rock_reporting_distance_m": 5., "limitations": ["filter only"],
                }
                (reporting / "stress_summary.json").write_text(json.dumps(summary))
                return reporting / "structure_and_near_rock.vtu"
            with patch.object(solver, "_validate_run", return_value=config), patch.object(
                combined_structure, "build_combined_structure",
            ), patch.object(combined_structure, "audit_combined_structure", return_value=geometry), patch.object(
                solver, "prepare", side_effect=prepare_job,
            ), patch.object(solver, "solve", return_value=output / "model" / "results" / "full.pvtu") as solve, patch.object(
                solver, "export_reporting_region", side_effect=export,
            ):
                report = solver.run_job(root, output, report_path)
                solve.assert_called_once_with(output / "model", partitions=4, timeout=None)
                analysis = report["stress_analysis"]
                self.assertEqual(analysis["max_tensile_principal_stress_pa"]["value_pa"], 2.4e6)
                self.assertEqual(analysis["max_compressive_principal_stress_pa"]["value_pa"], -3.6e6)
                self.assertEqual(analysis["element_count"], 30)
                self.assertEqual(analysis["tension_utilization"], 1.2)
                self.assertEqual(report["cells"], 60)
                self.assertEqual(report["model_kind"], "three_body")
                self.assertEqual(report["wall_bedrock_connection"], "not_in_contact")
                self.assertEqual(report["wall_foundation_connection"], "bonded")
                self.assertEqual(report["foundation_bedrock_connection"], "bonded")
                self.assertEqual(report["wall_connection"], "spring_jointed_blocks")
                self.assertEqual(report["wall_joints"], diagnostic_report["wall_joints"])
                self.assertIn("wall thickness blocks joined by linear springs", report["connection_assumption"])
                self.assertIn("Bonded wall/foundation interface; bonded foundation/bedrock interface",
                              report["active_support"])
                self.assertNotIn("plinth", json.dumps(report).lower())
                self.assertIn("normal roller springs", report["active_support"])
                self.assertEqual(report["rock_outer_support"]["model"], "rollers")
                self.assertIn(
                    "geometry.bedrock.support.spring_n_per_m3: inactive for bonded foundation/bedrock interface",
                    report["inactive_inputs"],
                )
                self.assertEqual(report["stress_bodies"]["bedrock"]["peak_tension_mpa"], 99)
                self.assertEqual(json.loads(report_path.read_text()), report)
                with self.assertRaisesRegex(ValueError, "fresh"):
                    solver.run_job(root, output, report_path)


class ReportingExportTests(unittest.TestCase):
    def test_export_runs_pvpython_and_requires_both_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "model" / "results" / "full.pvtu"
            expected = result.parent.parent / "reporting"
            def exported(*args, **kwargs):
                expected.mkdir(parents=True)
                (expected / "structure_and_near_rock.vtu").touch()
                (expected / "stress_summary.json").touch()
            with patch.object(solver, "pvpython_executable", return_value="pvpython"), patch.object(
                solver.subprocess, "run", side_effect=exported,
            ) as worker:
                path = solver.export_reporting_region(result, timeout=120)
                self.assertEqual(path, expected / "structure_and_near_rock.vtu")
                self.assertEqual(worker.call_args.kwargs, {"check": True, "timeout": 120})
                self.assertEqual(worker.call_args.args[0][0], "pvpython")
                self.assertIn(str(result), worker.call_args.args[0])
            (expected / "stress_summary.json").unlink()
            with patch.object(solver, "pvpython_executable", return_value="pvpython"), patch.object(
                solver.subprocess, "run",
            ), self.assertRaisesRegex(RuntimeError, "result and summary"):
                solver.export_reporting_region(result)


class ThreeBodySolverTests(unittest.TestCase):
    def setUp(self):
        self.reporting = self.enterContext(patch.object(
            solver, "export_reporting_region",
            side_effect=lambda result, **kwargs: result.parent.parent / "reporting" / "structure_and_near_rock.vtu",
        ))

    def test_reporting_failure_does_not_advertise_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            solver, "prepare",
        ), patch.object(solver.shutil, "which", return_value="solver"), patch.object(
            solver, "mpi_launcher",
        ), patch.object(solver, "pvpython_executable"), patch.object(
            solver, "solve", return_value=Path(directory) / "model" / "results" / "full.pvtu",
        ):
            self.reporting.side_effect = RuntimeError("Reporting export failed")
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Reporting export failed"):
                solver.run(ROOT, output)
            self.assertFalse((output / "latest-run.json").exists())
            report = json.loads(next(output.glob("*\\run.json")).read_text())
            self.assertEqual(report["status"], "failed")

    def test_preparation_does_not_solve_or_advertise_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            solver, "prepare",
        ) as prepare, patch.object(solver, "solve") as solve:
            output = Path(directory)
            report = solver.run(ROOT, output, prepare_only=True)
            prepare.assert_called_once_with(
                ROOT, Path(report["output_directory"]) / "model",
                joints="spring", integrated=True,
            )
            solve.assert_not_called()
            self.assertEqual(report["status"], "prepared")
            self.assertIsNone(report["result"])
            self.assertFalse((output / "latest-run.json").exists())

    def test_success_uses_unique_runs_configured_joints_and_no_implicit_timeout(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            solver, "prepare",
        ), patch.object(solver.shutil, "which", return_value="solver"), patch.object(
            solver, "mpi_launcher",
        ), patch.object(solver, "pvpython_executable"), patch.object(
            solver, "solve", side_effect=lambda output, **kwargs: output / "results" / "three_body_results_t0001.pvtu",
        ) as solve:
            output = Path(directory)
            first = solver.run(ROOT, output)
            second = solver.run(ROOT, output)
            self.assertNotEqual(first["output_directory"], second["output_directory"])
            self.assertEqual(second["status"], "succeeded")
            self.assertTrue(second["reporting_result"].endswith("structure_and_near_rock.vtu"))
            self.assertTrue(second["stress_summary"].endswith("stress_summary.json"))
            self.assertIsNone(solve.call_args.kwargs["timeout"])
            self.assertEqual(solve.call_args.kwargs["partitions"], 4)
            self.assertEqual(json.loads((output / "latest-run.json").read_text()), second)

    def test_failure_is_persisted_and_does_not_replace_last_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            solver, "prepare",
        ), patch.object(solver.shutil, "which", return_value="solver"), patch.object(
            solver, "mpi_launcher",
        ), patch.object(solver, "pvpython_executable"), patch.object(
            solver, "solve", side_effect=RuntimeError("not converged"),
        ):
            output = Path(directory)
            latest = output / "latest-run.json"
            latest.write_text('{"status": "previous success"}')
            with self.assertRaisesRegex(RuntimeError, "not converged"):
                solver.run(ROOT, output)
            reports = list(output.glob("*\\run.json"))
            self.assertEqual(len(reports), 1)
            failed = json.loads(reports[0].read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["result"])
            self.assertEqual(failed["error"], "not converged")
            self.assertEqual(json.loads(latest.read_text())["status"], "previous success")

    def test_missing_dependencies_fail_before_mesh_generation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            solver.shutil, "which", return_value=None,
        ), patch.object(solver, "prepare") as prepare:
            with self.assertRaisesRegex(RuntimeError, "ElmerGrid"):
                solver.run(ROOT, Path(directory))
            prepare.assert_not_called()
            self.assertFalse(list(Path(directory).iterdir()))

    def test_missing_support_definition_is_not_silently_bonded(self):
        config = json.loads((ROOT / "config.json").read_text())
        del config["geometry"]["bedrock"]["support"]["spring_n_per_m3"]["y"]
        with patch.object(solver, "load_config", return_value=config):
            with self.assertRaisesRegex(ValueError, "geometry.bedrock.support.spring_n_per_m3.y"):
                solver.run(ROOT, ROOT)

    def test_invalid_run_controls_fail_before_preparing(self):
        for controls in ({"partitions": 0}, {"timeout": 0}, {"timeout": -1}):
            with self.subTest(controls=controls), patch.object(solver, "prepare") as prepare:
                with self.assertRaises(ValueError):
                    solver.run(ROOT, ROOT, **controls)
                prepare.assert_not_called()

    def test_solver_rejects_stale_results_before_running_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "diagnostic.json").write_text("{}")
            (output / "results").mkdir()
            with patch.object(diagnostic, "convert_with_elmergrid") as convert:
                with self.assertRaisesRegex(ValueError, "fresh run"):
                    diagnostic.solve(output, timeout=None)
                convert.assert_not_called()

    def test_failed_result_validation_does_not_record_success(self):
        for field in ("finite_displacement", "finite_stress"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                report = {"joints": "bonded", "boundary_id_map": {},
                          "solver_input": "dam_model.sif", "result_name": "three_body_results",
                          "result": None}
                path = output / "diagnostic.json"
                path.write_text(json.dumps(report))
                gaps = {"finite_displacement": True, "finite_stress": True}
                gaps[field] = False
                with patch.object(diagnostic.shutil, "which", return_value="solver"), patch.object(
                    diagnostic, "mpi_launcher", return_value=("mpiexec", "-n"),
                ), patch.object(diagnostic, "pvpython_executable"), patch.object(
                    diagnostic, "convert_with_elmergrid", return_value=True,
                ), patch.object(diagnostic.subprocess, "Popen") as process, patch.object(
                    diagnostic, "attribute_result",
                ), patch.object(diagnostic, "measure_joint_gaps", return_value=gaps), patch.object(
                    Path, "is_file", return_value=True,
                ):
                    process.return_value.wait.return_value = 0
                    with self.assertRaisesRegex(RuntimeError, "non-finite"):
                        diagnostic.solve(output, timeout=None)
                    self.assertEqual(json.loads(path.read_text()), report)
                    self.assertEqual(process.call_args.args[0][-1], "dam_model.sif")
                    process.return_value.wait.assert_called_once_with(timeout=30.)

    def test_single_rank_uses_unpartitioned_mesh_and_wraps_serial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = {"joints": "bonded", "boundary_id_map": {},
                      "solver_input": "dam_model.sif", "result_name": "three_body_results",
                      "result": None}
            path = output / "diagnostic.json"
            path.write_text(json.dumps(report))
            gaps = {"finite_displacement": True, "finite_stress": True}
            with patch.object(diagnostic.shutil, "which", return_value="solver"), patch.object(
                diagnostic, "mpi_launcher", return_value=("mpiexec", "-n"),
            ), patch.object(diagnostic, "pvpython_executable"), patch.object(
                diagnostic, "convert_with_elmergrid", return_value=True,
            ) as convert, patch.object(diagnostic.subprocess, "Popen") as process, patch.object(
                diagnostic, "attribute_result",
            ) as attribute, patch.object(
                diagnostic, "measure_joint_gaps", return_value=gaps,
            ), patch.object(Path, "is_file", autospec=True, side_effect=lambda file: file.suffix == ".vtu"):
                process.return_value.wait.return_value = 0
                result = diagnostic.solve(output, partitions=1, timeout=None)
                convert.assert_called_once_with(output / "three_body.msh", output / "mesh", None, ())
                attribute.assert_called_once_with(output, result, timeout=None)
                self.assertEqual(result.suffix, ".pvtu")
                self.assertEqual(json.loads(path.read_text())["result"], str(result))

    def test_solver_wait_relays_output_and_heartbeats_without_a_default_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "solver.log"
            log.write_text("StressSolve: Starting assembly...\n")
            with patch.object(diagnostic.subprocess, "Popen") as process, patch("builtins.print") as print_output:
                worker = process.return_value
                worker.wait.side_effect = [subprocess.TimeoutExpired("solver", 30), 0]
                self.assertEqual(diagnostic.wait_for_solver(worker, log, None), 0)
                self.assertEqual(worker.wait.call_count, 2)
                self.assertTrue(all(call.kwargs["timeout"] == 30. for call in worker.wait.call_args_list))
                messages = " ".join(str(call.args) for call in print_output.call_args_list)
                self.assertIn("Starting assembly", messages)
                self.assertIn("Elmer running", messages)

    def test_solver_wait_enforces_explicit_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "solver.log"
            log.touch()
            with patch.object(diagnostic.subprocess, "Popen") as process, patch.object(
                diagnostic.time, "monotonic", side_effect=[0., 11.],
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    diagnostic.wait_for_solver(process.return_value, log, 10)
                process.return_value.wait.assert_not_called()

    def test_user_cancellation_is_persisted_without_a_success_pointer(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            solver, "prepare", side_effect=KeyboardInterrupt,
        ):
            output = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                solver.run(ROOT, output, prepare_only=True)
            reports = list(output.glob("*\\run.json"))
            self.assertEqual(json.loads(reports[0].read_text())["status"], "cancelled")
            self.assertFalse((output / "latest-run.json").exists())

    def test_interruption_and_timeout_terminate_only_the_launched_mpi_tree(self):
        for error in (KeyboardInterrupt(), subprocess.TimeoutExpired("solver", 10)):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / "diagnostic.json").write_text(json.dumps({
                    "joints": "bonded", "boundary_id_map": {},
                }))
                with patch.object(diagnostic.shutil, "which", return_value="solver"), patch.object(
                    diagnostic, "mpi_launcher", return_value=("mpiexec", "-n"),
                ), patch.object(diagnostic, "pvpython_executable"), patch.object(
                    diagnostic, "convert_with_elmergrid", return_value=True,
                ), patch.object(diagnostic.subprocess, "Popen") as process, patch.object(
                    diagnostic, "wait_for_solver", side_effect=error,
                ), patch.object(diagnostic.subprocess, "run") as terminate:
                    process.return_value.pid = 12345
                    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else RuntimeError
                    with self.assertRaises(expected):
                        diagnostic.solve(output, timeout=10)
                    terminate.assert_called_once_with(
                        ["taskkill", "/PID", "12345", "/T", "/F"], check=True,
                    )


if __name__ == "__main__":
    unittest.main()
