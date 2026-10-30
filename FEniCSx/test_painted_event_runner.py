"""Isolated experimental runs must fail explicitly without touching saved jobs."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from FEniCSx.painted_event_runner import main, run_saved_event, solver_options


class PaintedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.configuration = self.root / "config.json"
        config = {"geometry": {}, "material": {"concrete": {
            "tensile_strength_pa": 2e6, "selected_tensile_strength_pa": 1e6}}}
        self.configuration.write_text(json.dumps(config))
        self.saved = self.root / "saved.json"
        packet = {
            "schema_version": 1, "analysis_kind": "painted_tensile_loss", "job_id": "job",
            "selection_id": "saved", "status": "saved_selection_only", "solver_ready": False,
            "configuration_digest": hashlib.sha256(json.dumps(
                config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "request": {"affected_element_ids": [0], "crack_start_edge_ids": [1]}}
        packet["selection_sha256"] = hashlib.sha256(json.dumps(
            packet, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.saved.write_text(json.dumps(packet))
        self.options = {
            "maximum_source_cells": 4, "maximum_volume_change": 1e-12,
            "event": {}, "contact": {
                "gap_tolerance_m": 1e-12, "force_tolerance_n": 1e-7,
                "relative_residual_tolerance": 1e-8, "maximum_iterations": 100},
            "growth": {"fracture_energy_j_m2": 2700., "energy_tolerance_j": 1e-8,
                       "maximum_extensions": 10, "maximum_patch_faces": 2}}
        self.structure = SimpleNamespace(cells=[None] * 4)
        self.binding = {
            "affected_element_ids": [0], "marker_source_edges": [[0, 1]], "opening_limit_m": .003}
        preflight = patch("FEniCSx.painted_event_runner.source_preflight", return_value={
            "status": "passed_unsplit_candidate_preflight_not_solved"})
        self.preflight = preflight.start()
        self.addCleanup(preflight.stop)

    def run_event(self, output=None, *, mode="dry", options=None, job_id="job"):
        return run_saved_event(
            self.configuration, self.root, self.root, self.root / "fields.npz", self.saved, job_id,
            water_mode=mode, options=self.options if options is None else options,
            output=output or self.root / "output")

    def test_explicit_option_schema_and_numerical_validation(self):
        invalid = (
            {"maximum_source_cells": True}, {"maximum_source_cells": 0},
            {"maximum_volume_change": float("nan")}, {"maximum_volume_change": True},
            {"maximum_volume_change": 1.}, {"event": []}, {"event": {"unknown": 1}},
            {"contact": {}}, {"growth": {"fracture_energy_j_m2": -1}},
            {"unsupported": 1},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                solver_options({**self.options, **change}, "dry")
        for mode in ("unknown", "different_head"):
            with self.assertRaises(ValueError):
                solver_options(self.options, mode)
        with self.assertRaisesRegex(ValueError, "exactly"):
            solver_options(self.options, "common_head_hydrostatic")
        wet = {**self.options, "hydraulics": {
            "closure_tolerance_m": 1e-12, "flow_relative_tolerance": 1e-10}}
        self.assertEqual(solver_options(wet, "common_head_hydrostatic")["closure_tolerance_m"], 1e-12)
        for values in ({}, {"closure_tolerance_m": 0., "flow_relative_tolerance": 1e-10},
                       {"closure_tolerance_m": 1e-12, "flow_relative_tolerance": 1.}):
            with self.assertRaises(ValueError):
                solver_options({**wet, "hydraulics": values}, "common_head_hydrostatic")

    def test_invalid_snapshot_stops_before_environment_or_source_and_retains_failure(self):
        original_config = self.configuration.read_bytes()
        with patch("FEniCSx.painted_event_runner.environment") as environment, patch(
            "FEniCSx.painted_event_runner.verified_source_selection"
        ) as bind:
            with self.assertRaisesRegex(RuntimeError, "identity"):
                self.run_event(job_id="foreign")
            environment.assert_not_called()
            bind.assert_not_called()
        report = json.loads((self.root / "output" / "manifest.json").read_text())
        self.assertEqual(report["status"], "failed_experimental_painted_event")
        self.assertNotIn("accepted_results", report)
        self.assertFalse((self.root / "output" / "growth").exists())
        self.assertEqual(self.configuration.read_bytes(), original_config)

    def test_source_resource_ceiling_prevents_splitting_and_numerical_solve(self):
        with patch("FEniCSx.painted_event_runner.environment", return_value={"test": True}), patch(
            "FEniCSx.painted_event_runner.verified_source_selection",
            return_value=(self.structure, self.binding)
        ), patch("FEniCSx.painted_event_runner.solve_source_dry_growth") as solve:
            with self.assertRaisesRegex(RuntimeError, "exceeding explicit"):
                self.run_event(options={**self.options, "maximum_source_cells": 3})
            solve.assert_not_called()
            self.preflight.assert_not_called()
        report = json.loads((self.root / "output" / "manifest.json").read_text())
        self.assertEqual(report["source_cells"], 4)
        self.assertEqual(report["maximum_source_cells"], 3)
        self.assertNotIn("accepted_results", report)
        self.assertFalse((self.root / "output" / "growth").exists())

    def test_hydraulic_failure_never_falls_back_to_dry_and_preserves_input_snapshots(self):
        options = {**self.options, "hydraulics": {
            "closure_tolerance_m": 1e-12, "flow_relative_tolerance": 1e-10}}
        original = self.saved.read_bytes()
        with patch("FEniCSx.painted_event_runner.environment", return_value={"test": True}), patch(
            "FEniCSx.painted_event_runner.verified_source_selection",
            return_value=(self.structure, self.binding)
        ), patch("FEniCSx.painted_event_runner.solve_source_hydrostatic_growth",
                 side_effect=RuntimeError("different heads not supported")) as wet, patch(
            "FEniCSx.painted_event_runner.solve_source_dry_growth"
        ) as dry:
            with self.assertRaisesRegex(RuntimeError, "different heads"):
                self.run_event(mode="common_head_hydrostatic", options=options)
            dry.assert_not_called()
            self.assertEqual(wet.call_args.kwargs["selected_tensile_strength_pa"], 1e6)
        output = self.root / "output"
        report = json.loads((output / "manifest.json").read_text())
        self.assertEqual(report["status"], "failed_experimental_painted_event")
        self.assertNotIn("accepted_results", report)
        self.assertEqual((output / "inputs" / "saved.json").read_bytes(), original)
        self.assertEqual(self.saved.read_bytes(), original)
        self.assertEqual(report["growth_options"]["fracture_energy_j_m2"], 2700.)
        self.assertEqual(report["selected_tensile_strength_pa"], 1e6)

    def test_cli_requires_explicit_experiment_and_forwards_saved_input_paths(self):
        numerical = self.root / "numerical.json"
        numerical.write_text(json.dumps(self.options))
        arguments = [
            "--configuration", str(self.configuration), "--model-directory", str(self.root),
            "--baseline-directory", str(self.root), "--fields", str(self.root / "fields.npz"),
            "--saved-selection", str(self.saved), "--options", str(numerical),
            "--output", str(self.root / "output"), "--job-id", "job", "--water-mode", "dry"]
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            main(arguments)
        self.assertEqual(error.exception.code, 2)
        with patch("FEniCSx.painted_event_runner.run_saved_event", return_value={
            "status": "completed_experimental_not_engineering_validated",
            "accepted_results": {"event_report": "growth/initial/event_report.json"}
        }) as run, patch("sys.stdout", new_callable=io.StringIO) as stdout:
            main(["--experimental", *arguments])
        self.assertEqual(run.call_args.kwargs["saved_selection"], self.saved)
        self.assertEqual(run.call_args.kwargs["options"], self.options)
        self.assertEqual(json.loads(stdout.getvalue())["status"],
                         "completed_experimental_not_engineering_validated")

    def test_missing_accepted_artifact_is_failure_not_completed_manifest(self):
        def solve(*args, output, **kwargs):
            output.mkdir()
            (output / "source_report.json").write_text("{}")
            return {"accepted_event_directory": "missing"}

        with patch("FEniCSx.painted_event_runner.environment", return_value={"test": True}), patch(
            "FEniCSx.painted_event_runner.verified_source_selection",
            return_value=(self.structure, self.binding)
        ), patch("FEniCSx.painted_event_runner.solve_source_dry_growth", side_effect=solve):
            with self.assertRaisesRegex(RuntimeError, "Accepted result is missing"):
                self.run_event()
        report = json.loads((self.root / "output" / "manifest.json").read_text())
        self.assertEqual(report["status"], "failed_experimental_painted_event")
        self.assertNotIn("accepted_results", report)

    def test_prepare_only_records_no_accepted_result_and_never_calls_growth(self):
        with patch("FEniCSx.painted_event_runner.environment", return_value={"test": True}), patch(
            "FEniCSx.painted_event_runner.verified_source_selection",
            return_value=(self.structure, self.binding)
        ), patch("FEniCSx.painted_event_runner.solve_source_dry_growth") as solve:
            result = run_saved_event(
                self.configuration, self.root, self.root, self.root / "fields.npz", self.saved, "job",
                water_mode="dry", options=self.options, output=self.root / "prepared", prepare_only=True)
            solve.assert_not_called()
        self.assertEqual(result["status"], "prepared_experimental_not_solved")
        self.assertNotIn("accepted_results", result)
        self.assertFalse((self.root / "prepared" / "growth").exists())

    def test_run_announces_the_preparing_phase_before_any_progress(self):
        with patch("FEniCSx.painted_event_runner.environment", return_value={"test": True}), patch(
            "FEniCSx.painted_event_runner.verified_source_selection",
            return_value=(self.structure, self.binding)
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            run_saved_event(
                self.configuration, self.root, self.root, self.root / "fields.npz", self.saved, "job",
                water_mode="dry", options=self.options, output=self.root / "prepared", prepare_only=True)
        lines = stdout.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("Phase: "), lines[:2])
        self.assertEqual(json.loads(lines[0][len("Phase: "):])["phase"], "preparing")
        self.assertTrue(all(not line.startswith("Phase: ") for line in lines[1:]))

    def test_preflight_hydraulic_rejection_prevents_candidate_splitting(self):
        self.preflight.side_effect = ValueError("constant-pressure crest loading")
        with patch("FEniCSx.painted_event_runner.environment", return_value={"test": True}), patch(
            "FEniCSx.painted_event_runner.verified_source_selection",
            return_value=(self.structure, self.binding)
        ), patch("FEniCSx.painted_event_runner.solve_source_hydrostatic_growth") as wet:
            with self.assertRaisesRegex(RuntimeError, "constant-pressure crest"):
                self.run_event(mode="common_head_hydrostatic", options={
                    **self.options, "hydraulics": {
                        "closure_tolerance_m": 1e-12, "flow_relative_tolerance": 1e-10}})
            wet.assert_not_called()
        self.assertFalse((self.root / "output" / "growth").exists())


if __name__ == "__main__":
    unittest.main()
