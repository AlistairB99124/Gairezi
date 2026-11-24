import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Aster import wall_cohesive_workflow as workflow


class AdaptiveGrowthTests(unittest.TestCase):
    def test_reservoir_final_depth_and_intermediate_warnings_reach_engineering_gate(self):
        base = {
            "loading_mode": "reservoir_filling", "reservoir_loading": {"final_water_depth_m": 33.},
            "activation_warnings": ["Activation limit reached at 28 m."],
            "activation_history_complete": False,
        }
        manifest, report, _, _ = self.run_workflow(dict(base, reached_water_depth_m=32.5))
        self.assertFalse(report["engineering_goal"]["full_load"])
        self.assertFalse(report["engineering_goal"]["passed"])
        self.assertEqual(manifest["status"], "partial_not_engineering_validated")
        manifest, report, _, _ = self.run_workflow(dict(base, reached_water_depth_m=33.))
        self.assertTrue(report["engineering_goal"]["passed"])
        self.assertIn("Activation limit", manifest["warning"])
        self.assertFalse(report["activation_history_complete"])

    def test_cli_without_limit_passes_none_to_the_workflow(self):
        arguments = ["wall_cohesive_workflow", "--config", "config.json", "--model", "model",
                     "--baseline", "baseline", "--output", "output", "--job-id", "fixture"]
        with patch.object(sys, "argv", arguments), \
                patch.object(workflow, "run", return_value={
                    "status": "completed_not_engineering_validated"}) as run:
            workflow.main()
        self.assertIsNone(run.call_args.kwargs["time_limit_s"])

    def run_workflow(self, result, fracture_debug=None, residual_target=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = root / "config.json"
        payload = {"fracture_debug": fracture_debug}
        if residual_target is not None:
            payload["residual_target_tensile_stress_pa"] = residual_target
        config.write_text(json.dumps(payload), encoding="utf-8")
        model = root / "model"
        model.mkdir()
        (model / "three_body.msh").write_text("mesh", encoding="utf-8")
        def prepare(_config, _model, _baseline, output, **kwargs):
            output.mkdir()
            if kwargs.get("screening") is None:
                workflow.write_json(output / "screening.json", {"overstressed_source_rows": [1]})
                workflow.write_json(output / "manifest.json", {})
                return {}
            faces = [(1, 2, 3)]
            vertices = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
            parent = vertices.ravel()
            count = len(parent)
            np.savez(output / "cohesive_state.npz",
                     parent_node_ids=np.concatenate((parent, parent)),
                     minus=np.arange(count, dtype=np.int64).reshape(-1, 3),
                     plus=np.arange(count, 2 * count, dtype=np.int64).reshape(-1, 3))
            return {"candidate_faces": len(faces)}

        def solve(_baseline, _candidates, output, _time_limit):
            output.mkdir()
            return {**{
                "status": "completed",
                "accepted": True, "solver": "Code_Aster",
                "reached_load_factor": 1.,
                "opening_assessment": {"peak_opening_m": .0001},
                "paraview": {"directory": str(output)},
                "growth_source_rows": [],
                "maximum_principal_tensile_stress_pa": 0.,
            }, **result}

        with patch.object(workflow.automatic_preparation, "run", side_effect=prepare), \
                patch.object(workflow.automatic_cohesive, "run", side_effect=solve) as native:
            manifest = workflow.run(config=config, model=model, baseline=root / "baseline",
                                    output=root / "output", job_id="fixture", time_limit_s=30)
        run_dir = next((root / "output" / workflow.OUTPUT_ROOT).iterdir())
        report = json.loads((run_dir / workflow.REPORT_NAME).read_text(encoding="utf-8"))
        return manifest, report, native.call_count, run_dir

    def test_partial_solve_stops_without_restarting(self):
        manifest, report, calls, run_dir = self.run_workflow({
            "status": "partial", "reached_load_factor": .9128125,
            "growth_source_rows": [7], "capacity_assessment": {"message": "Newton did not converge"},
        })
        self.assertEqual(calls, 1)
        self.assertEqual(manifest["adaptive_round"], 1)
        self.assertEqual(report["adaptive_rounds_completed"], 1)
        self.assertEqual(manifest["status"], "partial_not_engineering_validated")
        self.assertEqual(manifest["reached_load_factor"], .9128125)
        self.assertEqual(manifest["remaining_growth_source_rows"], 1)
        self.assertIn("Newton did not converge", manifest["warning"])
        self.assertEqual(report["stop_reason"], "solver_failure_or_incomplete_load")
        self.assertEqual(report["near_base_crack_assessment"]["target_pa"], 500000.)
        self.assertEqual(report["near_base_crack_assessment"]["status"], "incomplete")
        self.assertFalse((run_dir / "solve_round_2").exists())

    def test_preload_failure_and_warning_are_retained_in_pipeline_manifest(self):
        manifest, report, _, _ = self.run_workflow({
            "status": "failed", "accepted": False,
            "preload": {"status": "failed", "message": "intact equilibrium failed"},
        })
        self.assertEqual(manifest["stop_reason"], "preload_failure")
        self.assertEqual(report["stop_reason"], "preload_failure")
        self.assertIn("intact equilibrium failed", manifest["error"])
        manifest, _, _, _ = self.run_workflow({
            "preload": {"status": "completed", "warnings": ["Earlier cracking may have occurred."]}})
        self.assertTrue(manifest["engineering_goal_passed"])
        self.assertIn("Earlier cracking", manifest["warning"])

    def test_resource_failure_preserves_original_cause_in_report_and_manifest(self):
        failure = {"kind": "storage", "message": "JEVEUX_40: No space left on device"}
        manifest, report, calls, _ = self.run_workflow({
            "status": "partial", "reached_load_factor": .24,
            "resource_failure": failure,
        })
        self.assertEqual(calls, 1)
        self.assertEqual(report["stop_reason"], "resource_limit")
        self.assertEqual(manifest["resource_failure"], failure)
        self.assertIn(failure["message"], manifest["error"])

    def test_os_kill_is_not_replaced_by_generic_missing_output_error(self):
        failure = {"kind": "process_killed", "signal": 9,
                   "message": "Code_Aster was killed by the operating system (SIGKILL)"}
        manifest, report, calls, _ = self.run_workflow({
            "status": "failed", "accepted": False, "reached_load_factor": .015,
            "process_failure": failure, "paraview": None,
        })
        self.assertEqual(calls, 1)
        self.assertEqual(report["stop_reason"], "process_killed")
        self.assertEqual(manifest["stop_reason"], "process_killed")
        self.assertEqual(manifest["process_failure"], failure)
        self.assertEqual(manifest["error"], failure["message"])

    def test_growth_stops_before_losing_accepted_fracture_history(self):
        manifest, report, calls, run_dir = self.run_workflow({
            "growth_source_rows": [7], "maximum_principal_tensile_stress_pa": 2.})
        self.assertEqual(calls, 1)
        self.assertEqual(manifest["status"], "partial_not_engineering_validated")
        self.assertEqual(report["stop_reason"], "fracture_history_transfer_unavailable")
        self.assertIn("fracture history", report["adaptive_growth_warning"])
        self.assertFalse((run_dir / "solve_round_2").exists())

    def test_native_continuation_reports_accepted_rounds_and_iteration_limit(self):
        rounds = [
            {"iteration": index, "accepted": True, "reached_load_factor": 1.,
             "maximum_principal_tensile_stress_pa": tension}
            for index, tension in ((1, 1.9e6), (2, 1.7e6))]
        manifest, report, calls, _ = self.run_workflow({
            "maximum_principal_tensile_stress_pa": 1.7e6, "growth_source_rows": [7],
            "continuation_rounds": rounds, "continuation_stop_reason": "iteration_limit",
        }, residual_target=500000.)
        self.assertEqual(calls, 1)
        self.assertEqual(report["iterations"], rounds)
        self.assertEqual(report["adaptive_rounds_completed"], 2)
        self.assertEqual(manifest["adaptive_round"], 2)
        self.assertEqual(report["stop_reason"], "iteration_limit")
        self.assertFalse(report["engineering_goal"]["passed"])
        self.assertNotIn("adaptive_growth_warning", report)

    def test_target_reached_by_continuation_is_completed(self):
        manifest, report, _, _ = self.run_workflow({
            "maximum_principal_tensile_stress_pa": 400000.,
            "continuation_rounds": [{"iteration": 1}, {"iteration": 2}],
            "continuation_stop_reason": "residual_target_reached",
        }, residual_target=500000.)
        self.assertEqual(manifest["status"], "completed_not_engineering_validated")
        self.assertTrue(report["engineering_goal"]["passed"])
        self.assertEqual(report["adaptive_rounds_completed"], 2)

    def test_only_full_load_below_target_is_completed(self):
        manifest, report, _, _ = self.run_workflow({})
        self.assertEqual(manifest["status"], "completed_not_engineering_validated")
        self.assertTrue(report["engineering_goal"]["passed"])
        manifest, report, _, _ = self.run_workflow({"maximum_principal_tensile_stress_pa": 2.})
        self.assertEqual(manifest["status"], "partial_not_engineering_validated")
        self.assertEqual(report["stop_reason"], "residual_tension_target_not_met")

    def test_null_fracture_overrides_use_solver_defaults(self):
        manifest, report, _, _ = self.run_workflow({})
        self.assertEqual(manifest["status"], "completed_not_engineering_validated")
        self.assertEqual(report["engineering_goal"]["whole_wall_tension_tolerance_pa"], 1.)

    def test_target_met_takes_precedence_over_further_eligible_growth(self):
        manifest, report, _, _ = self.run_workflow({
            "growth_source_rows": [7], "maximum_principal_tensile_stress_pa": 2_500_000.,
        }, fracture_debug={"aster": {"whole_wall_tension_tolerance_pa": 3_000_000.}})
        self.assertEqual(manifest["status"], "completed_not_engineering_validated")
        self.assertTrue(report["engineering_goal"]["passed"])
        self.assertFalse(report["engineering_goal"]["no_unjointed_tensile_growth"])

    def test_file_target_overrides_legacy_assessment_setting(self):
        manifest, report, _, _ = self.run_workflow({
            "maximum_principal_tensile_stress_pa": 400000.,
        }, fracture_debug={"aster": {"whole_wall_tension_tolerance_pa": 1.}},
            residual_target=500000.)
        self.assertEqual(manifest["status"], "completed_not_engineering_validated")
        self.assertEqual(report["engineering_goal"]["residual_target_tensile_stress_pa"], 500000.)


if __name__ == "__main__":
    unittest.main()
