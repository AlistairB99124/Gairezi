import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Aster import wall_cohesive_workflow as workflow


class AdaptiveGrowthTests(unittest.TestCase):
    def run_workflow(self, result, fracture_debug=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = root / "config.json"
        config.write_text(json.dumps({"fracture_debug": fracture_debug}), encoding="utf-8")
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

    def test_growth_stops_before_losing_accepted_fracture_history(self):
        manifest, report, calls, run_dir = self.run_workflow({
            "growth_source_rows": [7], "maximum_principal_tensile_stress_pa": 2.})
        self.assertEqual(calls, 1)
        self.assertEqual(manifest["status"], "partial_not_engineering_validated")
        self.assertEqual(report["stop_reason"], "fracture_history_transfer_unavailable")
        self.assertIn("fracture history", report["adaptive_growth_warning"])
        self.assertFalse((run_dir / "solve_round_2").exists())

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


if __name__ == "__main__":
    unittest.main()
