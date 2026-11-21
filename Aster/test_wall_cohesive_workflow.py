import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Aster import wall_cohesive_workflow as workflow


class AdaptiveGrowthTests(unittest.TestCase):
    def run_workflow(self, candidate_sets):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = root / "config.json"
        config.write_text(json.dumps({"fracture_debug": {"aster": {"adaptive_rounds": 5}}}), encoding="utf-8")
        model = root / "model"
        model.mkdir()
        (model / "three_body.msh").write_text("mesh", encoding="utf-8")
        preparation_calls = 0

        def prepare(_config, _model, _baseline, output, **kwargs):
            nonlocal preparation_calls
            output.mkdir()
            if kwargs.get("screening") is None:
                workflow.write_json(output / "screening.json", {"overstressed_source_rows": [1]})
                workflow.write_json(output / "manifest.json", {})
                return {}
            faces = candidate_sets[preparation_calls]
            preparation_calls += 1
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
            round_number = int(output.name.rsplit("_", 1)[-1])
            return {
                "status": "partial" if round_number == 1 else "completed",
                "accepted": True, "solver": "Code_Aster",
                "reached_load_factor": .9128125 if round_number == 1 else 1.,
                "opening_assessment": {"peak_opening_m": .0001},
                "paraview": {"directory": str(output)},
                "growth_source_rows": [7] if round_number == 1 else [],
                "maximum_principal_tensile_stress_pa": 0.,
                "capacity_assessment": {"message": "Newton did not converge"} if round_number == 1 else {},
            }

        with patch.object(workflow.automatic_preparation, "run", side_effect=prepare), \
                patch.object(workflow.automatic_cohesive, "run", side_effect=solve) as native:
            manifest = workflow.run(config=config, model=model, baseline=root / "baseline",
                                    output=root / "output", job_id="fixture", time_limit_s=30)
        run_dir = next((root / "output" / workflow.OUTPUT_ROOT).iterdir())
        report = json.loads((run_dir / workflow.REPORT_NAME).read_text(encoding="utf-8"))
        return manifest, report, native.call_count, run_dir

    def test_no_new_facet_stops_before_repeating_the_solve(self):
        manifest, report, calls, run_dir = self.run_workflow([
            [(1, 2, 3), (4, 5, 6)], [(6, 4, 5), (3, 1, 2)]])
        self.assertEqual(calls, 1)
        self.assertEqual(manifest["adaptive_round"], 1)
        self.assertEqual(report["adaptive_rounds_completed"], 1)
        self.assertEqual(manifest["status"], "partial_not_engineering_validated")
        self.assertEqual(manifest["reached_load_factor"], .9128125)
        self.assertEqual(manifest["remaining_growth_source_rows"], 1)
        self.assertIn("Newton did not converge", manifest["warning"])
        self.assertIn("no new usable cohesive facets", manifest["warning"])
        self.assertIn("no new usable cohesive facets", report["adaptive_growth_warning"])
        self.assertFalse((run_dir / "solve_round_2").exists())

    def test_new_facet_runs_next_round_even_when_count_is_unchanged(self):
        manifest, report, calls, _ = self.run_workflow([[(1, 2, 3)], [(4, 5, 6)]])
        self.assertEqual(calls, 2)
        self.assertEqual(manifest["adaptive_round"], 2)
        self.assertEqual(report["adaptive_rounds_completed"], 2)
        self.assertEqual(manifest["reached_load_factor"], 1.)
        self.assertNotIn("adaptive_growth_warning", report)

    def test_removing_facets_without_adding_any_is_not_growth(self):
        manifest, report, calls, _ = self.run_workflow([[(1, 2, 3), (4, 5, 6)], [(1, 2, 3)]])
        self.assertEqual(calls, 1)
        self.assertEqual(manifest["candidate_faces"], 2)
        self.assertEqual(report["adaptive_rounds_completed"], 1)
        self.assertEqual(manifest["status"], "partial_not_engineering_validated")


if __name__ == "__main__":
    unittest.main()
