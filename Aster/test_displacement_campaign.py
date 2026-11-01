import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Aster import displacement_campaign as campaign
from Aster import run_case


class CampaignTests(unittest.TestCase):
    def test_bounded_initial_matrix_and_restoration_controls(self):
        cases = campaign.initial_cases()
        self.assertLess(len(cases), campaign.MAX_LAUNCHES - campaign.RESERVED)
        self.assertEqual(len({c.name for c in cases}), len(cases))
        concrete = {"density": 2400, "youngs_modulus": 35e9, "poisson_ratio": .2,
                    "tensile_strength_pa": .8e6, "compressive_strength_pa": 30e6,
                    "cohesive_fracture_energy_j_m2": 100}
        for case in cases:
            properties = case.properties(concrete)
            self.assertEqual(properties["density"], 2400)
        controls = campaign.confirmations(campaign.Case("target", e_gpa=1, ft_mpa=.02, gf=1))
        self.assertEqual(len(controls), 5)
        self.assertTrue(controls[1].bonded)
        with self.assertRaises(ValueError):
            campaign.Case("invalid", e_gpa=0).properties(concrete)

    def test_threshold_uses_converged_physical_magnitude_not_nonconvergence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run").mkdir()
            campaign.write_json(root / "run" / "campaign_steps.json",
                [{"load_factor": .98, "maximum_wall_displacement_m": .099},
                 {"load_factor": 1., "maximum_wall_displacement_m": .100}])
            case = campaign.Case("case")
            result = campaign.summarize_case(root, case, {"status": "completed", "returncode": 0})
            self.assertEqual(result["status"], "converged_at_or_above_100mm")
            self.assertEqual(result["crossing_bracket"], [.98, 1.])
            unresolved = campaign.summarize_case(root, case, {"status": "failed", "returncode": 1})
            self.assertEqual(unresolved["status"], "nonconverged_unresolved")
            self.assertEqual(unresolved["maximum_wall_displacement_m"], .1)

    def test_configure_capture_fixed_load_and_refuse_changed_renderer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = ("TARGETS = [0.5, 1]\nsave()\nEXCIT = (\n)\n"
                    "for target in TARGETS:\n"
                    "    reached = target\n"
                    "    if True and target >= 1.0 and np.any([]):\n        break\n")
            (root / "case.comm").write_text(text)
            campaign.write_json(root / "case_inputs.json", {"load_factors": [.5, 1.]})
            np.savez(root / campaign.native.SPLIT_ARRAYS_NAME, nodes=np.zeros((4, 3)),
                     source_node_ids=np.arange(4), tetra4_concrete=np.array([[0, 1, 2, 3]]),
                     concrete_body_ids=np.array([1]))
            campaign.configure_case(root, campaign.Case("fine", fine=True))
            configured = (root / "case.comm").read_text()
            self.assertIn("campaign_capture(RESU)", configured)
            self.assertIn("if False and target >= 1.0", configured)
            factors = json.loads((root / "case_inputs.json").read_text())["load_factors"]
            self.assertEqual(factors[0], .01)
            self.assertEqual(factors[-1], 1.)
            self.assertLessEqual(np.diff(factors).max(), .02000001)
            with self.assertRaises(ValueError):
                campaign.configure_case(root, campaign.Case("second"))

    def test_no_wall_clock_mode_removes_only_outer_timeout(self):
        with patch("Aster.whole_model_crack.linux_command", return_value=["guard"]) as command, \
                patch.object(run_case, "run_solver", return_value=subprocess.CompletedProcess([], 0)) as solve:
            run_case.run_wsl_solver("Ubuntu", "/aster", Path("."), ["run_aster"], None, Path("log"))
            self.assertEqual(command.call_args.args[-1][-1], "run_aster")
            self.assertNotIn("timeout", command.call_args.args[-1])
            self.assertIsNone(solve.call_args.args[2])
            run_case.run_wsl_solver("Ubuntu", "/aster", Path("."), ["run_aster"], 30, Path("log"))
            self.assertIn("timeout", command.call_args.args[-1])
            self.assertEqual(solve.call_args.args[2], 60)

    def test_saved_job_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp) / "jobs" / "job"
            job.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "outside application job"):
                campaign.run_campaign(job, job / "sweep", preflight_only=True, execute=False,
                                      case_time_limit=30, minimum_free_gb=1)

    def test_source_hash_change_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "baseline"
            path.write_text("original")
            hashes = {str(path): campaign.native.file_hash(path)}
            campaign.verify_hashes(hashes)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "reference changed"):
                campaign.verify_hashes(hashes)


if __name__ == "__main__":
    unittest.main()
