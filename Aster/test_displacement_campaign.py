import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Aster import displacement_campaign as campaign
from Aster import run_case


class CampaignTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "Linux path and child environment contract")
    def test_linux_task_paths_and_backend_environment_are_isolated(self):
        before = dict(os.environ)
        with patch.object(run_case, "bundled_python_for", return_value=Path("/aster/bin/python3")):
            command = run_case.native_command("/aster/bin/run_aster", ["/aster/bin/python3", "-V"])
        self.assertEqual(command[0], "env")
        self.assertIn("LD_LIBRARY_PATH=/aster/lib:/aster/lib64", command)
        self.assertEqual(dict(os.environ), before)
        arguments = ["sweep", "--job", r"C:\jobs\reference", "--output", r"C:\sweeps",
                     "--constitutive-gate", r"C:\gate\gate.json", "--execute",
                     "--accept-native-cpu-guard", "--new-output"]
        with patch.object(sys, "argv", arguments), patch.object(campaign, "run_campaign") as execute:
            campaign.main()
        self.assertEqual(str(execute.call_args.args[0]), "/mnt/c/jobs/reference")
        self.assertEqual(str(execute.call_args.args[1].parent), "/mnt/c/sweeps")
        self.assertEqual(str(execute.call_args.kwargs["constitutive_gate"]), "/mnt/c/gate/gate.json")

    def test_constitutive_gate_checks_same_point_traction_and_irreversibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            steps = root / campaign.native.STEPS_NAME
            points = root / campaign.native.STEPS_NAME.replace(".npz", "_pointwise.npz")
            state = np.array([[0.], [1.], [2.], [2.]])
            jump = np.array([[1e-10], [1e-6], [.001], [.0005]])
            np.savez(steps, damage_state=state, normal_jump_m=jump)
            gate = root / "gate.json"

            def write_gate(traction):
                np.savez(points, state=state, normal_jump_m=jump, normal_traction_pa=traction)
                campaign.write_json(gate, {"status": "passed", "unloading_retains_separation": True,
                    "native_status": {"returncode": 0},
                    "evidence": {str(p): campaign.native.file_hash(p) for p in (steps, points)}})

            write_gate(np.array([[1.], [8e5], [0.], [0.]]))
            self.assertEqual(campaign.validate_constitutive_gate(gate)["sha256"], campaign.native.file_hash(gate))
            snapshot = campaign.materialize_constitutive_gate(gate, root / "snapshot")
            self.assertTrue(snapshot.is_file())
            campaign.validate_constitutive_gate(snapshot)
            points.write_bytes(b"Changed original evidence")
            campaign.validate_constitutive_gate(snapshot)
            with self.assertRaisesRegex(ValueError, "reference changed"):
                campaign.validate_constitutive_gate(gate)
            write_gate(np.array([[1.], [8e5], [500.], [0.]]))
            with self.assertRaisesRegex(ValueError, "zero traction"):
                campaign.validate_constitutive_gate(gate)
            state[-1] = 1.
            write_gate(np.array([[1.], [8e5], [0.], [0.]]))
            with self.assertRaisesRegex(ValueError, "irreversibility"):
                campaign.validate_constitutive_gate(gate)

    def test_progress_log_appends_timestamped_case_and_failure_events(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sweep.log"
            path.write_text("Existing investigation\n")
            campaign.log_progress("CASE_START", path=path, case="test", cap=3000)
            campaign.log_progress("BLOCKED", path=path, error="No uplift\nNo crack water")
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], "Existing investigation")
            self.assertRegex(lines[1], r"^\d{4}-\d{2}-\d{2}T.*\+00:00 CASE_START ")
            self.assertEqual(json.loads(lines[1].split(" ", 2)[2])["cap"], 3000)
            self.assertEqual(json.loads(lines[2].split(" ", 2)[2])["error"], "No uplift\nNo crack water")

    def test_missing_historical_reference_is_logged_before_solver_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "output"
            with patch.object(campaign, "log_progress") as progress, \
                    patch.object(campaign, "validate_reference", side_effect=ValueError("Missing reference")) as reference, \
                    patch.object(campaign.native, "solve") as solve:
                with self.assertRaisesRegex(ValueError, "Missing reference"):
                    campaign.run_campaign(Path(temp) / "jobs" / "job", root,
                        preflight_only=False, execute=True, case_time_limit=30,
                        minimum_free_gb=1, constitutive_gate=Path(temp) / "gate.json")
                reference.assert_called_once()
                solve.assert_not_called()
                self.assertEqual(json.loads((root / "campaign.json").read_text())["status"], "blocked")
                self.assertIn("BLOCKED", [call.args[0] for call in progress.call_args_list])
                self.assertFalse((root / "campaign.lock").exists())

    def test_hydraulic_controls_compare_against_new_intact_not_saved_cracking_result(self):
        controls = campaign.initial_cases(hydraulic=True)
        self.assertEqual(controls[0], campaign.Case("A1-reproduction"))
        self.assertEqual(controls[0].ft_mpa, 2.)
        self.assertIn(campaign.Case("A4-historical-material-full-load", e_gpa=10, nu=.15, ft_mpa=.8, gf=150), controls)

    def test_bounded_initial_matrix_and_restoration_controls(self):
        cases = campaign.initial_cases()
        self.assertLess(len(cases), campaign.MAX_LAUNCHES - campaign.RESERVED)
        self.assertEqual(len({c.name for c in cases}), len(cases))
        concrete = {"density": 2400, "youngs_modulus": 35e9, "poisson_ratio": .2,
                    "tensile_strength_pa": 2e6, "compressive_strength_pa": 30e6,
                    "cohesive_fracture_energy_j_m2": 100}
        for case in cases:
            properties = case.properties(concrete)
            self.assertEqual(properties["density"], 2400)
        controls = campaign.confirmations(campaign.Case("target", e_gpa=1, ft_mpa=.02, gf=1))
        self.assertEqual(len(controls), 5)
        self.assertTrue(controls[1].bonded)
        self.assertIn(campaign.Case("F-restore-ft_mpa", e_gpa=1, ft_mpa=2., gf=1, fine=True), controls)
        self.assertEqual(campaign.Case("intact-strength", ft_mpa=2.).properties(concrete)["tensile_strength_pa"], 2e6)
        with self.assertRaises(ValueError):
            campaign.Case("invalid", e_gpa=0).properties(concrete)
        with self.assertRaises(ValueError):
            campaign.Case("invalid", ft_mpa=2.1).properties(concrete)

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

    def test_full_sweep_cannot_bypass_native_constitutive_gate(self):
        with self.assertRaisesRegex(ValueError, "constitutive separation"):
            campaign.run_campaign(Path("job"), Path("output"), preflight_only=False, execute=True,
                              case_time_limit=30, minimum_free_gb=1)

    def test_parameter_refinement_keeps_other_axes_fixed_and_reserve_bounded(self):
        case = campaign.Case("hit", e_gpa=1, ft_mpa=.02, gf=1)
        below = campaign.Case("below", e_gpa=5, ft_mpa=.02, gf=1)
        controls = campaign.confirmations(case, [{"case": campaign.asdict(below),
                                            "status": "converged_below_100mm"}])
        self.assertEqual(len(controls), campaign.RESERVED)
        for refinement in controls[-3:]:
            self.assertGreater(refinement.e_gpa, 1)
            self.assertLess(refinement.e_gpa, 5)
            self.assertEqual(refinement.ft_mpa, .02)
            self.assertEqual(refinement.gf, 1)


if __name__ == "__main__":
    unittest.main()
