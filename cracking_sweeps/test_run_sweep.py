"""No solver launches: scenario invariants, native handoff and result acceptance."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cracking_sweeps import run_sweep as sweep


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.config = sweep.read_json(sweep.ROOT / "config.json")
        # Execution mocks below deliberately supply dry reports; wet validation has its own test.
        self.config["damage_selection"].pop("crack_water_pressure", None)
        self.specification = sweep.read_json(Path(__file__).with_name("scenarios.json"))

    def test_all_cases_change_only_declared_block_properties(self):
        original = deepcopy(self.config)
        cases = sweep.define_cases(self.config, self.specification)
        self.assertEqual(len(cases), 21)
        self.assertEqual(cases[0]["config"], original)
        self.assertEqual(cases[0]["block"], "damaged_element_properties+lift_joints")
        blocks = ("lift_joints", "damaged_element_properties")
        for case in cases:
            restored = deepcopy(case["config"])
            for block in blocks:
                before = original["damage_selection"][block]
                after = restored["damage_selection"][block]
                factors = case["factors"].get(block, {})
                for key, factor in factors.items():
                    if block == "lift_joints" and key in sweep.LIFT_JOINT_ABSOLUTE:
                        self.assertEqual(after[key], factor)
                    else:
                        self.assertEqual(after[key], before[key] * factor)
                for key in before:
                    if key not in factors:
                        self.assertEqual(after[key], before[key])
                restored["damage_selection"][block] = deepcopy(before)
            self.assertEqual(restored, original)
            if case["factors"]:
                self.assertEqual(case["block"], "+".join(sorted(case["factors"])))
        self.assertEqual(self.config, original)
        # The shipped baseline degrades the band's elements and weakens its lift planes.
        self.assertIn("damaged_element_properties", original["damage_selection"])
        self.assertIn("lift_joints", original["damage_selection"])

    def test_friction_coefficient_is_an_absolute_value_for_a_frictional_baseline(self):
        specification = dict(self.specification, cases=[
            {"id": "A00-baseline", "factors": {}},
            {"id": "H01-mu05", "factors": {"lift_joints": {"friction_coefficient": 0.5,
                                                           "tensile_strength_pa": 0.5}}}])
        with self.assertRaisesRegex(ValueError, "JOINT_MECA_FROT"):
            sweep.define_cases(self.config, specification)
        frictional = deepcopy(self.config)
        frictional["damage_selection"]["lift_joints"].update(law="JOINT_MECA_FROT", friction_coefficient=0.7)
        cases = sweep.define_cases(frictional, specification)
        varied = cases[1]["config"]["damage_selection"]["lift_joints"]
        self.assertEqual(varied["friction_coefficient"], 0.5)
        self.assertEqual(varied["tensile_strength_pa"],
                         0.5 * frictional["damage_selection"]["lift_joints"]["tensile_strength_pa"])
        specification["cases"][1]["factors"]["lift_joints"]["friction_coefficient"] = 11
        with self.assertRaisesRegex(ValueError, "friction_coefficient must lie"):
            sweep.define_cases(frictional, specification)

    def legacy_config(self) -> dict:
        """A degraded-element-only baseline (the pre-lift-joint sweep contract)."""
        config = deepcopy(self.config)
        config["damage_selection"].pop("lift_joints")
        config["damage_selection"]["damaged_element_properties"] = {
            **config["material"]["concrete"], "youngs_modulus": 35e6}
        return config

    def lift_only_config(self) -> dict:
        config = deepcopy(self.config)
        config["damage_selection"].pop("damaged_element_properties")
        return config

    def test_single_block_baselines_accept_flat_factors(self):
        config = self.legacy_config()
        specification = deepcopy(self.specification)
        specification["cases"] = [{"id": "A00", "factors": {}},
                                  {"id": "D01", "factors": {"youngs_modulus": .5, "tensile_strength_pa": .25}}]
        cases = sweep.define_cases(config, specification)
        self.assertEqual([case["block"] for case in cases], ["damaged_element_properties"] * 2)
        after = cases[1]["config"]["damage_selection"]["damaged_element_properties"]
        self.assertEqual(after["youngs_modulus"], 17.5e6)
        self.assertEqual(after["tensile_strength_pa"], .25 * config["material"]["concrete"]["tensile_strength_pa"])
        # Variables of the other block are rejected for a single-block baseline.
        specification["cases"][1]["factors"] = {"alpha": 1.}
        with self.assertRaisesRegex(ValueError, "damaged_element_properties"):
            sweep.define_cases(config, specification)
        specification["cases"][1]["factors"] = {"youngs_modulus": .5}
        with self.assertRaisesRegex(ValueError, "lift_joints"):
            sweep.define_cases(self.lift_only_config(), specification)
        specification["cases"][1]["factors"] = {"tensile_strength_pa": .5, "alpha": 0.}
        lift_cases = sweep.define_cases(self.lift_only_config(), specification)
        self.assertEqual(lift_cases[1]["block"], "lift_joints")
        self.assertEqual(lift_cases[1]["config"]["damage_selection"]["lift_joints"]["alpha"], 0.)
        # With both blocks configured, flat factors are ambiguous and must be keyed by block.
        specification["cases"][1]["factors"] = {"tensile_strength_pa": .5}
        with self.assertRaisesRegex(ValueError, "key the factors by block"):
            sweep.define_cases(self.config, specification)
        specification["cases"][1]["factors"] = {"lift_joints": {"tensile_strength_pa": .5}, "alpha": 0.}
        with self.assertRaisesRegex(ValueError, "mix"):
            sweep.define_cases(self.config, specification)
        specification["cases"][1]["factors"] = {"damaged_element_properties": {"youngs_modulus": .5}}
        with self.assertRaisesRegex(ValueError, "has no damaged_element_properties"):
            sweep.define_cases(self.lift_only_config(), specification)

    def test_base_band_selects_cells_even_below_intact_tensile_threshold(self):
        import numpy as np
        from FEniCSx.automatic_screening import screen_cells

        tetrahedron = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        coordinates = np.vstack((tetrahedron, tetrahedron + [0., 0., 3.]))
        cells = np.arange(8).reshape(2, 4)
        stress = np.array([np.diag([1.5e6, 0., 0.])] * 2)
        results = [screen_cells(stress, np.ones(2, dtype=int), np.arange(2), coordinates,
                                cells, strength, relative_tolerance=.001, crack_base=True)
                   for strength in (2e6, 1e6)]
        self.assertEqual(results[0]["status"], "no_overstressed_cells")
        self.assertEqual(results[1]["status"], "overstressed_cells_found")
        for result in results:
            self.assertEqual(result["selection"]["selected_source_rows"], [0])

    def test_bad_definitions_and_unsupported_baselines_are_rejected(self):
        L = "lift_joints"
        for key, value in (("id", "../escape"), ("factors", {L: {"density": .5}}),
                           ("factors", {L: {"youngs_modulus": .5}}),
                           ("factors", {L: {"tensile_strength_pa": 0}}),
                           ("factors", {L: {"tensile_strength_pa": float("nan")}}),
                           ("factors", {L: {"tensile_strength_pa": True}}),
                           ("factors", {L: {"tensile_strength_pa": 2}}),
                           ("factors", {L: {"alpha": 2.5}}), ("factors", {L: {"alpha": -1}}),
                           ("factors", {L: {"alpha": True}}), ("factors", {L: 0.5}),
                           ("factors", {"damaged_element_properties": {"alpha": 1.}}),
                           ("factors", {"damaged_element_properties": {"youngs_modulus": 2}})):
            with self.subTest(key=key, value=value):
                specification = deepcopy(self.specification)
                specification["cases"][1][key] = value
                with self.assertRaises(ValueError):
                    sweep.define_cases(self.config, specification)
        for field, value in (("maximum_overload_factor", 2), ("enabled", False),
                             ("damaged_joint_friction", {"friction_coefficient": 0, "adhesion_pa": 0})):
            with self.subTest(field=field):
                config = deepcopy(self.config)
                config["damage_selection"][field] = value
                with self.assertRaises(ValueError):
                    sweep.define_cases(config, self.specification)
        specification = deepcopy(self.specification)
        specification["cases"][1]["id"] = specification["cases"][0]["id"]
        with self.assertRaises(ValueError):
            sweep.define_cases(self.config, specification)
        specification = deepcopy(self.specification)
        specification["cases"][0]["factors"] = {"tensile_strength_pa": .5}
        with self.assertRaises(ValueError):
            sweep.define_cases(self.config, specification)

    @staticmethod
    def report(opening=.002):
        return {
            "accepted": True, "status": "completed", "solve_status": "completed",
            "returncode": 0, "assessment_authority": "code_aster", "analysis_scope": "whole_model",
            "reached_load_factor": 1.,
            "maximum_wall_displacement_m": .01,
            "steps": [{"load_factor": 1., "peak_opening_m": opening, "separated_faces": 2}],
        }

    def test_target_uses_unrounded_normal_opening_at_full_load(self):
        self.assertTrue(sweep.classify_report(self.report(.002))["target_reached"])
        self.assertTrue(sweep.classify_report(self.report(.003))["target_reached"])
        self.assertFalse(sweep.classify_report(self.report(math.nextafter(.002, 0)))["target_reached"])
        actual = sweep.classify_report(self.report(1.1017025871864025e-7))
        self.assertEqual(actual["status"], "converged_below_target")
        self.assertAlmostEqual(actual["peak_normal_opening_mm"], .00011017025871864025)
        for opening in (float("nan"), float("inf"), -.1, None, "0.002", True):
            with self.subTest(opening=opening):
                self.assertFalse(sweep.classify_report(self.report(opening))["target_reached"])

    def test_partial_failed_unaccepted_or_wrong_authority_never_reaches_target(self):
        for key, value in (("accepted", False), ("status", "partial"),
                           ("solve_status", "stopped"), ("returncode", 2),
                           ("assessment_authority", "fenicsx"), ("analysis_scope", "submodel"),
                           ("reached_load_factor", .9), ("reached_load_factor", 1.1),
                           ("steps", []),
                           ("steps", [{"load_factor": .9, "peak_opening_m": .01}]),
                           ("steps", [{"load_factor": 1.1, "peak_opening_m": .01}])):
            with self.subTest(key=key, value=value):
                report = self.report(.01)
                report[key] = value
                self.assertFalse(sweep.classify_report(report)["target_reached"])
        for malformed in ([], {"steps": "invalid"}, {"steps": [None]}):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                sweep.classify_report(malformed)

    def test_derivative_handoff_preserves_selection_baseline_hashes_and_supports(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            prepared = root / "prepared"
            prepared.mkdir()
            baseline_config = root / "config.json"
            sweep.write_json(baseline_config, self.config)
            (prepared / "cohesive_state.npz").write_bytes(b"fixed physical facets")
            summary = {
                "status": "prepared_aster_candidates",
                "config_digest": "original-solved-baseline",
                "screening_sha256": "fixed-screening",
                "baseline_sha256": {"mesh_manifest.json": "original-mesh-hash"},
                "support_degradation": {"interface_scale": .01},
                "material_assignment": {"joint_properties": {"lift_joint_facets": 4, "law": "JOINT_MECA_RUPT"},
                                        "properties": {"placeholder": True},
                                        "selected_source_rows": [7, 9]},
            }
            sweep.write_json(prepared / "summary.json", summary)
            sweep.write_json(prepared / "manifest.json", {
                "status": "prepared_aster_candidates",
                "summary_sha256": sweep.file_hash(prepared / "summary.json"),
                "candidate_state_sha256": sweep.file_hash(prepared / "cohesive_state.npz"),
            })
            originals = {path: sweep.file_hash(path) for path in prepared.iterdir()}
            case = sweep.define_cases(self.config, self.specification)[-1]
            directory = root / case["id"]
            candidate = sweep.materialize_case(case, directory, prepared, baseline_config)
            derivative = sweep.read_json(candidate / "summary.json")
            manifest = sweep.read_json(candidate / "manifest.json")
            self.assertEqual(derivative["config_digest"], summary["config_digest"])
            self.assertEqual(derivative["baseline_sha256"], summary["baseline_sha256"])
            self.assertEqual(derivative["material_assignment"]["selected_source_rows"], [7, 9])
            self.assertEqual(derivative["support_degradation"], summary["support_degradation"])
            joints = case["config"]["damage_selection"]["lift_joints"]
            recorded = derivative["material_assignment"]["joint_properties"]
            self.assertEqual(recorded["lift_joint_facets"], 4)
            for key in ("tensile_strength_pa", "cohesive_fracture_energy_j_m2", "alpha"):
                self.assertEqual(recorded[key], joints[key])
            self.assertAlmostEqual(recorded["critical_opening_m"],
                                   2 * joints["cohesive_fracture_energy_j_m2"] / joints["tensile_strength_pa"])
            # The degraded material travels with the derivative and drives the damaged facets' CZM.
            elements = case["config"]["damage_selection"]["damaged_element_properties"]
            self.assertEqual(derivative["material_assignment"]["properties"], elements)
            self.assertEqual(derivative["tensile_strength_pa"], elements["tensile_strength_pa"])
            self.assertEqual(derivative["fracture_energy_j_m2"], elements["cohesive_fracture_energy_j_m2"])
            self.assertEqual(manifest["summary_sha256"], sweep.file_hash(candidate / "summary.json"))
            self.assertEqual(manifest["candidate_state_sha256"], sweep.file_hash(candidate / "cohesive_state.npz"))
            self.assertEqual({path: sweep.file_hash(path) for path in prepared.iterdir()}, originals)
            self.assertEqual(sweep.read_json(baseline_config), self.config)
            with self.assertRaises(FileExistsError):
                sweep.materialize_case(case, directory, prepared, baseline_config)
            # A derivative configuring fewer blocks than the candidates were prepared with is refused.
            legacy = sweep.define_cases(self.legacy_config(), {**self.specification, "cases": [
                {"id": "A00", "factors": {}}]})[0]
            with self.assertRaisesRegex(ValueError, "lift joints the derivative no longer configures"):
                sweep.materialize_case(legacy, root / "legacy", prepared, baseline_config)
            lift_only = sweep.define_cases(self.lift_only_config(), {**self.specification, "cases": [
                {"id": "A00", "factors": {}}]})[0]
            with self.assertRaisesRegex(ValueError, "degraded elements the derivative no longer configures"):
                sweep.materialize_case(lift_only, root / "lift_only", prepared, baseline_config)

    def test_native_prepare_accepts_derivative_provenance_without_solvers(self):
        import numpy as np
        from Aster import automatic_preparation as prep
        from Aster import automatic_cohesive as native
        from Aster.test_automatic_preparation import AutomaticPreparationTests
        from Aster.test_automatic_cohesive import column_fixture, fixture_manifest, fixture_base_comm

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model, baseline, config_path = AutomaticPreparationTests().fixture(root)
            config = sweep.read_json(config_path)
            config["damage_selection"] = {
                "enabled": True, "crack_base": True,
                "damaged_element_properties": self.config["damage_selection"]["damaged_element_properties"],
                "lift_joints": self.config["damage_selection"]["lift_joints"],
                "support_degradation": {"interface_scale": .01},
            }
            sweep.write_json(config_path, config)
            arrays = column_fixture()
            arrays["source_cell_rows"] = np.arange(len(arrays["tetra4_concrete"]) + len(arrays["tetra4_bedrock"]))
            np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
            manifest = fixture_manifest(arrays)
            manifest.update(source_mesh=str(model / "three_body.msh"), source_config=str(config_path),
                            source_hashes={key: sweep.file_hash(model / key) for key in ("three_body.msh", "config.json")})
            sweep.write_json(baseline / "mesh_manifest.json", manifest)
            with patch("subprocess.Popen", side_effect=AssertionError("No solvers permitted")):
                prep.run(config_path, model, baseline, root / "screening", job_id="fixture", relative_tolerance=.001)
                prepared = root / "prepared"
                prep.run(config_path, model, baseline, prepared, job_id="fixture",
                         relative_tolerance=.001, screening=root / "screening")
                derivative = deepcopy(config)
                derivative["damage_selection"]["lift_joints"] = dict(
                    config["damage_selection"]["lift_joints"],
                    tensile_strength_pa=5e4, cohesive_fracture_energy_j_m2=.3125, alpha=0.5)
                derivative["damage_selection"]["damaged_element_properties"] = dict(
                    config["damage_selection"]["damaged_element_properties"],
                    youngs_modulus=3.5e9, tensile_strength_pa=4e5)
                candidate = sweep.materialize_case({"id": "test", "factors": {}, "config": derivative,
                                                    "block": "damaged_element_properties+lift_joints"},
                                                   root / "case", prepared, config_path)
                def render(root=None, mesh_manifest_path=None):
                    return fixture_base_comm(json.loads(mesh_manifest_path.read_text()), 4e3), {
                        "material": {"concrete": {"youngs_modulus_pa": 30e9, "poissons_ratio": .2}}}
                with patch("Aster.generate_three_body_case.render_comm", side_effect=render):
                    inputs = native.prepare(baseline, candidate, root / "native")
            # The damaged facets' CZM follows the derivative's degraded material, the intact CZM
            # keeps the concrete values and the lift law carries the derivative's joint values.
            self.assertEqual(inputs["joint"]["tensile_strength_pa"], 4e5)
            self.assertEqual(inputs["joint"]["intact_joint"]["tensile_strength_pa"],
                             config["material"]["concrete"]["tensile_strength_pa"])
            self.assertEqual(inputs["joint"]["lift_joint"]["tensile_strength_pa"], 5e4)
            self.assertEqual(inputs["joint"]["lift_joint"]["fracture_energy_j_m2"], .3125)
            self.assertEqual(inputs["joint"]["lift_joint"]["alpha"], .5)
            self.assertEqual(inputs["material_assignment"]["properties"]["youngs_modulus"], 3.5e9)
            comm = (root / "native" / "case.comm").read_text()
            self.assertIn("SIGM_MAX=50000, PENA_RUPTURE=", comm)
            self.assertIn("ALPHA=0.5", comm)
            self.assertIn("E=3500000000", comm)
            self.assertIn("GROUP_MA='DAMAGED_ELEMENTS', MATER=DAMAGED_MAT", comm)
            self.assertIn("GROUP_MA='LIFT_JOINTS', MATER=LIFT_MAT", comm)
            compile(comm, "sweep.comm", "exec")

    def test_execute_keeps_failed_cases_and_persists_following_target(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "cases").mkdir()
            sweep.write_json(root / "config.json", self.config)
            for directory in ("baseline", "prepared", "screening"):
                (root / directory).mkdir()
                (root / directory / "immutable").write_text("unchanged")
            cases = sweep.define_cases(self.config, self.specification)[:3]

            def materialize(case, directory, prepared, config):
                directory.mkdir()
                sweep.write_json(directory / "config.json", case["config"])
                return directory / "candidate"

            def run_module(module, arguments, log, allowed_codes):
                directory = log.parent
                if directory.name == cases[0]["id"]:
                    raise RuntimeError("deliberate worker failure")
                (directory / "aster").mkdir()
                sweep.write_json(directory / "aster" / "aster_cohesive_report.json", self.report())
                return 0

            with patch.object(sweep, "prepare_reference", return_value=(root / "baseline", root / "prepared")), \
                    patch.object(sweep, "materialize_case", side_effect=materialize), \
                    patch.object(sweep, "run_module", side_effect=run_module):
                state = sweep.execute(root, cases, candidate_root=root / "candidate")
            self.assertEqual(sweep.read_json(root / "candidate" / "config.json"), cases[1]["config"])
            self.assertEqual(state["status"], "completed_with_failures")
            self.assertEqual(state["target_cases"], [cases[1]["id"]])
            self.assertEqual(state["failed_cases"], [cases[0]["id"]])
            self.assertEqual(state["stop_reason"], "target_reached")
            self.assertEqual(state["skipped_cases"], [cases[2]["id"]])
            self.assertFalse((root / "cases" / cases[2]["id"]).exists())
            self.assertEqual(sweep.read_json(root / "sweep.json"), state)
            self.assertTrue((root / "results.csv").is_file())
            self.assertTrue((root / "cases" / cases[0]["id"] / "result.json").is_file())

    def test_requested_wet_sweep_rejects_a_dry_target_report(self):
        from Aster.hydraulic_loads import CRACK_WATER_SETTINGS
        self.config["damage_selection"]["crack_water_pressure"] = True
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "cases").mkdir()
            for directory in ("baseline", "prepared", "screening"):
                (root / directory).mkdir()
            sweep.write_json(root / "config.json", self.config)
            cases = sweep.define_cases(self.config, self.specification)[:3]

            def materialize(case, directory, prepared, config):
                directory.mkdir()
                sweep.write_json(directory / "config.json", case["config"])
                return directory / "candidate"

            def run_module(module, arguments, log, allowed_codes):
                report = self.report(.003)
                if log.parent.name != cases[0]["id"]:
                    report.update(
                        hydraulic_loads={"settings": CRACK_WATER_SETTINGS},
                        hydraulic_steps=[{
                            "load_factor": 1., "active_set_stable": True, "iterations": 2, "wet_faces": 1,
                            "paired_resultant_n": [0., 0., 0.], "paired_moment_n_m": [0., 0., 0.],
                            "uplift_resultant_n": [0., 0., 0.], "sum_nodal_force_magnitudes_n": 100.}])
                (log.parent / "aster").mkdir()
                sweep.write_json(log.parent / "aster" / "aster_cohesive_report.json", report)
                return 0

            with patch.object(sweep, "prepare_reference", return_value=(root / "baseline", root / "prepared")), \
                    patch.object(sweep, "materialize_case", side_effect=materialize), \
                    patch.object(sweep, "run_module", side_effect=run_module):
                state = sweep.execute(root, cases, candidate_root=root / "candidate")
            self.assertEqual(state["failed_cases"], [cases[0]["id"]])
            self.assertEqual(state["target_cases"], [cases[1]["id"]])
            self.assertEqual(state["skipped_cases"], [cases[2]["id"]])
            self.assertIn("hydraulic metadata", sweep.read_json(
                root / "cases" / cases[0]["id"] / "result.json")["error"])

    def test_below_target_and_partial_large_opening_do_not_stop_the_sweep(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "cases").mkdir()
            sweep.write_json(root / "config.json", self.config)
            for directory in ("baseline", "prepared", "screening"):
                (root / directory).mkdir()
            cases = sweep.define_cases(self.config, self.specification)[:3]

            def materialize(case, directory, prepared, config):
                directory.mkdir()
                return directory / "candidate"

            def run_module(module, arguments, log, allowed_codes):
                report = self.report(.001)
                if log.parent.name == cases[1]["id"]:
                    report = self.report(.003)
                    report.update(status="partial", accepted=False, reached_load_factor=.9)
                    report["steps"][-1]["load_factor"] = .9
                (log.parent / "aster").mkdir()
                sweep.write_json(log.parent / "aster" / "aster_cohesive_report.json", report)
                return 0

            with patch.object(sweep, "prepare_reference", return_value=(root / "baseline", root / "prepared")), \
                    patch.object(sweep, "materialize_case", side_effect=materialize), \
                    patch.object(sweep, "run_module", side_effect=run_module):
                state = sweep.execute(root, cases)
            self.assertEqual(len(state["results"]), 3)
            self.assertEqual(state["target_cases"], [])
            self.assertEqual(state["stop_reason"], "cases_exhausted")
            self.assertEqual(state["skipped_cases"], [])

    def test_reference_failure_and_cancellation_remain_explicit(self):
        for error, status in ((ValueError("invalid reference"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                with patch.object(sweep, "prepare_reference", side_effect=error):
                    with self.assertRaises(type(error)):
                        sweep.execute(root, [])
                self.assertEqual(sweep.read_json(root / "sweep.json")["status"], status)


if __name__ == "__main__":
    unittest.main()
