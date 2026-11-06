"""Aster-only preparation/provenance; no native solver execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Aster import automatic_preparation as prep
from Aster.test_automatic_cohesive import column_fixture, fixture_manifest, fixture_base_comm


class AutomaticPreparationTests(unittest.TestCase):
    def test_selected_material_is_saved_and_assigned_by_exact_source_rows(self):
        from Aster import automatic_cohesive as native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config_path = self.fixture(root)
            config = json.loads(config_path.read_text())
            properties = {"density": 2100., "youngs_modulus": 12e9, "poisson_ratio": .23,
                          "tensile_strength_pa": 7e4, "compressive_strength_pa": 15e6,
                          "cohesive_fracture_energy_j_m2": 60.}
            config["damage_selection"] = {"damaged_element_properties": properties}
            prep.write_json(config_path, config)
            arrays = column_fixture()
            arrays["source_cell_rows"] = np.arange(len(arrays["tetra4_concrete"]) + len(arrays["tetra4_bedrock"]))
            np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
            manifest = fixture_manifest(arrays)
            manifest.update(source_mesh=str(model / "three_body.msh"), source_config=str(config_path),
                            source_hashes={name: prep.file_hash(model / name) for name in ("three_body.msh", "config.json")})
            prep.write_json(baseline / "mesh_manifest.json", manifest)
            screening, prepared = root / "screening", root / "prepared"
            result = prep.run(config_path, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
            self.assertEqual(result["material_assignment"]["properties"], properties)
            result = prep.run(config_path, model, baseline, prepared, job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            def render(root=None, mesh_manifest_path=None):
                return fixture_base_comm(json.loads(mesh_manifest_path.read_text()), 4e3), {
                    "material": {"concrete": {"youngs_modulus_pa": 30e9, "poissons_ratio": .2}}}
            with patch("Aster.generate_three_body_case.render_comm", side_effect=render):
                inputs = native.prepare(baseline, prepared, root / "native")
            with np.load(root / "native" / "three_body_mesh_arrays.npz") as stored:
                selected = stored["damaged_cells"]
                np.testing.assert_array_equal(selected, np.isin(stored["source_cell_rows"],
                                               result["material_assignment"]["selected_source_rows"]))
                self.assertTrue(selected.any())
                self.assertFalse(selected[len(stored["tetra4_concrete"]):].any())
            comm = (root / "native" / "case.comm").read_text()
            self.assertIn("E=12000000000", comm)
            self.assertIn("RHO=2100", comm)
            self.assertIn("GROUP_MA='DAMAGED_ELEMENTS', MATER=DAMAGED_MAT", comm)
            self.assertEqual(inputs["joint"]["tensile_strength_pa"], 7e4)
            self.assertEqual(inputs["baseline_displacement_sha256"], prep.file_hash(baseline / "aster_full_displacement.npz"))
            compile(comm, "selected.comm", "exec")

    def test_friction_joints_replace_the_damaged_cohesive_law(self):
        from Aster import automatic_cohesive as native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config_path = self.fixture(root)
            config = json.loads(config_path.read_text())
            properties = {"density": 2100., "youngs_modulus": 12e9, "poisson_ratio": .23,
                          "tensile_strength_pa": 100., "compressive_strength_pa": 15e6,
                          "cohesive_fracture_energy_j_m2": 1.}
            config["damage_selection"] = {"damaged_element_properties": properties,
                                          "damaged_joint_friction": {"friction_coefficient": 1.,
                                                                     "adhesion_pa": 0.}}
            prep.write_json(config_path, config)
            arrays = column_fixture()
            arrays["source_cell_rows"] = np.arange(len(arrays["tetra4_concrete"]) + len(arrays["tetra4_bedrock"]))
            np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
            manifest = fixture_manifest(arrays)
            manifest.update(source_mesh=str(model / "three_body.msh"), source_config=str(config_path),
                            source_hashes={name: prep.file_hash(model / name) for name in ("three_body.msh", "config.json")})
            prep.write_json(baseline / "mesh_manifest.json", manifest)
            screening, prepared = root / "screening", root / "prepared"
            prep.run(config_path, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
            result = prep.run(config_path, model, baseline, prepared, job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            self.assertEqual(result["damaged_joint_friction"]["friction_coefficient"], 1.)

            def render(root=None, mesh_manifest_path=None):
                return fixture_base_comm(json.loads(mesh_manifest_path.read_text()), 4e3), {
                    "material": {"concrete": {"youngs_modulus_pa": 30e9, "poissons_ratio": .2}}}
            with patch("Aster.generate_three_body_case.render_comm", side_effect=render):
                inputs = native.prepare(baseline, prepared, root / "native")
            friction = inputs["joint"]["damaged_friction"]
            self.assertEqual(friction["law"], "JOINT_MECA_FROT")
            self.assertEqual(friction["k_n_pa_per_m"], inputs["joint"]["initial_stiffness_pa_per_m"])
            self.assertGreater(friction["residual_tolerance_n"], 0.)
            # The CZM parameters describe the intact joints, not the strengthless damaged ones.
            self.assertEqual(inputs["joint"]["tensile_strength_pa"],
                             json.loads(config_path.read_text())["material"]["concrete"]["tensile_strength_pa"])
            comm = (root / "native" / "case.comm").read_text()
            self.assertIn("JOINT_MECA_FROT=_F(K_N=", comm)
            self.assertIn("MU=1, ADHESION=0", comm)
            self.assertIn("_F(GROUP_MA='DAMAGED_JOINTS', MATER=FROT_MAT),", comm)
            self.assertIn("_F(RELATION='JOINT_MECA_FROT', GROUP_MA='DAMAGED_JOINTS'),", comm)
            self.assertIn("FRICTION_CELLS = np.asarray(", comm)
            compile(comm, "friction.comm", "exec")

    def test_disabled_selection_prepares_a_control_with_no_candidates(self):
        from Aster import automatic_cohesive as native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config_path = self.fixture(root)
            config = json.loads(config_path.read_text())
            properties = {"density": 2100., "youngs_modulus": 12e9, "poisson_ratio": .23,
                          "tensile_strength_pa": 100., "compressive_strength_pa": 15e6,
                          "cohesive_fracture_energy_j_m2": 1.}
            config["damage_selection"] = {"enabled": False,
                                          "damaged_element_properties": properties,
                                          "damaged_joint_friction": {"friction_coefficient": 1.,
                                                                     "adhesion_pa": 0.}}
            prep.write_json(config_path, config)
            arrays = column_fixture()
            arrays["source_cell_rows"] = np.arange(len(arrays["tetra4_concrete"]) + len(arrays["tetra4_bedrock"]))
            np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
            manifest = fixture_manifest(arrays)
            manifest.update(source_mesh=str(model / "three_body.msh"), source_config=str(config_path),
                            source_hashes={name: prep.file_hash(model / name) for name in ("three_body.msh", "config.json")})
            prep.write_json(baseline / "mesh_manifest.json", manifest)
            screening, prepared = root / "screening", root / "prepared"
            result = prep.run(config_path, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
            self.assertIs(result["selection"]["enabled"], False)
            self.assertEqual(result["selection"]["selected_source_rows"], [])
            self.assertEqual(result["material_assignment"]["selected_source_rows"], [])
            result = prep.run(config_path, model, baseline, prepared, job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            self.assertEqual(result["status"], prep.PREPARED)
            self.assertEqual(result["candidate_faces"], 0)
            self.assertIn("disabled", result["candidate_basis"])

            def render(root=None, mesh_manifest_path=None):
                return fixture_base_comm(json.loads(mesh_manifest_path.read_text()), 4e3), {
                    "material": {"concrete": {"youngs_modulus_pa": 30e9, "poissons_ratio": .2}}}
            with patch("Aster.generate_three_body_case.render_comm", side_effect=render):
                inputs = native.prepare(baseline, prepared, root / "native")
            self.assertFalse(inputs["joint"]["has_joints"])
            comm = (root / "native" / "case.comm").read_text()
            self.assertNotIn("DAMAGED_JOINTS", comm)
            self.assertNotIn("JOINT_MECA_FROT=_F(", comm)
            self.assertNotIn("RELATION='JOINT_MECA_FROT'", comm)
            compile(comm, "control.comm", "exec")

    def test_support_degradation_passes_through_only_when_selection_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config_path = self.fixture(root)
            config = json.loads(config_path.read_text())
            properties = {"density": 2400., "youngs_modulus": 35e6, "poisson_ratio": .2,
                          "tensile_strength_pa": 2e6, "compressive_strength_pa": 30e6,
                          "cohesive_fracture_energy_j_m2": 100.}
            config["damage_selection"] = {"enabled": True, "damaged_element_properties": properties,
                                          "support_degradation": {"interface_scale": 0.01}}
            prep.write_json(config_path, config)
            arrays = column_fixture()
            arrays["source_cell_rows"] = np.arange(len(arrays["tetra4_concrete"]) + len(arrays["tetra4_bedrock"]))
            np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
            manifest = fixture_manifest(arrays)
            manifest.update(source_mesh=str(model / "three_body.msh"), source_config=str(config_path),
                            source_hashes={name: prep.file_hash(model / name) for name in ("three_body.msh", "config.json")})
            prep.write_json(baseline / "mesh_manifest.json", manifest)
            screening, prepared = root / "screening", root / "prepared"
            prep.run(config_path, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
            result = prep.run(config_path, model, baseline, prepared, job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            self.assertEqual(result["support_degradation"], {"interface_scale": 0.01})
            summary = json.loads((prepared / "summary.json").read_text())
            self.assertEqual(summary["support_degradation"], {"interface_scale": 0.01})
            # The control run keeps the intact supports.
            config["damage_selection"]["enabled"] = False
            prep.write_json(config_path, config)
            manifest["source_hashes"] = {name: prep.file_hash(model / name)
                                         for name in ("three_body.msh", "config.json")}
            prep.write_json(baseline / "mesh_manifest.json", manifest)
            control_screening, control = root / "control_screening", root / "control"
            prep.run(config_path, model, baseline, control_screening, job_id="fixture", relative_tolerance=1e-4)
            result = prep.run(config_path, model, baseline, control, job_id="fixture",
                              relative_tolerance=1e-4, screening=control_screening)
            self.assertNotIn("support_degradation", result)

    def test_native_hand_off_keeps_consistent_support_and_overload(self):
        from Aster import automatic_cohesive as native
        from Aster.support_trace import add_consistent_support, render_constraints
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config_path = self.fixture(root)
            config = json.loads(config_path.read_text())
            config["damage_selection"] = {"maximum_overload_factor": 2.}
            prep.write_json(config_path, config)
            arrays = column_fixture()
            arrays["source_cell_rows"] = np.arange(len(arrays["tetra4_concrete"]) + len(arrays["tetra4_bedrock"]))
            seat = arrays["tri3_rock_bottom"]
            add_consistent_support(arrays, seat, np.repeat((np.eye(3) * 1e13)[None], len(seat), axis=0))
            np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
            np.savez(baseline / "aster_full_displacement.npz", coordinates_m=arrays["nodes"],
                     source_node_ids=arrays["source_node_ids"],
                     displacement_m=arrays["nodes"] * np.array((0., 0., .001)))
            manifest = fixture_manifest(arrays)
            manifest.update(source_mesh=str(model / "three_body.msh"), source_config=str(config_path),
                            source_hashes={name: prep.file_hash(model / name)
                                           for name in ("three_body.msh", "config.json")})
            prep.write_json(baseline / "mesh_manifest.json", manifest)
            screening, prepared = root / "screening", root / "prepared"
            prep.run(config_path, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
            prep.run(config_path, model, baseline, prepared, job_id="fixture",
                     relative_tolerance=1e-4, screening=screening)

            def render(root=None, mesh_manifest_path=None):
                case_manifest = json.loads(mesh_manifest_path.read_text())
                base = fixture_base_comm(case_manifest, 4e3)
                base = base.replace("RESU = MECA_STATIQUE(", render_constraints(arrays) + "RESU = MECA_STATIQUE(")
                return base, {"material": {"concrete": {"youngs_modulus_pa": 30e9, "poissons_ratio": .2}}}

            with patch("Aster.generate_three_body_case.render_comm", side_effect=render):
                inputs = native.prepare(baseline, prepared, root / "native")
            self.assertEqual(inputs["load_factors"][-1], 2.)
            self.assertTrue(inputs["joint"]["stop_overload_on_opening"])
            self.assertIn("_F(CHARGE=SUPPORT_TRACE)", (root / "native" / "case.comm").read_text())

    def test_candidate_selection_preserves_original_source_cell_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config = self.fixture(root)
            with np.load(baseline / "three_body_mesh_arrays.npz") as stored:
                arrays = {key: stored[key] for key in stored.files}
            arrays["source_cell_rows"] = arrays["source_cell_rows"][::-1]
            np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
            screening = root / "screening"
            result = prep.run(config, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
            source_concrete = set(arrays["source_cell_rows"][:len(arrays["concrete_body_ids"])][
                np.isin(arrays["concrete_body_ids"], (1, 3))])
            self.assertTrue(set(result["selection"]["selected_source_rows"]).issubset(source_concrete))
            self.assertGreater(result["selection"]["selected_wall_cells"], 0)
            result = prep.run(config, model, baseline, root / "prepared", job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            self.assertGreater(result["candidate_faces"], 0)

    def test_no_eligible_facets_still_prepares_the_peak_layer_band_when_uncapped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config = self.fixture(root)
            with np.load(baseline / "aster_full_displacement.npz") as fields:
                values = {key: fields[key] for key in fields.files}
            values["displacement_m"] *= 0.
            np.savez(baseline / "aster_full_displacement.npz", **values)
            screening = root / "screening"
            result = prep.run(config, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
            self.assertGreater(result["selection"]["selected_cells"], 0)
            result = prep.run(config, model, baseline, root / "prepared", job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            self.assertGreater(result["candidate_faces"], 0)
            self.assertFalse(result["accepted"])
            with np.load(root / "prepared" / "cohesive_state.npz") as state:
                self.assertGreater(len(state["minus"]), 0)

    def test_capped_no_eligible_facets_prepares_native_elastic_steps_not_fake_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config = self.fixture(root)
            data = json.loads(config.read_text())
            data["damage_selection"] = {"max_damage_count": 4}
            prep.write_json(config, data)
            manifest = json.loads((baseline / "mesh_manifest.json").read_text())
            manifest["source_hashes"] = {name: prep.file_hash(model / name)
                                         for name in ("three_body.msh", "config.json")}
            prep.write_json(baseline / "mesh_manifest.json", manifest)
            with np.load(baseline / "aster_full_displacement.npz") as fields:
                values = {key: fields[key] for key in fields.files}
            values["displacement_m"] *= 0.
            np.savez(baseline / "aster_full_displacement.npz", **values)
            screening = root / "screening"
            prep.run(config, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
            result = prep.run(config, model, baseline, root / "prepared", job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            self.assertEqual(result["candidate_faces"], 0)
            self.assertFalse(result["accepted"])
            with np.load(root / "prepared" / "cohesive_state.npz") as state:
                self.assertEqual(state["minus"].shape, (0, 3))
                self.assertNotIn("displacement_m", state.files)

    def fixture(self, root):
        model, baseline = root / "model", root / "baseline"
        model.mkdir()
        baseline.mkdir()
        config = {"material": {"concrete": {"youngs_modulus": 30e9, "poisson_ratio": .2,
                    "tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 80.},
                    "bedrock": {"youngs_modulus": 30e9, "poisson_ratio": .25}}}
        configuration = model / "config.json"
        prep.write_json(configuration, config)
        (model / "three_body.msh").write_text("source")
        arrays = column_fixture()
        arrays["source_cell_rows"] = np.arange(len(arrays["tetra4_concrete"]) + len(arrays["tetra4_bedrock"]))
        np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
        displacement = arrays["nodes"] * np.array((0., 0., .001))
        np.savez(baseline / "aster_full_displacement.npz", coordinates_m=arrays["nodes"],
                 source_node_ids=arrays["source_node_ids"], displacement_m=displacement)
        prep.write_json(baseline / "run_manifest.json", {"status": "completed"})
        prep.write_json(baseline / "mesh_manifest.json", {
            "source_mesh": str(model / "three_body.msh"),
            "source_hashes": {name: prep.file_hash(model / name) for name in ("three_body.msh", "config.json")}})
        return model, baseline, configuration

    def test_screen_and_prepare_without_supporting_solver(self):
        with tempfile.TemporaryDirectory() as directory, patch("subprocess.Popen", side_effect=AssertionError("solver")):
            root = Path(directory)
            model, baseline, config = self.fixture(root)
            screening = root / "screening"
            result = prep.run(config, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
            self.assertEqual(result["solver"], "code_aster")
            self.assertGreater(result["facet_check"]["critical_facets"], 0)
            output = root / "prepared"
            result = prep.run(config, model, baseline, output, job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            self.assertEqual(result["status"], prep.PREPARED)
            self.assertFalse(result["accepted"])
            with np.load(output / "cohesive_state.npz") as state:
                self.assertGreater(len(state["minus"]), 0)
                with np.load(baseline / "three_body_mesh_arrays.npz") as arrays:
                    np.testing.assert_array_equal(state["coordinates_m"], arrays["nodes"][state["parent_node_ids"]])

    def test_changed_or_unaccepted_baseline_is_rejected(self):
        for changed in ("run_manifest.json", "source_mesh", "displacement_order"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, baseline, config = self.fixture(root)
                if changed == "run_manifest.json":
                    prep.write_json(baseline / changed, {"status": "failed"})
                elif changed == "source_mesh":
                    (model / "three_body.msh").write_text("changed")
                else:
                    with np.load(baseline / "aster_full_displacement.npz") as stored:
                        fields = {key: stored[key] for key in stored.files}
                    fields["source_node_ids"] = fields["source_node_ids"][::-1]
                    np.savez(baseline / "aster_full_displacement.npz", **fields)
                with self.assertRaises(ValueError):
                    prep.run(config, model, baseline, root / "screening", job_id="fixture", relative_tolerance=1e-4)
                self.assertEqual(json.loads((root / "screening" / "manifest.json").read_text())["status"],
                                 "failed_aster_preparation")

    def test_changed_screening_and_baseline_cannot_prepare_stale_candidates(self):
        for changed in ("screening", "baseline"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, baseline, config = self.fixture(root)
                screening = root / "screening"
                prep.run(config, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
                if changed == "screening":
                    (screening / "screening.json").write_text("{}")
                else:
                    prep.write_json(baseline / "run_manifest.json", {"status": "completed", "new_run": True})
                with self.assertRaises(ValueError):
                    prep.run(config, model, baseline, root / "prepared", job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)


if __name__ == "__main__":
    unittest.main()
