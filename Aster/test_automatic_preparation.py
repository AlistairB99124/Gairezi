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
            source_wall = set(arrays["source_cell_rows"][:len(arrays["concrete_body_ids"])][arrays["concrete_body_ids"] == 1])
            self.assertTrue(set(result["selection"]["selected_source_rows"]).issubset(source_wall))
            result = prep.run(config, model, baseline, root / "prepared", job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            self.assertGreater(result["candidate_faces"], 0)

    def test_no_eligible_facets_prepares_native_elastic_steps_not_fake_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config = self.fixture(root)
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
