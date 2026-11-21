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
    def test_explicit_water_option_reaches_native_renderer_without_enabling_uplift(self):
        from Aster import automatic_cohesive as native
        from Aster.hydraulic_loads import CRACK_WATER_SETTINGS
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, baseline, config_path = self.fixture(root)
                config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
                config["damage_selection"].pop("crack_water_pressure", None)
                if enabled:
                    config["damage_selection"]["crack_water_pressure"] = True
                prep.write_json(config_path, config)
                arrays = column_fixture()
                arrays["source_cell_rows"] = np.arange(
                    len(arrays["tetra4_concrete"]) + len(arrays["tetra4_bedrock"]))
                np.savez(baseline / native.SPLIT_ARRAYS_NAME, **arrays)
                manifest = fixture_manifest(arrays)
                manifest.update(
                    source_mesh=str(model / "three_body.msh"), source_config=str(config_path),
                    source_hashes={name: prep.file_hash(model / name)
                                   for name in ("three_body.msh", "config.json")},
                    source_pressures={name: {"surface_z_m": level, "gradient_pa_per_m": 9810.}
                                      for name, level in (("UpstreamHydrostaticPressure", 3.),
                                                          ("DownstreamTailwater", 0.))},
                    source_crest_pressure_pa=0.,
                    groups={"CONCRETE": {"min_xyz_m": [0., 0., 0.]}, "ROCK_BOTTOM": {"area_m2": 1.}},
                    wall_foundation_spring_group_all="WALL_FOUNDATION_SPRINGS",
                    wall_foundation_spring_total_area_m2=1., simplifications_vs_elmer=[])
                prep.write_json(baseline / native.MESH_MANIFEST_NAME, manifest)
                screening, prepared = root / "screening", root / "prepared"
                prep.run(config_path, model, baseline, screening, job_id="fixture", relative_tolerance=1e-4)
                summary = prep.run(config_path, model, baseline, prepared, job_id="fixture",
                                   relative_tolerance=1e-4, screening=screening)
                self.assertEqual(summary["crack_water_pressure"], enabled)
                # Only the source-mesh audit is mocked: this fixture has synthetic source bytes.
                with patch("Aster.source_contract.audit_source_materials"):
                    inputs = native.prepare(baseline, prepared, root / "native")
                comm = (root / "native" / native.CASE_NAME).read_text()
                self.assertNotIn("CHARGE=UPLIFT", comm)
                if enabled:
                    self.assertEqual(inputs["joint"]["hydraulic_loads"]["settings"], CRACK_WATER_SETTINGS)
                    self.assertEqual(inputs["joint"]["hydraulic_loads"]["uplift"]["mode"], "none")
                    self.assertIn("hydraulic_open_faces(trial)", comm)
                    # Hydraulic trials keep the DECOUPE sub-stepping of the dry path.
                    self.assertIn("times = DEFI_LIST_INST(", comm)
                    self.assertIn("ACTION='DECOUPE'", comm)
                    self.assertNotIn("    times = DEFI_LIST_REEL(DEBUT=reached", comm)
                else:
                    self.assertNotIn("hydraulic_loads", inputs["joint"])
                    self.assertNotIn("hydraulic_open_faces", comm)
                compile(comm, "prepared.comm", "exec")

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

    def test_all_overstressed_scope_ignores_legacy_damaged_material_override(self):
        from Aster import automatic_cohesive as native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config_path = self.fixture(root)
            config = json.loads(config_path.read_text())
            config["damage_selection"] = {
                "enabled": True,
                "damaged_element_properties": {
                    "density": 2100., "youngs_modulus": 12e9, "poisson_ratio": .23,
                    "tensile_strength_pa": 7e4, "compressive_strength_pa": 15e6,
                    "cohesive_fracture_energy_j_m2": 60.},
            }
            prep.write_json(config_path, config)
            arrays = column_fixture()
            arrays["source_cell_rows"] = np.arange(
                len(arrays["tetra4_concrete"]) + len(arrays["tetra4_bedrock"]))
            np.savez(baseline / "three_body_mesh_arrays.npz", **arrays)
            manifest = fixture_manifest(arrays)
            manifest.update(
                source_mesh=str(model / "three_body.msh"), source_config=str(config_path),
                source_hashes={name: prep.file_hash(model / name)
                               for name in ("three_body.msh", "config.json")})
            prep.write_json(baseline / "mesh_manifest.json", manifest)
            screening, prepared = root / "screening", root / "prepared"
            prep.run(config_path, model, baseline, screening, job_id="fixture",
                     relative_tolerance=1e-4, candidate_scope="all_overstressed")
            result = prep.run(
                config_path, model, baseline, prepared, job_id="fixture",
                relative_tolerance=1e-4, screening=screening,
                candidate_scope="all_overstressed")
            self.assertNotIn("material_assignment", result)
            self.assertEqual(result["tensile_strength_pa"],
                             config["material"]["concrete"]["tensile_strength_pa"])
            self.assertEqual(result["fracture_energy_j_m2"],
                             config["material"]["concrete"]["cohesive_fracture_energy_j_m2"])

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

    def test_lift_joints_weaken_only_the_horizontal_planes_and_keep_elements_intact(self):
        from Aster import automatic_cohesive as native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config_path = self.fixture(root)
            config = json.loads(config_path.read_text())
            config["damage_selection"] = {"enabled": True, "crack_base": True,
                                          "lift_joints": {"tensile_strength_pa": 2e5,
                                                          "cohesive_fracture_energy_j_m2": 5.,
                                                          "max_normal_deviation_deg": 15., "alpha": 1.5,
                                                          "pena_contact": 2.}}
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
            lift = result["selection"]["lift_joints"]
            self.assertGreater(lift["lift_joint_facets"], 0)
            self.assertLess(lift["lift_joint_facets"], lift["candidate_facets"])
            self.assertEqual(lift["law"], "JOINT_MECA_RUPT")
            self.assertEqual(len(lift["source_faces"]), lift["lift_joint_facets"])
            self.assertNotIn("properties", result["material_assignment"])
            self.assertEqual(result["material_assignment"]["joint_properties"]["tensile_strength_pa"], 2e5)
            result = prep.run(config_path, model, baseline, prepared, job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            self.assertEqual(result["lift_joint_faces"], lift["lift_joint_facets"])
            # The CZM strength stays the intact concrete's; the lift values travel separately.
            self.assertEqual(result["tensile_strength_pa"], config["material"]["concrete"]["tensile_strength_pa"])
            with np.load(prepared / "cohesive_state.npz") as state:
                mask = state["lift_joint_faces"]
                self.assertEqual(int(mask.sum()), lift["lift_joint_facets"])
                self.assertEqual(len(mask), len(state["minus"]))

            def render(root=None, mesh_manifest_path=None):
                return fixture_base_comm(json.loads(mesh_manifest_path.read_text()), 4e3), {
                    "material": {"concrete": {"youngs_modulus_pa": 30e9, "poissons_ratio": .2}}}
            with patch("Aster.generate_three_body_case.render_comm", side_effect=render):
                inputs = native.prepare(baseline, prepared, root / "native")
            law = inputs["joint"]["lift_joint"]
            penalty = inputs["joint"]["initial_stiffness_pa_per_m"]
            self.assertEqual(law["law"], "JOINT_MECA_RUPT")
            # The catalogue caps PENA_RUPTURE at 10, so K_N follows from the joint's own energy:
            # below the shared mesh penalty but not below the neighbouring elements' E/h
            # (U2.05.07 4.1.2); the critical opening is still 2 Gf / f_t.
            self.assertEqual(law["pena_rupture"], 10.0)
            self.assertAlmostEqual(law["k_n_pa_per_m"], 2e5 ** 2 * 11 / (2 * 5.))
            self.assertLess(law["k_n_pa_per_m"], penalty)
            self.assertGreaterEqual(law["k_n_pa_per_m"], law["neighbour_stiffness_pa_per_m"])
            self.assertAlmostEqual(law["neighbour_stiffness_pa_per_m"], 30e9 / lift["mean_facet_edge_m"])
            self.assertEqual(law["mesh_penalty_pa_per_m"], penalty)
            self.assertAlmostEqual(law["critical_opening_m"], 5e-5)
            self.assertAlmostEqual(law["tensile_strength_pa"] * (1 + law["pena_rupture"]) / law["k_n_pa_per_m"], 5e-5)
            self.assertAlmostEqual(law["elastic_limit_opening_m"], 5e-5 / 11)
            self.assertEqual(inputs["joint"]["lift_joint_count"], lift["lift_joint_facets"])
            self.assertEqual(inputs["joint"]["damaged_joint_count"], 0)
            with np.load(root / "native" / "three_body_mesh_arrays.npz") as stored:
                self.assertFalse(stored["damaged_cells"].any())
                self.assertFalse(stored["damaged_joint_faces"].any())
                self.assertEqual(int(stored["lift_joint_faces"].sum()), lift["lift_joint_facets"])
            comm = (root / "native" / "case.comm").read_text()
            self.assertIn("LIFT_MAT = DEFI_MATERIAU(", comm)
            self.assertIn("JOINT_MECA_RUPT=_F(K_N=", comm)
            self.assertIn("SIGM_MAX=200000, PENA_RUPTURE=10,", comm)
            self.assertIn("PENA_CONTACT=2, ALPHA=1.5", comm)
            self.assertIn("_F(GROUP_MA='LIFT_JOINTS', MATER=LIFT_MAT),", comm)
            self.assertIn("_F(GROUP_MA='INTACT_JOINTS', MATER=INTACT_JOINT_MAT),", comm)
            self.assertNotIn("GROUP_MA='DAMAGED_JOINTS'", comm)
            self.assertIn("_F(RELATION='JOINT_MECA_RUPT', GROUP_MA='LIFT_JOINTS'),", comm)
            self.assertIn("_F(RELATION='CZM_LIN_REG', GROUP_MA='INTACT_JOINTS'),", comm)
            self.assertIn("LIFT_CELLS = np.asarray(", comm)
            self.assertIn("LIFT_GF = 5.0", comm)
            self.assertIn("RECH_LINEAIRE=_F(METHODE='MIXTE'", comm)
            self.assertNotIn("DAMAGED_MAT", comm)
            self.assertNotIn("DAMAGED_ELEMENTS", comm)
            self.assertNotIn("JOINT_MECA_FROT=_F(", comm)
            self.assertNotIn("RELATION='JOINT_MECA_FROT'", comm)
            compile(comm, "lift.comm", "exec")

    def test_degraded_elements_and_lift_joints_combine_into_three_joint_classes(self):
        """The band's elements take the degraded material, the facets between them its CZM,
        and the near-horizontal planes the weak lift-joint law; the rest stays intact."""
        from Aster import automatic_cohesive as native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline, config_path = self.fixture(root)
            config = json.loads(config_path.read_text())
            properties = {"density": 2100., "youngs_modulus": 12e9, "poisson_ratio": .23,
                          "tensile_strength_pa": 7e5, "compressive_strength_pa": 15e6,
                          "cohesive_fracture_energy_j_m2": 60.}
            config["damage_selection"] = {"enabled": True, "crack_base": True,
                                          "damaged_element_properties": properties,
                                          "lift_joints": {"tensile_strength_pa": 2e5,
                                                          "cohesive_fracture_energy_j_m2": 5.}}
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
            assignment = result["material_assignment"]
            self.assertEqual(assignment["properties"], properties)
            self.assertEqual(assignment["joint_properties"]["tensile_strength_pa"], 2e5)
            lift_count = assignment["joint_properties"]["lift_joint_facets"]
            result = prep.run(config_path, model, baseline, prepared, job_id="fixture",
                              relative_tolerance=1e-4, screening=screening)
            # The CZM of the degraded facets uses the degraded material; the lift law is separate.
            self.assertEqual(result["tensile_strength_pa"], 7e5)
            self.assertEqual(result["fracture_energy_j_m2"], 60.)
            self.assertEqual(result["lift_joint_faces"], lift_count)

            def render(root=None, mesh_manifest_path=None):
                return fixture_base_comm(json.loads(mesh_manifest_path.read_text()), 4e3), {
                    "material": {"concrete": {"youngs_modulus_pa": 30e9, "poissons_ratio": .2}}}
            with patch("Aster.generate_three_body_case.render_comm", side_effect=render):
                inputs = native.prepare(baseline, prepared, root / "native")
            joint = inputs["joint"]
            self.assertEqual(joint["tensile_strength_pa"], 7e5)
            self.assertEqual(joint["intact_joint"]["tensile_strength_pa"], config["material"]["concrete"]["tensile_strength_pa"])
            self.assertEqual(joint["lift_joint"]["tensile_strength_pa"], 2e5)
            self.assertEqual(joint["lift_joint_count"], lift_count)
            self.assertGreater(joint["damaged_joint_count"], 0)
            # Every candidate facet lies inside the band in this fixture, so there is no
            # intact class; the two others must partition the candidates.
            self.assertEqual(joint["candidate_face_count"], joint["lift_joint_count"] + joint["damaged_joint_count"])
            with np.load(root / "native" / "three_body_mesh_arrays.npz") as stored:
                lift, damaged = stored["lift_joint_faces"], stored["damaged_joint_faces"]
                self.assertTrue(stored["damaged_cells"].any())
                self.assertEqual(int(lift.sum()), lift_count)
                self.assertEqual(int(damaged.sum()), joint["damaged_joint_count"])
                self.assertFalse((lift & damaged).any())
            comm = (root / "native" / "case.comm").read_text()
            self.assertIn("GROUP_MA='DAMAGED_ELEMENTS', MATER=DAMAGED_MAT", comm)
            self.assertIn("E=12000000000", comm)
            self.assertIn("_F(GROUP_MA='LIFT_JOINTS', MATER=LIFT_MAT),", comm)
            self.assertIn("_F(GROUP_MA='DAMAGED_JOINTS', MATER=JOINT_MAT),", comm)
            self.assertNotIn("GROUP_MA='INTACT_JOINTS'", comm)
            self.assertIn("_F(RELATION='JOINT_MECA_RUPT', GROUP_MA='LIFT_JOINTS'),", comm)
            self.assertIn("_F(RELATION='CZM_LIN_REG', GROUP_MA='DAMAGED_JOINTS'),", comm)
            self.assertIn("SIGM_C=700000", comm)
            self.assertIn("SIGM_MAX=200000", comm)
            compile(comm, "combined.comm", "exec")

    def test_lift_joint_parameters_follow_r7_01_25(self):
        from Aster.automatic_cohesive import CorroborationError, lift_joint_parameters
        # Rupture law at the catalogue cap: K_N = sigma^2 (1+P) / 2Gf = 1.1e10, below the 1e12
        # mesh penalty and above the neighbouring elements' E/h (1e9 = penalty / 1000).
        law = lift_joint_parameters({"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 5.}, 1e12)
        self.assertEqual(law["pena_rupture"], 10.0)
        self.assertAlmostEqual(law["k_n_pa_per_m"], 1e10 * 11 / 10.)
        self.assertEqual(law["neighbour_stiffness_pa_per_m"], 1e9)
        # ssnp162b: delta_max = (1 + PENA_RUPTURE) sigma / K_N and Gf = sigma delta_max / 2.
        self.assertAlmostEqual(law["tensile_strength_pa"] * (1 + law["pena_rupture"]) / law["k_n_pa_per_m"],
                               law["critical_opening_m"])
        self.assertAlmostEqual(0.5 * law["tensile_strength_pa"] * law["critical_opening_m"], 5.)
        self.assertAlmostEqual(law["elastic_limit_opening_m"], law["critical_opening_m"] / 11)
        self.assertEqual((law["alpha"], law["pena_contact"]), (1., 1.))
        self.assertEqual(law["k_t_pa_per_m"], law["k_n_pa_per_m"])
        self.assertIn("catalogue PENA_RUPTURE cap", law["stiffness_basis"])
        # U2.05.07 4.1.2: the joint must not be softer than its neighbours. Gf = 100 J/m2 would
        # need K_N = 5.5e8 < 1e9, which made the lift plane a soft layer that shed its load;
        # the error names the largest energy the law can carry (1e10 * 11 / 2e9 = 55 J/m2).
        with self.assertRaisesRegex(CorroborationError, r"softer than its neighbouring elements.*55 J/m2"):
            lift_joint_parameters({"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 100.}, 1e12)
        lift_joint_parameters({"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 54.}, 1e12)
        # An explicit neighbour stiffness (E/h over the lift facets) replaces penalty / 1000.
        with self.assertRaisesRegex(CorroborationError, "softer than its neighbouring"):
            lift_joint_parameters({"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 5.}, 1e12, 2e10)
        # A very brittle joint (tiny Gf) would need K_N above the mesh penalty: keep the
        # penalty and solve PENA_RUPTURE, which then lands below the cap.
        brittle = lift_joint_parameters({"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 1e-2}, 1e12)
        self.assertEqual(brittle["k_n_pa_per_m"], 1e12)
        self.assertAlmostEqual(brittle["pena_rupture"], 2 * 1e12 * 1e-2 / 1e10 - 1)
        self.assertLess(brittle["pena_rupture"], 10.0)
        self.assertIn("mesh penalty", brittle["stiffness_basis"])
        # Elastic opening beyond the critical opening leaves no softening branch.
        with self.assertRaisesRegex(CorroborationError, "PENA_RUPTURE > 0"):
            lift_joint_parameters({"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 1e-3}, 1e12)
        for bad in ({"tensile_strength_pa": 0, "cohesive_fracture_energy_j_m2": 1.},
                    {"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 5., "alpha": 3},
                    {"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 5., "pena_contact": 0},
                    {"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 5., "law": "JOINT_MECA_ENDO"},
                    {"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 5., "law": "JOINT_MECA_FROT"},
                    {"tensile_strength_pa": 1e5, "cohesive_fracture_energy_j_m2": 5., "law": "JOINT_MECA_FROT",
                     "friction_coefficient": 0.}):
            with self.subTest(bad=bad), self.assertRaises(CorroborationError):
                lift_joint_parameters(bad, 1e12)

    def test_frictional_lift_joint_is_stiff_and_puts_the_cone_apex_at_the_tensile_strength(self):
        from Aster.automatic_cohesive import FRICTION_OPENING_LIMIT_M, lift_joint_parameters
        # The fracture energy is not used by the perfectly plastic law, so it cannot soften the joint.
        law = lift_joint_parameters({"tensile_strength_pa": 2e4, "cohesive_fracture_energy_j_m2": 1e6,
                                     "law": "JOINT_MECA_FROT", "friction_coefficient": 0.7}, 1e12)
        self.assertEqual(law["law"], "JOINT_MECA_FROT")
        self.assertEqual((law["k_n_pa_per_m"], law["k_t_pa_per_m"]), (1e12, 1e9))
        self.assertAlmostEqual(law["adhesion_pa"], 0.7 * 2e4)
        self.assertAlmostEqual(law["adhesion_pa"] / law["friction_coefficient"], law["tensile_strength_pa"])
        self.assertEqual(law["elastic_limit_opening_m"], FRICTION_OPENING_LIMIT_M)
        self.assertEqual(law["fracture_energy_j_m2"], 0.)
        self.assertIsNone(law["critical_opening_m"])

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
