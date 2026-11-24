import unittest
from copy import deepcopy
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from Aster import hydraulic_loads as water
from Aster import automatic_cohesive as cohesive
from Aster.test_automatic_cohesive import column_fixture, crack_faces, fixture_base_comm, fixture_manifest
from pathlib import Path


class HydraulicTests(unittest.TestCase):
    def test_reservoir_pressure_moves_waterline_and_integrates_resultants(self):
        from Aster.reservoir_loading import ReservoirPressure
        arrays = column_fixture()
        pressure = ReservoirPressure(arrays, 100.)
        upstream = ("tri3_upstream",)
        dry = pressure.forces(-1., upstream)
        np.testing.assert_allclose(dry, 0.)
        # Source upstream includes foundation z=0..1 and wall z=1..3, width 1 m.
        for surface in (.5, 1.5, 3.):
            forces = pressure.forces(surface, upstream)
            self.assertAlmostEqual(forces[:, 0].sum(), 50. * surface ** 2)
            self.assertAlmostEqual(np.cross(arrays["nodes"], forces)[:, 1].sum(),
                                   100. * surface ** 3 / 6.)
            np.testing.assert_allclose(forces[:, 1:], 0.)
        for surface in (1.5, 3.):
            np.testing.assert_allclose(pressure.forces(surface, ("tri3_crest",)), 0.)
        crest = pressure.forces(3.3, ("tri3_crest",))
        self.assertAlmostEqual(crest[:, 2].sum(), -30.)
        # Opposite face pressure acts into the wall, independent of upstream depth.
        tailwater = pressure.forces(.5, ("tri3_downstream",))
        self.assertAlmostEqual(tailwater[:, 0].sum(), -12.5)

    def test_failed_mechanical_trial_releases_temporary_water_load(self):
        load, destroy, saved = object(), MagicMock(), MagicMock()
        scope = {
            "np": np, "time": time, "RESU": None, "options": {}, "EXCIT": (), "RAMP": object(),
            "MODELE": object(), "reached": 0., "target": .5, "target_level": 0, "mechanical_s": 0.,
            "hydraulic_arrays": {"nodes": np.zeros((1, 3))},
            "hydraulic_settings": {"maximum_iterations": 1}, "hydraulic_subdivision_levels": 4,
            "crack_water": SimpleNamespace(evaluate=lambda *_: (np.ones(1, dtype=bool), np.ones((1, 3)))),
            "hydraulic_open_faces": lambda _: None, "force_entries": lambda _: [{"NOEUD": "N1", "FX": 1.}],
            "_F": lambda **kwargs: kwargs, "AFFE_CHAR_MECA": lambda **_: load,
            "STAT_NON_LINE": MagicMock(side_effect=RuntimeError("mechanical convergence failed")),
            "CA": SimpleNamespace(AsterError=RuntimeError), "stop_for_resource_failure": lambda _: False,
            "DETRUIRE": destroy, "bisect_load_interval": lambda *_: None, "save": saved,
        }
        exec("while True:\n" + water.COUPLED_SOLVE + "    break\n", scope)
        destroy.assert_called_once_with(CONCEPT={"NOM": load})
        self.assertIn("mechanical convergence failed", saved.call_args.args[0]["message"])

    def test_crack_water_can_be_enabled_without_base_uplift(self):
        self.assertEqual(water.validate_settings(water.CRACK_WATER_SETTINGS)["base_uplift"], "none")
        for value in (True, 0, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                water.validate_settings(dict(water.CRACK_WATER_SETTINGS, opening_tolerance_m=value))

    def test_failed_hydraulic_load_target_is_bisected_with_a_bounded_depth(self):
        self.assertEqual(water.bisect_load_interval(.15, .16, 0, 8), (.155, 1))
        self.assertEqual(water.bisect_load_interval(.15, .16, 8, 8), None)
        with self.assertRaises(ValueError):
            water.bisect_load_interval(.16, .15, 0, 8)

    def test_linear_pressure_resultant_and_moment(self):
        vertices = np.array([[0., 0, 0], [2., 0, 0], [0., 1, 0]])
        pressure = np.array([300., 100., 300.])
        forces = water.triangle_pressure_forces(vertices, pressure)
        np.testing.assert_allclose(forces, (np.ones((3, 3)) + np.eye(3)) @ pressure / 12)
        self.assertAlmostEqual(forces.sum(), 700 / 3)
        self.assertAlmostEqual((vertices[:, 0] * forces).sum(), 400 / 3)

    def test_clipped_waterline_integrates_only_wet_area(self):
        vertices = np.array([[0., 0, -1], [1., 0, 1], [0., 1, 1]])
        forces = water.triangle_pressure_forces(vertices, np.array([1., -1., -1.]))
        area = np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) / 2
        self.assertAlmostEqual(forces.sum(), area / 12)
        np.testing.assert_allclose(water.triangle_pressure_forces(vertices, np.full(3, -1.)), 0)

    @staticmethod
    def network():
        # Two connected horizontal faces and one isolated face; plus-side copies.
        points = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [1., 1, 0],
                           [3., 0, 0], [4., 0, 0], [3., 1, 0]])
        faces = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
        return {"nodes": np.vstack([points, points]), "source_node_ids": np.tile(np.arange(7), 2),
                "penta6_joints": np.hstack([faces, faces + 7]),
                "tri3_upstream": np.array([[0, 2, 6]])}

    def test_only_open_reservoir_connected_cracks_get_paired_pressure(self):
        arrays = self.network()
        system = water.CrackWater(arrays, 2., 100., 1e-9)
        displacement = np.zeros_like(arrays["nodes"])
        wet, force = system.evaluate(displacement)
        self.assertFalse(wet.any())
        np.testing.assert_allclose(force, 0)
        displacement[:7, 2] = .001
        wet, force = system.evaluate(displacement)
        np.testing.assert_array_equal(wet, [True, True, False])
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.cross(arrays["nodes"], force).sum(axis=0), 0, atol=1e-12)
        self.assertAlmostEqual(force[:7, 2].sum(), 200.)
        displacement[:7, 2] = -.001
        self.assertFalse(system.evaluate(displacement)[0].any())
        displacement[:7, 2] = .001
        dry = water.CrackWater(arrays, -1., 100., 1e-9)
        self.assertFalse(dry.evaluate(displacement)[0].any())

    def test_closed_connection_does_not_feed_disconnected_open_face(self):
        arrays = self.network()
        system = water.CrackWater(arrays, 2., 100., 1e-9)
        displacement = np.zeros_like(arrays["nodes"])
        displacement[3, 2] = .001
        self.assertFalse(system.evaluate(displacement)[0].any())

    def test_rising_crack_water_surface_recomputes_submergence_and_force_weights(self):
        arrays = self.network()
        system = water.CrackWater(arrays, -1., 100., 1e-9)
        displacement = np.zeros_like(arrays["nodes"])
        displacement[:7, 2] = .001
        self.assertFalse(system.evaluate(displacement)[0].any())
        system.set_surface(1.)
        wet, force = system.evaluate(displacement)
        np.testing.assert_array_equal(wet, [True, True, False])
        self.assertAlmostEqual(force[:7, 2].sum(), 100.)
        system.set_surface(2.)
        self.assertAlmostEqual(system.evaluate(displacement)[1][:7, 2].sum(), 200.)
        system.set_surface(-1.)
        self.assertFalse(system.evaluate(displacement)[0].any())
        with self.assertRaises(ValueError):
            system.set_surface(float("nan"))

    def test_elastic_joint_compliance_cannot_admit_water_before_damage(self):
        arrays = self.network()
        system = water.CrackWater(arrays, 2., 100., 1e-9)
        displacement = np.zeros_like(arrays["nodes"])
        displacement[:7, 2] = .001
        wet, forces = system.evaluate(displacement, np.zeros(3, dtype=bool))
        self.assertFalse(wet.any())
        np.testing.assert_allclose(forces, 0)
        wet, _ = system.evaluate(displacement, np.array([True, True, True]))
        np.testing.assert_array_equal(wet, [True, True, False])
        wet, _ = system.evaluate(displacement, np.array([False, True, True]))
        self.assertFalse(wet.any())
        with self.assertRaises(ValueError):
            system.evaluate(displacement, np.ones(3))

    @staticmethod
    def lift_plane():
        """Horizontal sector of a wall at z = 0: radii 10 (downstream) to 12 (upstream), 8 facets."""
        radii, angles = (10., 11., 12.), (0., .1, .2)
        index = lambda r, a: 3 * r + a
        points = np.array([[radius * np.cos(angle), radius * np.sin(angle), 0.]
                           for radius in radii for angle in angles])
        faces = []
        for r in range(2):
            for a in range(2):
                faces += [[index(r, a), index(r + 1, a), index(r + 1, a + 1)],
                          [index(r, a), index(r + 1, a + 1), index(r, a + 1)]]
        faces = np.array(faces)
        return {"nodes": np.vstack([points, points]), "source_node_ids": np.tile(np.arange(9), 2),
                "penta6_joints": np.hstack([faces, faces + 9]),
                "tri3_upstream": np.array([[index(2, 0), index(2, 1), index(2, 2)]]),
                "tri3_downstream": np.array([[index(0, 0), index(0, 1), index(0, 2)]])}, faces

    def test_lift_plane_uplift_is_linear_across_the_wall_and_full_head_where_wet(self):
        arrays, faces = self.lift_plane()
        points = arrays["nodes"][:9]
        areas = np.linalg.norm(np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                                        points[faces[:, 2]] - points[faces[:, 0]]), axis=1) / 2
        pore = np.ones(8, dtype=bool)
        pore[-1] = False
        # Reservoir 5 m and tailwater 1 m above the plane, unit weight 100 N/m3.
        system = water.CrackWater(arrays, 5., 100., 1e-9, pore_faces=pore, downstream_z=1.)
        radius = np.linalg.norm(points[:, :2], axis=1)
        pressure = 100. * (5. + (1. - 5.) * (12. - radius) / 2.)    # 500 Pa upstream, 100 Pa downstream
        self.assertAlmostEqual(system.pore_report["maximum_pressure_pa"], 500.)
        self.assertAlmostEqual(system.pore_report["minimum_pressure_pa"], 100.)
        self.assertEqual(system.pore_report["faces"], 7)
        closed = np.zeros_like(arrays["nodes"])
        wet, force = system.evaluate(closed, np.zeros(8, dtype=bool))
        self.assertFalse(wet.any())
        expected = (areas[pore] * pressure[faces[pore]].mean(axis=1)).sum()
        # Equal and opposite on the two lips: the joint carries the effective stress.
        self.assertAlmostEqual(abs(force[:9, 2].sum()), expected)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(force[:9] + force[9:], 0, atol=1e-9)
        # Open, reservoir-connected faces take the full head instead (never both).
        opened = closed.copy()
        opened[:9, 2] = 1e-3 * np.sign(system.normals[0, 2])
        wet, force = system.evaluate(opened, np.ones(8, dtype=bool))
        self.assertTrue(wet.all())
        self.assertAlmostEqual(abs(force[:9, 2].sum()), areas.sum() * 500.)
        # Without the plane's faces on both wall surfaces the profile cannot be built.
        with self.assertRaisesRegex(ValueError, "upstream and downstream faces"):
            water.CrackWater({**arrays, "tri3_downstream": np.array([[0, 0, 0]])}, 5., 100., 1e-9,
                             pore_faces=pore, downstream_z=1.)
        for bad in (np.ones(8), np.ones(7, dtype=bool), np.zeros(8, dtype=bool)):
            with self.assertRaises(ValueError):
                water.CrackWater(arrays, 5., 100., 1e-9, pore_faces=bad, downstream_z=1.)
        with self.assertRaisesRegex(ValueError, "tailwater"):
            water.CrackWater(arrays, 5., 100., 1e-9, pore_faces=pore)
        self.assertEqual(water.CrackWater(arrays, 5., 100., 1e-9).pore_report, {"mode": "none"})

    def test_missing_or_unstable_wet_history_cannot_be_accepted(self):
        report = {"hydraulic_loads": {"settings": water.CRACK_WATER_SETTINGS},
                  "steps": [{"load_factor": 1.}],
                  "hydraulic_steps": [{"load_factor": 1., "active_set_stable": True, "iterations": 2,
                      "wet_faces": 1, "paired_resultant_n": [0., 0., 0.],
                      "paired_moment_n_m": [0., 0., 0.], "uplift_resultant_n": [0., 0., 0.],
                      "sum_nodal_force_magnitudes_n": 100.}]}
        water.validate_crack_water_evidence(report)
        for key, value in (("active_set_stable", False), ("load_factor", .9), ("wet_faces", -1),
                           ("iterations", 21), ("sum_nodal_force_magnitudes_n", float("nan")),
                           ("uplift_resultant_n", [0., 0., 1.])):
            broken = deepcopy(report)
            broken["hydraulic_steps"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                water.validate_crack_water_evidence(broken)
        with self.assertRaises(ValueError):
            water.validate_crack_water_evidence({})
        broken = deepcopy(report)
        broken["hydraulic_steps"] = []
        with self.assertRaises(ValueError):
            water.validate_crack_water_evidence(broken)

    def test_wet_load_factors_are_explicit_and_dry_defaults_are_unchanged(self):
        self.assertEqual(cohesive.load_factors({}), [.5, .6, .7, .8, .9, 1.])
        factors = cohesive.load_factors({"crack_water_pressure": True})
        self.assertEqual(factors[0], .01)
        self.assertEqual(factors[-1], 1.)
        self.assertTrue(np.all(np.diff(factors) > 0))
        self.assertLessEqual(max(np.diff(factors)), .025 + 1e-9)
        finer = cohesive.load_factors({
            "crack_water_pressure": True, "solver_controls": {"load_increment": .005}})
        self.assertEqual(finer[:3], [.01, .015, .02])
        self.assertEqual(finer[-1], 1.)
        self.assertLessEqual(max(np.diff(finer)), .005 + 1e-9)

    def test_uplift_on_wall_only_using_measured_heel_to_toe(self):
        arrays = column_fixture()
        radius = arrays["nodes"][:, 0] + 80.
        angle = arrays["nodes"][:, 1] / 80.
        arrays["nodes"][:, 0] = radius * np.cos(angle)
        arrays["nodes"][:, 1] = radius * np.sin(angle)
        arrays["tri3_upstream"], arrays["tri3_downstream"] = arrays["tri3_downstream"], arrays["tri3_upstream"]
        forces, report = water.uplift_forces(arrays, 3., 1., 100.)
        self.assertGreater(report["faces"], 0)
        self.assertGreater(report["area_m2"], 1.)
        self.assertLess(report["area_m2"], 1.02)
        self.assertGreater(forces[:, 2].sum(), 99.)
        self.assertLess(forces[:, 2].sum(), 101.)
        foundation = np.unique(arrays["tetra4_concrete"][arrays["concrete_body_ids"] != 1])
        np.testing.assert_allclose(forces[foundation], 0)
        np.testing.assert_allclose(forces[:, :2], 0, atol=1e-12)

    def test_hydraulic_renderer_keeps_trial_damage_out_of_accepted_state(self):
        arrays = column_fixture()
        split = cohesive.split_for_joints(arrays, crack_faces(arrays))
        cohesive.integrate_joint_faces(split)
        joint = cohesive.joint_parameters(arrays, crack_faces(arrays),
                                         {"tensile_strength_pa": 8e5, "fracture_energy_j_m2": 100}, 30e9, .2)
        joint["hydraulic_loads"] = {"settings": water.SETTINGS, "upstream_z_m": 3.,
                                    "gradient_pa_per_m": 9810., "uplift": {"resultant_n": [0., 0., 100.]}}
        comm = cohesive.render_cohesive_comm(fixture_base_comm(fixture_manifest(arrays), 1e4),
                    joint, [.01, .1, 1.], Path("run") / "steps.npz", Path("run") / "status.json",
                    interpolation=split["arrays"])
        compile(comm, "case.comm", "exec")
        self.assertIn("trial_options['ETAT_INIT'] = _F(EVOL_NOLI=previous, INST=reached)", comm)
        self.assertNotIn("reuse=RESU, ETAT_INIT", comm)
        self.assertIn("Crack-water active set did not converge", comm)
        self.assertIn('status["hydraulic_steps"] = hydraulic_history', comm)
        # Trials keep the dry path's DECOUPE sub-stepping: the trial's last archived instant is
        # still the target and ETAT_INIT(INST=reached) is unaffected by archived sub-instants.
        self.assertIn("times = DEFI_LIST_INST(", comm)
        self.assertIn("ACTION='DECOUPE'", comm)
        self.assertIn("hydraulic_open_faces(trial)", comm)
        self.assertIn("(damage >= 1. - 1e-9) & (jump > HYDRAULIC_OPENING_LIMIT)", comm)
        self.assertIn("TARGET_LEVELS[target_index:target_index + 1]", comm)
        self.assertIn("hydraulic_subdivision_levels = 4", comm)
        self.assertIn("ASTER_HYDRAULIC_SUBDIVIDE", comm)

    def test_uplift_is_invariant_under_cohesive_and_relief_node_splitting(self):
        arrays = column_fixture()
        radius = arrays["nodes"][:, 0] + 80.
        angle = arrays["nodes"][:, 1] / 80.
        arrays["nodes"][:, 0] = radius * np.cos(angle)
        arrays["nodes"][:, 1] = radius * np.sin(angle)
        arrays["tri3_upstream"], arrays["tri3_downstream"] = arrays["tri3_downstream"], arrays["tri3_upstream"]
        # Identify candidate faces on the unchanged height plane using source rows.
        base_fixture = column_fixture()
        split = cohesive.split_for_joints(arrays, crack_faces(base_fixture))
        cohesive.relieve_neighbourhoods(split)
        cohesive.integrate_joint_faces(split)
        _, intact = water.uplift_forces(arrays, 3., 1., 9810.)
        _, cracked = water.uplift_forces(split["arrays"], 3., 1., 9810.)
        self.assertEqual(intact["faces"], cracked["faces"])
        np.testing.assert_allclose(intact["resultant_n"], cracked["resultant_n"], rtol=1e-12)
        np.testing.assert_allclose(intact["moment_n_m"], cracked["moment_n_m"], rtol=1e-12)


class NativeCrackWaterTests(unittest.TestCase):
    def test_native_opened_section_carries_reservoir_pressure_without_uplift(self):
        self.solve_section(with_joints=True)

    def test_native_jointless_control_remains_dry_with_stable_hydraulic_history(self):
        self.solve_section(with_joints=False)

    def test_wet_fine_and_coarse_increments_agree_for_column_fixture(self):
        coarse = self.solve_section(with_joints=True)
        fine = self.solve_section(with_joints=True, factors=[i / 20 for i in range(1, 21)])
        np.testing.assert_array_equal(coarse["damage"], fine["damage"])
        np.testing.assert_allclose(coarse["jump"], fine["jump"], rtol=0.02, atol=1e-8)
        np.testing.assert_allclose(coarse["traction"], fine["traction"], rtol=0.02, atol=100.)
        np.testing.assert_allclose(coarse["displacement"], fine["displacement"], rtol=0.02, atol=1e-8)
        np.testing.assert_array_equal(coarse["wet_faces"], fine["wet_faces"])

    def solve_section(self, *, with_joints, factors=None):
        from Aster.test_automatic_cohesive import _aster_launcher
        if not _aster_launcher():
            self.skipTest("Code_Aster is not installed")
        arrays = column_fixture()
        faces = crack_faces(arrays) if with_joints else np.empty((0, 3), dtype=np.int64)
        split = cohesive.split_for_joints(arrays, faces)
        cohesive.integrate_joint_faces(split)
        joint = cohesive.joint_parameters(arrays, faces,
            {"tensile_strength_pa": 3e5, "fracture_energy_j_m2": 100.}, 30e9, .2)
        joint["has_joints"] = with_joints
        joint["hydraulic_loads"] = {
            "settings": water.CRACK_WATER_SETTINGS, "upstream_z_m": 3.,
            "gradient_pa_per_m": 4e5, "uplift": {"mode": "none", "resultant_n": [0., 0., 0.]}}
        if factors is None:
            factors = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.] if with_joints else [.5, 1.]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "run").mkdir()
            np.savez(root / cohesive.SPLIT_ARRAYS_NAME, penta6_joints=split["joints"], **split["arrays"])
            comm = cohesive.render_cohesive_comm(fixture_base_comm(
                cohesive.split_manifest(fixture_manifest(arrays), split), 4e5),
                joint, factors, root / "run" / cohesive.STEPS_NAME, root / "run" / cohesive.STATUS_NAME,
                tie_pairs=len(split["arrays"]["tie_node_pairs"]), interpolation=split["arrays"])
            (root / cohesive.CASE_NAME).write_text(comm, encoding="utf-8")
            status = cohesive.solve(root, time_limit_s=240, memory_limit_mb=2048)
            log = (root / "run" / "solver.live.log").read_text(encoding="utf-8", errors="replace")
            self.assertEqual(status["status"], "completed", f"{status}\n{log[-6000:]}")
            self.assertEqual(status["returncode"], 0)
            history = status["hydraulic_steps"]
            self.assertEqual(len(history), len(factors))
            self.assertEqual(len(status["step_timings"]), len(factors))
            for timing, factor in zip(status["step_timings"], factors):
                self.assertEqual(timing["load_factor"], factor)
                self.assertGreater(timing["mechanical_s"], 0.)
                self.assertGreaterEqual(timing["step_s"], timing["mechanical_s"])
                self.assertGreater(timing["archive_s"], 0.)
            self.assertEqual(history[0]["wet_faces"], 0)
            if with_joints:
                self.assertGreater(history[-1]["wet_faces"], 0, status)
                self.assertGreater(history[-1]["sum_nodal_force_magnitudes_n"], 0)
            else:
                self.assertTrue(all(step["wet_faces"] == 0 for step in history))
                self.assertTrue(all(step["sum_nodal_force_magnitudes_n"] == 0 for step in history))
            water.validate_crack_water_evidence({
                "hydraulic_loads": joint["hydraulic_loads"], "hydraulic_steps": history,
                "steps": [{"load_factor": value} for value in factors]})
            for step in history:
                np.testing.assert_allclose(step["paired_resultant_n"], 0., atol=1e-8)
                np.testing.assert_allclose(step["paired_moment_n_m"], 0., atol=1e-8)
                np.testing.assert_allclose(step["uplift_resultant_n"], 0.)
            with np.load(root / "run" / cohesive.STEPS_NAME.replace(".npz", "_hydraulic.npz")) as stored:
                self.assertEqual(stored["wet_faces"].shape, (len(factors), len(faces)))
                wet_faces = stored["wet_faces"][-1].copy()
            with np.load(root / "run" / cohesive.STEPS_NAME.replace(".npz", "_displacement.npz")) as fields:
                displacement = fields["final"].copy()
            with np.load(root / "run" / cohesive.STEPS_NAME) as fields:
                return {"damage": fields["damage_state"][-1].copy(),
                        "jump": fields["normal_jump_m"][-1].copy(),
                        "traction": fields["normal_traction_pa"][-1].copy(),
                        "displacement": displacement, "wet_faces": wet_faces}


if __name__ == "__main__":
    unittest.main()
