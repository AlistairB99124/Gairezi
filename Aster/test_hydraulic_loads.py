import unittest

import numpy as np

from Aster import hydraulic_loads as water
from Aster import automatic_cohesive as cohesive
from Aster.test_automatic_cohesive import column_fixture, crack_faces, fixture_base_comm, fixture_manifest
from pathlib import Path


class HydraulicTests(unittest.TestCase):
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
        self.assertNotIn("times = DEFI_LIST_INST(", comm)

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


if __name__ == "__main__":
    unittest.main()
