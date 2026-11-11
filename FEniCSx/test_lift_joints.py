import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift_joints import (lift_joint_facet_mask, lift_joint_report, lift_joint_settings,  # noqa: E402
                         facet_unit_normals)


def _cube_tets(origin, size, offset):
    """Six tetrahedra filling one cube; node ids offset into a shared point list."""
    x, y, z = origin
    corners = np.array([[x, y, z], [x + size, y, z], [x + size, y + size, z], [x, y + size, z],
                        [x, y, z + size], [x + size, y, z + size], [x + size, y + size, z + size],
                        [x, y + size, z + size]], dtype=float)
    tets = np.array([[0, 1, 3, 7], [0, 1, 7, 4], [1, 2, 3, 7], [1, 2, 7, 6], [1, 5, 4, 7], [1, 5, 7, 6]]) + offset
    return corners, tets


def _interior_faces(tets):
    local = np.array(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    keys = np.sort(tets[:, local].reshape(-1, 3), axis=1)
    owners = np.repeat(np.arange(len(tets)), 4)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse.ravel(), kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    interior = counts == 2
    pairs = np.stack((owners[order][starts[interior]], owners[order][starts[interior] + 1]), axis=1)
    return unique[interior], pairs


class StackedCubes(unittest.TestCase):
    """Two cubes stacked in z sharing their 4 top/bottom nodes: the shared plane is horizontal."""

    def setUp(self):
        lower, lower_tets = _cube_tets((0, 0, 0), 1.0, 0)
        upper, upper_tets = _cube_tets((0, 0, 1.0), 1.0, 4)
        # Upper cube reuses the lower cube's top nodes 4..7 as its bottom nodes 0..3.
        self.points = np.vstack([lower, upper[4:]])
        self.tets = np.vstack([lower_tets, upper_tets])
        self.faces, self.owners = _interior_faces(self.tets)
        self.lower_cells = np.arange(6)
        self.upper_cells = np.arange(6, 12)

    def test_facet_normals_are_unit_vectors(self):
        normals = facet_unit_normals(self.points, self.faces)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)

    def test_only_horizontal_facets_between_selected_cells_are_lift_joints(self):
        everything = np.ones(len(self.tets), dtype=bool)
        mask = lift_joint_facet_mask(self.points, self.faces, self.owners, everything, 15.0)
        normals = facet_unit_normals(self.points, self.faces)
        z = self.points[self.faces][:, :, 2]
        # Exactly the two triangles of the shared z=1 plane.
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(np.all(np.isclose(z[mask], 1.0)))
        self.assertTrue(np.all(np.isclose(np.abs(normals[mask][:, 2]), 1.0)))
        # Every unselected facet is non-horizontal (within the cubes) or rejected.
        self.assertTrue(np.all(np.abs(normals[~mask][:, 2]) < math.cos(math.radians(15.0))))

    def test_both_owner_cells_must_be_selected(self):
        mask = lift_joint_facet_mask(self.points, self.faces, self.owners, self.lower_cells, 15.0)
        self.assertEqual(int(mask.sum()), 0)
        mask = lift_joint_facet_mask(self.points, self.faces, self.owners, self.upper_cells, 15.0)
        self.assertEqual(int(mask.sum()), 0)
        both = np.concatenate([self.lower_cells, self.upper_cells])
        self.assertEqual(int(lift_joint_facet_mask(self.points, self.faces, self.owners, both, 15.0).sum()), 2)

    def test_deviation_tolerance_admits_tilted_planes(self):
        tilted = self.points.copy()
        tilted[4:8, 2] += np.array([0.0, 0.0, 0.2, 0.2])  # tilt the shared plane about x by ~11.3 deg
        everything = np.ones(len(self.tets), dtype=bool)
        self.assertEqual(int(lift_joint_facet_mask(tilted, self.faces, self.owners, everything, 15.0).sum()), 2)
        self.assertEqual(int(lift_joint_facet_mask(tilted, self.faces, self.owners, everything, 5.0).sum()), 0)

    def test_inputs_are_validated(self):
        everything = np.ones(len(self.tets), dtype=bool)
        with self.assertRaises(ValueError):
            lift_joint_facet_mask(self.points, self.faces, self.owners[:-1], everything, 15.0)
        for bad in (0, 90, -1):
            with self.assertRaises(ValueError):
                lift_joint_facet_mask(self.points, self.faces, self.owners, everything, bad)
        self.assertEqual(len(lift_joint_facet_mask(self.points, np.zeros((0, 3)), np.zeros((0, 2)), everything, 15.0)), 0)

    def test_report_counts_area_and_elevation(self):
        settings = lift_joint_settings({"tensile_strength_pa": 2e5, "cohesive_fracture_energy_j_m2": 10.0})
        everything = np.ones(len(self.tets), dtype=bool)
        mask = lift_joint_facet_mask(self.points, self.faces, self.owners, everything, settings["max_normal_deviation_deg"])
        report = lift_joint_report(self.points, self.faces, mask, settings)
        self.assertEqual(report["lift_joint_facets"], 2)
        self.assertEqual(report["candidate_facets"], len(self.faces))
        self.assertAlmostEqual(report["lift_joint_area_m2"], 1.0)
        self.assertEqual(report["lift_plane_z_m"], [1.0, 1.0])
        self.assertAlmostEqual(report["critical_opening_m"], 1e-4)
        self.assertEqual(report["law"], "JOINT_MECA_RUPT")
        self.assertIn("15 degrees", report["rule"])
        empty = lift_joint_report(self.points, self.faces, np.zeros(len(self.faces), dtype=bool), settings)
        self.assertEqual(empty["lift_joint_facets"], 0)
        self.assertIsNone(empty["lift_plane_z_m"])
        with self.assertRaises(ValueError):
            lift_joint_report(self.points, self.faces, mask[:-1], settings)


class SettingsTests(unittest.TestCase):
    def test_defaults_fill_in(self):
        settings = lift_joint_settings({"tensile_strength_pa": 2e5, "cohesive_fracture_energy_j_m2": 10.0})
        self.assertEqual(settings["max_normal_deviation_deg"], 15.0)
        self.assertIs(settings["include_base_contact"], False)
        self.assertEqual(settings["law"], "JOINT_MECA_RUPT")
        self.assertEqual(settings["alpha"], 1.0)
        self.assertEqual(settings["pena_contact"], 1.0)

    def test_invalid_settings_are_rejected(self):
        base = {"tensile_strength_pa": 2e5, "cohesive_fracture_energy_j_m2": 10.0}
        for changes in ({"tensile_strength_pa": 0}, {"cohesive_fracture_energy_j_m2": -1},
                        {"tensile_strength_pa": True}, {"max_normal_deviation_deg": 90},
                        {"max_normal_deviation_deg": 0}, {"law": "JOINT_MECA_FROT"}, {"alpha": 2.5},
                        {"alpha": -0.1}, {"pena_contact": 0}, {"include_base_contact": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                lift_joint_settings({**base, **changes})
        with self.assertRaises(ValueError):
            lift_joint_settings(None)


if __name__ == "__main__":
    unittest.main()
