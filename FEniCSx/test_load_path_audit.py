"""Check wall-bedrock node identity and checkpoint topology mapping."""
import unittest

import numpy as np

from load_path_audit import match_checkpoint_geometry, region_connection


class LoadPathAuditTests(unittest.TestCase):
    def test_coincident_interface_without_shared_ids_is_not_bonded(self):
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1],
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=float)
        report = region_connection(points, [[0, 1, 2, 3], [4, 5, 6, 7]], [1, 2])
        self.assertEqual(report["shared_vertex_ids"], 0)
        self.assertEqual(report["coincident_coordinate_count_rounded_1e_9_m"], 3)
        self.assertFalse(report["direct_continuous_p1_displacement_transfer"])
        bonded = region_connection(points, [[0, 1, 2, 3], [0, 1, 2, 7]], [1, 2])
        self.assertEqual(bonded["shared_vertex_ids"], 3)
        self.assertTrue(bonded["direct_continuous_p1_displacement_transfer"])

    def test_missing_body_region_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly wall"):
            region_connection(np.zeros((4, 3)), [[0, 1, 2, 3]], [2])

    def test_checkpoint_mapping_preserves_duplicate_interface_nodes_after_reordering(self):
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1],
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=float)
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        permutation = np.array([7, 0, 4, 1, 5, 2, 6, 3])
        inverse = np.argsort(permutation)
        actual = match_checkpoint_geometry(
            points, cells, points[permutation], inverse[cells[::-1]], np.array([2, 1]),
        )
        np.testing.assert_array_equal(actual, permutation)
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            match_checkpoint_geometry(points, cells, points + 1, cells, np.array([1, 2]))
