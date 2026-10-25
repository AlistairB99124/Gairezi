import unittest

import numpy as np

from Aster.compare_quadratic_baseline import (
    centroid_kinematics, compare_region, recovered_peak_audit, tetrahedron_kinematics,
)
from Aster.quadratic_mesh import elevate


class QuadraticComparisonTests(unittest.TestCase):
    def test_direct_peak_audit_reproduces_uniform_affine_stress(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        mesh = elevate(vertices, np.array([[0, 1, 2, 3]]), np.array([[0, 1, 2]]))
        xyz = mesh["nodes"]
        native = {"candidate_node_ids": np.arange(10), "displacement_m": xyz * .001,
                  "stress_pa": np.tile([1., 1., 1., 0, 0, 0], (10, 1))}
        report = recovered_peak_audit(
            xyz, mesh["tetra10"], native, 1000., 0.,
            mesh["tri6"], np.array([7]), {2: 'Name = "joint"\nTarget Boundaries(1) = 7'},
        )
        self.assertEqual(report["incident_cell_count"], 1)
        self.assertLess(report["relative_recovered_vs_arithmetic_tensor_error"], 1e-14)
        self.assertEqual(report["incident_source_boundaries"][0]["name"], "joint")
        for barycentric in ([0, 0, 0, 0], [2, -1, 0, 0], [float("nan"), 0, 0, 1]):
            with self.subTest(barycentric=barycentric), self.assertRaisesRegex(ValueError, "barycentric"):
                tetrahedron_kinematics(xyz, mesh["tetra10"], native["displacement_m"], barycentric)

    def test_quadratic_centroid_gradient_is_exact_for_quadratic_polynomial(self):
        vertices = np.array([[2., 3., 4.], [4., 3., 4.], [3., 6., 4.], [3., 4., 8.]])
        mesh = elevate(vertices, np.array([[0, 1, 2, 3]]), np.empty((0, 3), dtype=int))
        xyz = mesh["nodes"]
        displacement = np.column_stack([xyz[:, 0] ** 2, xyz[:, 1] ** 2, xyz[:, 2] ** 2])
        strain, volume = centroid_kinematics(xyz, mesh["tetra10"], displacement)
        np.testing.assert_allclose(strain[0], np.diag(2 * vertices.mean(axis=0)), atol=1e-12)
        self.assertAlmostEqual(volume[0], 4.)
        for barycentric in (np.array([1., 0, 0, 0]), np.array([.1, .2, .3, .4]),
                            np.array([.5, .5, 0, 0])):
            strain, _ = tetrahedron_kinematics(xyz, mesh["tetra10"], displacement, barycentric)
            np.testing.assert_allclose(strain[0], np.diag(2 * (barycentric @ vertices)), atol=1e-12)

    def setUp(self):
        self.reference = {
            "source_node_ids": np.array([7, 2]),
            "points_m": np.array([[1., 0, 0], [0., 0, 0]]),
            "displacement_m": np.ones((2, 3)),
            "stress_pa": np.array([[2e6, 0, 0, 0, 0, 0], [1e6, 0, 0, 0, 0, 0]]),
        }
        self.native = {
            "source_node_ids": np.array([2, 7, -1]),
            "points_m": np.array([[0., 0, 0], [1., 0, 0], [.5, 0, 0]]),
            "displacement_m": np.ones((3, 3)) * 1.01,
            "stress_pa": np.array([[1e6, 0, 0, 0, 0, 0], [2.2e6, 0, 0, 0, 0, 0],
                                    [3e6, 0, 0, 0, 0, 0]]),
        }

    def test_comparison_separates_common_nodes_from_midside_peak(self):
        report = compare_region(self.reference, self.native)
        self.assertEqual(report["matched_original_nodes"], 2)
        self.assertAlmostEqual(report["relative_l2_displacement_difference"], .01)
        self.assertAlmostEqual(report["recovered_tension_at_common_original_nodes"]["difference_percent"], 10)
        self.assertEqual(report["aster_all_nodes_recovered_tension"]["mpa"], 3)

    def test_missing_duplicate_or_mislocated_original_identity_fails(self):
        for ids in ([2, 99, -1], [2, 2, -1], [7, 2, -1]):
            self.native["source_node_ids"] = np.array(ids)
            with self.subTest(ids=ids), self.assertRaises((ValueError, AssertionError)):
                compare_region(self.reference, self.native)

    def test_identical_partition_duplicates_are_verified_not_averaged(self):
        for key, values in self.reference.items():
            self.reference[key] = np.concatenate([values, values[:1]])
        report = compare_region(self.reference, self.native)
        self.assertEqual(report["matched_original_nodes"], 2)
        self.reference["stress_pa"][-1, 0] += 1
        with self.assertRaisesRegex(ValueError, "Conflicting repeated"):
            compare_region(self.reference, self.native)
