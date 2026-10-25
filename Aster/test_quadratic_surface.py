"""Exact integration and discrete assembly checks for quadratic spring surfaces."""
import unittest

import numpy as np

from Aster.quadratic_surface import discrete_entries, group_stiffness_blocks, surface_blocks, triangle_mass


TRIANGLE = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.],
                     [1., 0., 0.], [1., 1.5, 0.], [0., 1.5, 0.]])


def dense(pairs, blocks, count):
    matrix = np.zeros((3 * count, 3 * count))
    for (first, second), block in zip(pairs, blocks):
        matrix[3 * first:3 * first + 3, 3 * second:3 * second + 3] += block
    return matrix


class QuadraticSurfaceTests(unittest.TestCase):
    def test_mixed_order_interface_integrates_affine_jump_exactly(self):
        nodes = np.vstack([TRIANGLE, TRIANGLE[:3]])
        quadratic, linear = np.arange(6).reshape(1, 6), np.arange(6, 9).reshape(1, 3)
        stiffness = np.diag([2., 3., 4.])
        pairs, blocks = surface_blocks(nodes, quadratic, stiffness, linear)
        matrix = dense(pairs, blocks, 9)
        reverse = surface_blocks(nodes, linear, stiffness, quadratic)
        np.testing.assert_allclose(matrix, dense(*reverse, 9), atol=1e-14)
        affine = np.column_stack([nodes[:, 0], nodes[:, 1], np.ones(len(nodes))])
        np.testing.assert_allclose(matrix @ affine.ravel(), 0., atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(matrix).min(), -1e-13)
        jump = np.array([.2, -.3, .4])
        state = np.vstack([np.tile(jump, (6, 1)), np.zeros((3, 3))]).ravel()
        self.assertAlmostEqual(state @ matrix @ state, 3 * jump @ stiffness @ jump)
        np.testing.assert_allclose(triangle_mass(3., 3, 6).T, triangle_mass(3., 6, 3))
        np.testing.assert_allclose(triangle_mass(3., 3, 3), (np.ones((3, 3)) + np.eye(3)) / 4)

    def test_grouping_bounds_every_block_without_erasing_small_stiffness(self):
        rng = np.random.default_rng(6)
        originals = rng.normal(size=(50, 3, 3)) * 10. ** rng.uniform(-12, 15, (50, 1, 1))
        blocks = np.repeat(originals, 2, axis=0)
        grouped, ids, audit = group_stiffness_blocks(blocks)
        self.assertEqual(len(grouped), 50)
        error = np.max(np.abs(grouped[ids] - blocks), axis=(1, 2))
        self.assertTrue(np.all(error <= 1e-10 * np.max(np.abs(blocks), axis=(1, 2))))
        self.assertLessEqual(audit["maximum_block_relative_error"], 1e-10)
        self.assertTrue(np.all(np.max(np.abs(grouped[ids]), axis=(1, 2)) > 0))

    def test_exact_mass_is_positive_with_correct_constant_and_linear_moments(self):
        mass = triangle_mass(3.)
        np.testing.assert_allclose(mass, mass.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(mass).min(), 0)
        self.assertAlmostEqual(mass.sum(), 3.)
        # TRI6 integrated shape weights are zero at vertices, area/3 at midpoints.
        np.testing.assert_allclose(mass.sum(axis=1), [0, 0, 0, 1, 1, 1], atol=1e-14)
        x = TRIANGLE[:, 0]
        self.assertAlmostEqual(x @ mass @ x, 3 * 2 ** 2 / 6)

    def test_pure_midside_mode_is_not_left_unsupported(self):
        pairs, blocks = surface_blocks(TRIANGLE, np.arange(6).reshape(1, 6), np.eye(3))
        matrix = dense(pairs, blocks, 6)
        displacement = np.zeros((6, 3))
        displacement[3, 0] = 1
        self.assertGreater(displacement.ravel() @ matrix @ displacement.ravel(), 0)
        # The previous vertex-only operator would give exactly zero.
        self.assertAlmostEqual(matrix[9, 9], 3 * 8 / 45)

    def test_split_interface_has_zero_common_motion_and_correct_uniform_jump_energy(self):
        nodes = np.vstack([TRIANGLE, TRIANGLE])
        stiffness = np.diag([2., 3., 4.])
        pairs, blocks = surface_blocks(nodes, np.arange(6).reshape(1, 6), stiffness,
                                        np.arange(6, 12).reshape(1, 6))
        matrix = dense(pairs, blocks, 12)
        rng = np.random.default_rng(51)
        displacement = rng.normal(size=(6, 3))
        np.testing.assert_allclose(matrix @ np.tile(displacement.ravel(), 2), 0., atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(matrix).min(), -1e-13)
        jump = np.array([.2, -.3, .4])
        state = np.vstack([np.tile(jump, (6, 1)), np.zeros((6, 3))]).ravel()
        self.assertAlmostEqual(state @ matrix @ state, 3 * jump @ stiffness @ jump)
        # The identical coordinates retain 12 independent node identities.
        self.assertEqual(matrix.shape, (36, 36))

    def test_general_discrete_encoding_reconstructs_the_exact_operator(self):
        nodes = np.vstack([TRIANGLE, TRIANGLE])
        pairs, blocks = surface_blocks(nodes, np.arange(6).reshape(1, 6), np.eye(3),
                                        np.arange(6, 12).reshape(1, 6))
        point_nodes, segment_nodes, points, segments = discrete_entries(pairs, blocks)
        matrix = np.zeros((36, 36))
        for node, entry in zip(point_nodes, points):
            matrix[3 * node:3 * node + 3, 3 * node:3 * node + 3] += entry["matrix"]
        for (first, second), entry in zip(segment_nodes, segments):
            ids = [3 * first + i for i in range(3)] + [3 * second + i for i in range(3)]
            matrix[np.ix_(ids, ids)] += entry["matrix"]
        np.testing.assert_allclose(matrix, dense(pairs, blocks, 12), atol=1e-14)

    def test_directional_support_remains_free_tangentially(self):
        normal = np.array([.6, .8, 0])
        pairs, blocks = surface_blocks(TRIANGLE, np.arange(6).reshape(1, 6),
                                        10 * np.outer(normal, normal))
        matrix = dense(pairs, blocks, 6)
        tangent = np.tile([-.8, .6, 0.], 6)
        np.testing.assert_allclose(matrix @ tangent, 0., atol=1e-14)

    def test_invalid_surface_or_stiffness_fails(self):
        faces = np.arange(6).reshape(1, 6)
        for stiffness in (np.diag([-1., 1, 1]), np.ones((2, 2)),
                          np.array([[1., 1, 0], [0, 1, 0], [0, 0, 1]])):
            with self.subTest(stiffness=stiffness), self.assertRaises(ValueError):
                surface_blocks(TRIANGLE, faces, stiffness)
        curved = TRIANGLE.copy()
        curved[3, 2] = 1
        with self.assertRaisesRegex(ValueError, "straight-sided"):
            surface_blocks(curved, faces, np.eye(3))
        with self.assertRaisesRegex(ValueError, "split"):
            surface_blocks(TRIANGLE, faces, np.eye(3), faces)


if __name__ == "__main__":
    unittest.main()
