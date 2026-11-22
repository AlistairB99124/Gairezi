"""Check reduced foundation equilibrium against an explicit foundation."""
import unittest

import numpy as np

from Aster.foundation_condensation import condense_foundation, prepare_condensed_support
from Aster.support_trace import add_consistent_support
from FEniCSx.test_cohesive_contact import tetra_stiffness


class FoundationCondensationTests(unittest.TestCase):
    def setUp(self):
        self.nodes = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., -1.],
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        ])
        self.springs = np.array([[1e9, 2e9, 3e9], [2e9, 3e9, 4e9], [3e9, 4e9, 5e9]])
        self.tensor = np.diag([8e9, 7e9, 9e9])
        self.arrays = {
            "nodes": self.nodes,
            "tetra4_concrete": np.array([[4, 5, 6, 7], [0, 1, 2, 3]]),
            "concrete_body_ids": np.array([1, 3]),
            "seg2_wall_foundation_springs": np.array([[4, 0], [5, 1], [6, 2]]),
            "interface_stiffness_n_per_m": self.springs,
            "interface_names": np.array(["wall_foundation"] * 3),
            "wall_foundation_tributary_area_m2": np.ones(3),
            "support_faces": np.array([[0, 1, 2]]),
            "support_face_tensors_n_per_m3": self.tensor[None],
            "source_node_ids": np.arange(len(self.nodes)),
        }
        add_consistent_support(self.arrays, self.arrays["support_faces"], self.tensor[None])
        self.properties = dict(youngs_modulus=35e9, poisson_ratio=.2,
                               density=2400., gravity_m_s2=-9.81)

    def test_condensed_operator_preserves_equilibrium_for_multiple_wall_loads(self):
        result = condense_foundation(self.arrays, **self.properties)
        np.testing.assert_array_equal(result.wall_nodes, [4, 5, 6])

        matrix = np.zeros((21, 21))
        matrix[:12, :12] = tetra_stiffness(self.nodes[:4])
        surface_mass = (np.ones((3, 3)) + np.eye(3)) / 24
        matrix[:9, :9] += np.kron(surface_mass, self.tensor)
        for node, spring in enumerate(self.springs):
            interior = 3 * node + np.arange(3)
            wall = 12 + 3 * node + np.arange(3)
            matrix[interior, interior] += spring
            matrix[wall, wall] += spring
            matrix[interior, wall] -= spring
            matrix[wall, interior] -= spring
        matrix[12:, 12:] += np.eye(9) * 2e8
        internal_gravity = np.zeros(12)
        internal_gravity[2::3] = 2400. * -9.81 / 24
        for retained in (np.array([], dtype=int), np.array([0, 3]), np.arange(4)):
            reduced_foundation = condense_foundation(
                self.arrays, retained_foundation_nodes=retained, **self.properties)
            np.testing.assert_array_equal(reduced_foundation.retained_foundation_nodes, retained)
            indices = np.r_[np.arange(12, 21), (3 * retained[:, None] + np.arange(3)).ravel()]
            wall_stiffness = np.diag(np.r_[np.full(9, 2e8), np.zeros(3 * len(retained))])
            for wall_load in (
                np.zeros(9),
                np.arange(1., 10.) * 1e5,
                np.array([0., 0., -1e6, 2e6, 0., 0., 0., -3e6, 0.]),
            ):
                with self.subTest(retained=retained, wall_load=wall_load):
                    full = np.linalg.solve(matrix, np.concatenate((internal_gravity, wall_load)))
                    load = np.r_[wall_load, np.zeros(3 * len(retained))]
                    reduced = np.linalg.solve(reduced_foundation.stiffness_n_per_m + wall_stiffness,
                                              reduced_foundation.gravity_load_n + load)
                    np.testing.assert_allclose(reduced, full[indices], rtol=1e-10, atol=1e-14)
        self.assertGreater(np.linalg.norm(result.gravity_load_n), 0.)
        self.assertGreater(np.linalg.norm(result.stiffness_n_per_m[:3, 3:6]), 0.)

    def test_rejects_missing_or_incompatible_foundation_support(self):
        for replacement in (
            {"support_faces": np.empty((0, 3), dtype=int),
             "support_face_tensors_n_per_m3": np.empty((0, 3, 3))},
            {"seg2_wall_foundation_springs": np.array([[0, 4], [1, 5], [2, 6]])},
            {"support_face_tensors_n_per_m3": np.diag([-1., 1., 1.])[None]},
            {"support_matrices_n_per_m": np.zeros((3, 3, 3))},
            {"tetra4_bedrock": np.array([[0, 1, 2, 3]])},
            {"tri3_upstream": np.array([[0, 1, 2]])},
        ):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                condense_foundation({**self.arrays, **replacement}, **self.properties)
        for retained in (np.array([4]), np.array([0, 0]), np.array([0.5])):
            with self.subTest(retained=retained), self.assertRaises(ValueError):
                condense_foundation(self.arrays, retained_foundation_nodes=retained, **self.properties)

    def test_link_budget_retains_an_exact_sparse_support(self):
        result = condense_foundation(self.arrays, max_links=9, **self.properties)
        size = len(result.wall_nodes) + len(result.retained_foundation_nodes)
        blocks = result.stiffness_n_per_m.reshape(size, 3, size, 3).transpose(0, 2, 1, 3)
        first, second = np.triu_indices(size, 1)
        self.assertLessEqual(np.count_nonzero(np.any(blocks[first, second] != 0, axis=(1, 2))), 9)
        full = condense_foundation(self.arrays, **self.properties)
        retained_size = len(result.retained_foundation_nodes) * 3
        wall_size = len(full.wall_nodes) * 3
        wall_stiffness = np.diag(np.r_[np.full(wall_size, 2e8), np.zeros(retained_size)])
        load = np.r_[np.arange(wall_size) * 1e5, np.zeros(retained_size)]
        partial_u = np.linalg.solve(result.stiffness_n_per_m + wall_stiffness,
                                    result.gravity_load_n + load)
        full_u = np.linalg.solve(full.stiffness_n_per_m + np.eye(wall_size) * 2e8,
                                 full.gravity_load_n + load[:wall_size])
        np.testing.assert_allclose(partial_u[:wall_size], full_u, rtol=1e-10, atol=1e-14)
        with self.assertRaises(ValueError):
            condense_foundation(self.arrays, max_links=1, **self.properties)

    def test_support_substitution_preserves_wall_and_rejects_foundation_damage(self):
        joints = np.empty((0, 6), dtype=int)
        converted, report = prepare_condensed_support(
            self.arrays, joint_cells=joints, max_links=9, **self.properties)
        np.testing.assert_array_equal(converted["tetra4_concrete"], self.arrays["tetra4_concrete"][:1])
        self.assertEqual(report["foundation_cells_removed"], 1)
        self.assertEqual(len(converted["support_nodes"]), 0)
        self.assertEqual(len(converted["seg2_wall_foundation_springs"]), 0)
        self.assertEqual(len(converted["point_node_ids"]), len(converted["condensed_gravity_load_n"]))
        with self.assertRaises(ValueError):
            prepare_condensed_support({**self.arrays, "damaged_cells": np.array([False, True])},
                                      joint_cells=joints, max_links=9, **self.properties)
        with self.assertRaises(ValueError):
            prepare_condensed_support(self.arrays, joint_cells=np.array([[0, 1, 2, 4, 5, 6]]),
                                      max_links=9, **self.properties)


if __name__ == "__main__":
    unittest.main()
