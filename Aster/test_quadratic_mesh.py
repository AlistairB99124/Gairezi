"""Topology-only degree elevation must retain coincident split interfaces."""
import unittest

import numpy as np

from Aster.quadratic_mesh import elevate, operators


class QuadraticMeshTests(unittest.TestCase):
    def test_source_operator_pairs_mixed_interface_in_both_slave_directions(self):
        tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        nodes = np.vstack([tetra, tetra])
        cells = np.arange(8).reshape(2, 4)
        elevated = elevate(nodes, cells, cells[:, :3], np.array([True, False]))
        diagnostic = {"boundary_id_map": {"bottom": 2}, "rock_outer_support": {"faces": {
            "bottom": {"outward_normal": [0., 0., 1.], "normal_stiffness_n_per_m3": 10.}}}}
        for slave, master in ((1, 2), (2, 1)):
            blocks = {
                slave: f"Target Boundaries(1) = {slave}\nMortar BC = Integer {master}\n" +
                    "\n".join(f"Mortar BC Coefficient Displacement {i} = Real {i}" for i in (1, 2, 3)),
                master: f"Target Boundaries(1) = {master}",
            }
            blocks[2] += "\nSpring 33 = Real 10"
            pairs, values, audit = operators(elevated["nodes"], elevated["tri6"],
                                              np.array([1, 2]), blocks, diagnostic)
            self.assertTrue(np.isfinite(values).all())
            self.assertGreater(len(pairs), 0)
            self.assertEqual(audit["interfaces"][0]["slave_trace_nodes"], 6 if slave == 1 else 3)
            self.assertEqual(audit["interfaces"][0]["master_trace_nodes"], 3 if slave == 1 else 6)
            self.assertEqual(audit["supports"][0]["trace_nodes"], 3)

    def test_wall_only_elevation_preserves_linear_split_bodies_and_every_cell(self):
        tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        nodes = np.vstack([tetra, tetra, tetra])
        cells = np.arange(12).reshape(3, 4)
        result = elevate(nodes, cells, cells[:, :3], np.array([True, False, False]))
        self.assertEqual(len(result["nodes"]), 18)
        np.testing.assert_array_equal(result["tetra10"][:, :4], cells)
        np.testing.assert_array_equal(result["tetra10"][1:, 4:], -1)
        np.testing.assert_array_equal(result["tri6"][1:, 3:], -1)
        np.testing.assert_array_equal(result["tri6"][0, 3:], result["tetra10"][0, 4:7])
        np.testing.assert_array_equal(result["nodes"][:12], nodes)

    def test_mixed_order_rejects_shared_body_nodes_and_invalid_mask(self):
        nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                          [0., 0., 1.], [0., 0., -1.]])
        cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        with self.assertRaisesRegex(ValueError, "hanging nodes"):
            elevate(nodes, cells, cells[:, :3], np.array([True, False]))
        for mask in (np.array([1, 0]), np.array([True]), np.array([False, False])):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "mask"):
                elevate(nodes, cells, cells[:, :3], mask)

    def test_degree_elevation_shares_only_topological_edges(self):
        nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                          [0., 0., 1.], [0., 0., -1.]])
        cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        result = elevate(nodes, cells, np.array([[0, 1, 2], [0, 2, 1]]))
        self.assertEqual(result["nodes"].shape, (14, 3))
        np.testing.assert_array_equal(result["tetra10"][:, :4], cells)
        first, second = result["tri6"]
        self.assertEqual(set(first), set(second))
        np.testing.assert_allclose(result["nodes"][first[3:]],
                                   [[.5, 0, 0], [.5, .5, 0], [0, .5, 0]])

    def test_coincident_interface_sides_remain_independent_including_midsides(self):
        tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        nodes = np.vstack([tetra, tetra])
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        result = elevate(nodes, cells, np.array([[0, 1, 2], [4, 5, 6]]))
        self.assertEqual(len(result["nodes"]), 20)
        first, second = result["tri6"]
        self.assertEqual(set(first) & set(second), set())
        np.testing.assert_array_equal(result["nodes"][first], result["nodes"][second])

    def test_surface_without_volume_edge_is_rejected(self):
        nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                          [3., 3., 3.]])
        with self.assertRaisesRegex(ValueError, "volume topology"):
            elevate(nodes, np.array([[0, 1, 2, 3]]), np.array([[0, 1, 4]]))


if __name__ == "__main__":
    unittest.main()
