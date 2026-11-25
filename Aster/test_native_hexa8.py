import unittest

import numpy as np

from Aster import automatic_cohesive as native
from Aster.cohesive_continuation import assess_round
from Aster.mixed_topology import interior_wall_faces, native_screening


def two_hexes() -> dict:
    # z face numbering follows MED NORM_HEXA8.
    points = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        [1., 0., 0.], [2., 0., 0.], [2., 1., 0.], [1., 1., 0.],
        [1., 0., 1.], [2., 0., 1.], [2., 1., 1.], [1., 1., 1.],
    ])
    # Distinct coincident source nodes make the interface explicit.
    return {
        "nodes": points,
        "hexa8_wall": np.array([list(range(8)), list(range(8, 16))], dtype=np.int64),
        "tetra4_foundation": np.empty((0, 4), dtype=np.int64),
        "tetra4_bedrock": np.empty((0, 4), dtype=np.int64),
        "source_node_ids": np.arange(len(points), dtype=np.int64),
        "source_cell_rows": np.array([10, 11], dtype=np.int64),
        "tri3_upstream": np.empty((0, 3), dtype=np.int64),
        "tri3_downstream": np.empty((0, 3), dtype=np.int64),
        "tri3_crest": np.empty((0, 3), dtype=np.int64),
        "quad4_upstream": np.empty((0, 4), dtype=np.int64),
        "quad4_downstream": np.empty((0, 4), dtype=np.int64),
        "quad4_crest": np.empty((0, 4), dtype=np.int64),
        "support_nodes": np.empty(0, dtype=np.int64),
    }


class NativeHexa8Tests(unittest.TestCase):
    def test_quad_interior_inventory_uses_two_native_hex_owners(self):
        arrays = two_hexes()
        # Coincident nodes are welded for topology identity in the source contract.
        arrays["hexa8_wall"][1] = [1, 9, 10, 2, 5, 13, 14, 6]
        faces, owners = interior_wall_faces(arrays["hexa8_wall"])
        self.assertEqual(faces.shape, (1, 4))
        np.testing.assert_array_equal(owners, [[0, 1]])

    def test_screening_consumes_native_cell_tensors_without_p1_recovery(self):
        arrays = two_hexes()
        arrays["hexa8_wall"][1] = [1, 9, 10, 2, 5, 13, 14, 6]
        stress = np.zeros((2, 3, 3))
        stress[:, 0, 0] = [3., 1.]
        result = native_screening(
            arrays, stress, arrays["source_cell_rows"], tensile_strength_pa=2.)
        self.assertEqual(result["peak_source_row"], 10)
        self.assertEqual(result["selection"]["selected_source_rows"], [10, 11])
        self.assertEqual(result["facet_check"]["critical_facets"], 1)

    def test_native_split_creates_one_hexa8_joint_per_quad(self):
        arrays = two_hexes()
        arrays["hexa8_wall"][1] = [1, 9, 10, 2, 5, 13, 14, 6]
        faces, _ = interior_wall_faces(arrays["hexa8_wall"])
        split = native.split_for_joints(arrays, faces)
        self.assertEqual(split["joints"].shape, (1, 8))
        native.integrate_joint_faces(split)
        np.testing.assert_array_equal(split["arrays"]["hexa8_joints"], split["joints"])
        np.testing.assert_array_equal(split["arrays"]["joint_parent_faces"], [0])
        self.assertEqual(split["arrays"]["nodes"].shape[0], 20)

    def test_continuation_refuses_to_recover_hex_stress_from_displacement(self):
        arrays = two_hexes()
        with self.assertRaisesRegex(ValueError, "Code_Aster cell stress"):
            assess_round(
                arrays["nodes"], arrays["hexa8_wall"], np.ones(2, dtype=bool),
                np.zeros_like(arrays["nodes"]), np.array([[0, 1]]), np.array([False]),
                1., 30e9, .2)

    def test_crack_water_accepts_quad8_joint_and_connectivity(self):
        from Aster.hydraulic_loads import CrackWater
        nodes = np.array([
            [0., 0., 0.], [0., 1., 0.], [0., 1., 1.], [0., 0., 1.],
            [0., 0., 0.], [0., 1., 0.], [0., 1., 1.], [0., 0., 1.],
        ])
        arrays = {
            "nodes": nodes, "quad8_joints": np.array([list(range(8))]),
            "source_node_ids": np.arange(8),
            "tri3_upstream": np.empty((0, 3), dtype=int),
            "quad4_upstream": np.array([[0, 1, 2, 3]]),
            "tri3_downstream": np.empty((0, 3), dtype=int),
            "quad4_downstream": np.empty((0, 4), dtype=int),
        }
        water = CrackWater(arrays, upstream_z=2., gradient=9810., tolerance=1.e-9)
        displacement = np.zeros_like(nodes)
        displacement[:4, 0] = 1.e-6
        wet, forces = water.evaluate(displacement)
        np.testing.assert_array_equal(wet, [True])
        self.assertTrue(np.isfinite(forces).all())
        self.assertGreater(np.linalg.norm(forces), 0.)


if __name__ == "__main__":
    unittest.main()
