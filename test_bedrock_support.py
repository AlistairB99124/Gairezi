"""Independent stiffness, orientation, energy and invalid-input checks."""
import unittest

import numpy as np

from bedrock_support import consistent_face_mass, elastic_layer_tensors, lumped_layer_support, reduced_layer_arrays


class BedrockSupportTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array(((0., 0., 0.), (2., 0., 0.), (0., 3., 0.)))
        self.faces = np.array(((0, 1, 2),))
        self.properties = dict(youngs_modulus=50e9, poisson_ratio=.25, bottom_z_m=-10.)

    def test_exact_horizontal_coefficients_and_units(self):
        result = elastic_layer_tensors(self.points, self.faces, **self.properties)
        np.testing.assert_allclose(result["stiffness_n_per_m3"][0], np.diag((2e9, 2e9, 6e9)))
        self.assertEqual(result["effective_depth_m"][0], 10.)
        self.assertEqual(result["face_area_m2"][0], 3.)

    def test_supplied_reference_node_units_and_mesh_area_scaling(self):
        coefficients = {"x": 2e10, "y": 2e10, "z": 53333333000.}
        # Each node receives 0.01 m2 from this 0.03 m2 triangle.
        points = np.array(((0., 0., 0.), (.2, 0., 0.), (0., .3, 0.)))
        support = lumped_layer_support(points, self.faces, **self.properties,
                                       spring_n_per_m3=coefficients)
        np.testing.assert_allclose(np.diag(support["support_matrices_n_per_m"][0]),
                                   (200000000., 200000000., 533333330.))
        scaled = lumped_layer_support(points * 2, self.faces, **self.properties,
                                      spring_n_per_m3=coefficients)
        np.testing.assert_allclose(scaled["support_matrices_n_per_m"],
                                   4 * support["support_matrices_n_per_m"])
        sloped = points.copy()
        sloped[2, 2] = .1
        tensor = elastic_layer_tensors(sloped, self.faces, **self.properties,
                                      spring_n_per_m3=coefficients)["stiffness_n_per_m3"][0]
        np.testing.assert_array_equal(tensor, np.diag((2e10, 2e10, 53333333000.)))

    def test_invalid_supplied_coefficients_rejected(self):
        for springs in ({"x": 0, "y": 2e10, "z": 5e10},
                        {"x": True, "y": 2e10, "z": 5e10}, {"x": 2e10}):
            with self.subTest(springs=springs), self.assertRaises(ValueError):
                elastic_layer_tensors(self.points, self.faces, **self.properties,
                                      spring_n_per_m3=springs)

    def test_orientation_invariant_and_positive_definite(self):
        points = self.points.copy()
        points[2, 2] = 2.
        result = elastic_layer_tensors(points, self.faces, **self.properties)
        reverse = elastic_layer_tensors(points, self.faces[:, ::-1], **self.properties)
        matrix = result["stiffness_n_per_m3"][0]
        np.testing.assert_allclose(matrix, reverse["stiffness_n_per_m3"][0])
        self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0))
        self.assertNotEqual(matrix[1, 2], 0.)

    def test_lumping_preserves_uniform_translation_energy(self):
        result = lumped_layer_support(self.points, self.faces, **self.properties)
        vector = np.array((.001, -.002, .003))
        energy = .5 * sum(vector @ matrix @ vector for matrix in result["support_matrices_n_per_m"])
        expected = .5 * result["face_area_m2"][0] * vector @ result["stiffness_n_per_m3"][0] @ vector
        self.assertAlmostEqual(energy, expected)
        np.testing.assert_array_equal(result["support_nodes"], (0, 1, 2))

    def test_shared_node_accumulates_face_contributions(self):
        points = np.vstack((self.points, (2., 3., 0.)))
        faces = np.array(((0, 1, 2), (1, 3, 2)))
        result = lumped_layer_support(points, faces, **self.properties)
        np.testing.assert_allclose(result["support_matrices_n_per_m"][1],
                                   2 * result["support_matrices_n_per_m"][0])

    def test_consistent_face_mass_and_aster_quadrature_preserve_affine_energy_and_resultant(self):
        from Aster.support_trace import add_consistent_support, WEIGHTS, render_constraints

        mass = consistent_face_mass(self.points, self.faces)[0]
        np.testing.assert_allclose(mass, (np.ones((3, 3)) + np.eye(3)) / 4)
        tensor = np.array(((2., .2, .3), (.2, 3., .4), (.3, .4, 4.))) * 1e9
        arrays = {"nodes": self.points.copy(), "source_node_ids": np.arange(3)}
        add_consistent_support(arrays, self.faces, tensor[None])
        interpolation = np.kron(WEIGHTS, np.eye(3))
        stiffness = np.zeros((9, 9))
        for index, matrix in enumerate(arrays["support_matrices_n_per_m"]):
            stiffness[3 * index:3 * index + 3, 3 * index:3 * index + 3] = matrix
        np.testing.assert_allclose(interpolation.T @ stiffness @ interpolation, np.kron(mass, tensor))
        displacement = np.array(((.001, -.002, .003), (.004, .002, -.001), (-.003, .001, .002)))
        force = (np.kron(mass, tensor) @ displacement.ravel()).reshape(3, 3).sum(axis=0)
        row_sum_force = sum(mass.sum(axis=1)[i] * tensor @ displacement[i] for i in range(3))
        np.testing.assert_allclose(force, row_sum_force, atol=1e-8)
        np.testing.assert_array_equal(arrays["source_node_ids"], (0, 1, 2, -1, -1, -1))
        compile(render_constraints(arrays), "<support-trace>", "exec")

    def test_normal_and_shear_loads_produce_compliant_equilibrium(self):
        points = self.points.copy()
        points[2, 2] = 2.
        result = lumped_layer_support(points, self.faces, **self.properties)
        cross = np.cross(points[1] - points[0], points[2] - points[0])
        normal = cross / np.linalg.norm(cross)
        tangent = np.array((1., 0., 0.))
        force = 3000 * normal + 1000 * tangent
        stiffness = result["support_matrices_n_per_m"].sum(axis=0)
        displacement = np.linalg.solve(stiffness, force)
        area = result["face_area_m2"][0]
        expected = (3000 * normal / result["normal_stiffness_n_per_m3"][0]
                    + 1000 * tangent / result["tangential_stiffness_n_per_m3"][0]) / area
        np.testing.assert_allclose(displacement, expected)
        np.testing.assert_allclose(stiffness @ displacement, force)
        self.assertGreater(np.linalg.norm(displacement), 0.)
        self.assertAlmostEqual(.5 * displacement @ stiffness @ displacement,
                               .5 * force @ displacement)

    def test_invalid_geometry_and_material_rejected(self):
        for replacement in (dict(youngs_modulus=0), dict(poisson_ratio=.5),
                            dict(bottom_z_m=0), dict(bottom_z_m=float("nan"))):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                elastic_layer_tensors(self.points, self.faces, **{**self.properties, **replacement})
        for faces in (np.array(((0, 0, 1),)), np.array(((0, 1, 9),)),
                      np.array(((0, 1, 2), (2, 1, 0))), np.empty((0, 3), dtype=int)):
            with self.subTest(faces=faces), self.assertRaises(ValueError):
                elastic_layer_tensors(self.points, faces, **self.properties)

    def test_reduced_arrays_keep_foundation_and_remove_rigid_constraints(self):
        points = np.vstack((self.points, (0., 0., 1.), (0., 0., -1.)))
        arrays = {
            "nodes": points, "source_node_ids": np.arange(5),
            "tetra4_concrete": np.array(((0, 1, 2, 3),)),
            "concrete_body_ids": np.array((3,)),
            "tetra4_bedrock": np.array(((0, 1, 2, 4),)),
        }
        config = {
            "material": {"bedrock": {"youngs_modulus": 50e9, "poisson_ratio": .25}},
            "geometry": {"bedrock": {"domain": {"bottom_z_m": -10.}}},
        }
        reduced, report = reduced_layer_arrays(arrays, config)
        self.assertEqual(len(reduced["nodes"]), 4)
        self.assertEqual(len(reduced["tetra4_bedrock"]), 0)
        np.testing.assert_array_equal(reduced["concrete_body_ids"], (3,))
        self.assertEqual(len(reduced["fixed_support_nodes"]), 0)
        np.testing.assert_array_equal(reduced["support_nodes"], (0, 1, 2))
        self.assertTrue(np.all(np.linalg.eigvalsh(reduced["support_matrices_n_per_m"]) > 0))
        self.assertEqual(report["model"], "elastic_layer")
        np.testing.assert_array_equal(arrays["nodes"], points)


if __name__ == "__main__":
    unittest.main()
