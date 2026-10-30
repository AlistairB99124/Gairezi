import unittest

import numpy as np

from automatic_screening import screen_cells

UNIT_TET = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])


def fixture(stresses, regions):
    count = len(stresses)
    coordinates = np.concatenate([UNIT_TET + [2. * index, 0., 0.] for index in range(count)])
    cells = np.arange(4 * count).reshape(count, 4)
    tensors = np.zeros((count, 3, 3))
    for index, value in enumerate(stresses):
        tensors[index] = np.diag([value, -1e5, -2e5])
    return tensors, np.asarray(regions, dtype=np.int32), np.arange(10, 10 + count), coordinates, cells


class AutomaticScreeningTest(unittest.TestCase):
    def test_classifies_wall_cells_and_excludes_other_regions(self):
        tensors, regions, rows, coordinates, cells = fixture(
            [3e6, 1.9999e6, 1e6, 5e6, 2.5e6], [1, 1, 1, 3, 1])
        result = screen_cells(tensors, regions, rows, coordinates, cells, 2e6, relative_tolerance=1e-3)
        self.assertEqual(result["status"], "overstressed_cells_found")
        self.assertEqual(result["wall_cells"], 4)
        self.assertEqual(result["overstressed_cells"], 2)
        self.assertEqual(result["near_critical_cells"], 1)
        self.assertEqual([item["source_row"] for item in result["overstressed"]], [10, 14])
        self.assertEqual(result["overstressed_source_rows"], [10, 14])
        self.assertEqual(result["peak_source_row"], 10)
        self.assertAlmostEqual(result["peak_ratio_to_strength"], 1.5)
        self.assertAlmostEqual(result["overstressed_volume_m3"], 2 / 6)
        np.testing.assert_allclose(np.abs(result["peak_principal_direction"]), [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(result["peak_location_m"], [.25, .25, .25])
        self.assertFalse(result["is_final_crack_prediction"])
        self.assertFalse(result["solver_ready"])

    def test_rotated_tensor_uses_principal_not_component_stress(self):
        tensors, regions, rows, coordinates, cells = fixture([0.], [1])
        tensors[0] = [[0., 2.5e6, 0.], [2.5e6, 0., 0.], [0., 0., 0.]]
        result = screen_cells(tensors, regions, rows, coordinates, cells, 2e6, relative_tolerance=1e-6)
        self.assertEqual(result["overstressed_cells"], 1)
        self.assertAlmostEqual(result["peak_principal_stress_pa"], 2.5e6)

    def test_below_strength_reports_no_initiation(self):
        tensors, regions, rows, coordinates, cells = fixture([1.051e6, -3e6], [1, 1])
        result = screen_cells(tensors, regions, rows, coordinates, cells, 2e6, relative_tolerance=1e-6)
        self.assertEqual(result["status"], "no_overstressed_cells")
        self.assertEqual(result["overstressed"], [])
        self.assertEqual(result["facet_check"]["interior_wall_facets"], 0)
        self.assertFalse(result["facet_check"]["initiates"])

    def test_facet_check_uses_the_more_stressed_owner_cell(self):
        # A one-cell-thick tension skin: averaging would dilute 3 MPa with the relaxed neighbour.
        coordinates = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
        cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
        tensors = np.stack([np.diag([0., 0, 3e6]), np.diag([0., 0, 0.])])
        result = screen_cells(tensors, np.array([1, 1], dtype=np.int32), np.array([7, 8]), coordinates, cells,
                              2e6, relative_tolerance=1e-3)
        check = result["facet_check"]
        self.assertEqual(check["interior_wall_facets"], 1)
        self.assertEqual(check["critical_facets"], 1)
        self.assertTrue(check["initiates"])
        self.assertAlmostEqual(check["peak_facet_normal_stress_pa"], 3e6)
        self.assertAlmostEqual(check["peak_two_cell_average_facet_normal_stress_pa"], 1.5e6)

    def test_rejects_invalid_inputs(self):
        tensors, regions, rows, coordinates, cells = fixture([1e6, 1e6], [1, 1])
        asymmetric = tensors.copy()
        asymmetric[0, 0, 1] = 1e6
        bad_cases = [
            (asymmetric, regions, rows, coordinates, cells, 2e6, 1e-6),
            (tensors, regions, np.array([1, -1]), coordinates, cells, 2e6, 1e-6),
            (tensors, np.array([2, 2], dtype=np.int32), rows, coordinates, cells, 2e6, 1e-6),
            (tensors, regions, rows, coordinates, cells, 0., 1e-6),
            (tensors, regions, rows, coordinates, cells, 2e6, 0.),
            (tensors, regions, rows, coordinates, cells + 100, 2e6, 1e-6),
            (tensors[:1], regions, rows, coordinates, cells, 2e6, 1e-6),
        ]
        for stress, region, row, xyz, tets, strength, tolerance in bad_cases:
            with self.assertRaises(ValueError):
                screen_cells(stress, region, row, xyz, tets, strength, relative_tolerance=tolerance)


if __name__ == "__main__":
    unittest.main()
