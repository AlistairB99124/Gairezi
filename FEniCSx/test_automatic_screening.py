import unittest

import numpy as np

from automatic_screening import (crack_base, interior_wall_facet_normal_stress, max_damage_count,
                                 screen_cells, select_damage_cells, selection_enabled)

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
        selection = result["selection"]
        self.assertEqual((selection["eligible_cells"], selection["selected_cells"]), (2, 2))
        self.assertEqual(selection["selected_source_rows"], [7, 8])
        self.assertIsNone(selection["max_damage_count"])
        capped = screen_cells(tensors, np.array([1, 1], dtype=np.int32), np.array([7, 8]), coordinates, cells,
                              2e6, relative_tolerance=1e-3, max_damage_count=2)["selection"]
        self.assertEqual((capped["max_damage_count"], capped["selected_facets"], capped["capped"]), (2, 1, False))

    def test_uncapped_selection_takes_peak_layer_plus_below_including_foundation(self):
        # Disjoint unit tets at z offsets: peak wall cell at 0, wall below at -1 (selected),
        # wall above at +1 (excluded), foundation at the peak elevation (selected),
        # and a region-2 cell at the peak elevation (never selected).
        offsets = (0., -1., 1., 0., 0.)
        regions = np.array([1, 1, 1, 3, 2], dtype=np.int32)
        coordinates = np.concatenate([UNIT_TET + [2. * index, 0., offset]
                                      for index, offset in enumerate(offsets)])
        cells = np.arange(4 * len(offsets)).reshape(len(offsets), 4)
        tensors = np.zeros((len(offsets), 3, 3))
        for index, value in enumerate((3e6, 1e6, 2.5e6, 4e6, 5e6)):
            tensors[index] = np.diag([value, -1e5, -2e5])
        result = screen_cells(tensors, regions, np.arange(10, 15), coordinates, cells,
                              2e6, relative_tolerance=1e-3)
        selection = result["selection"]
        self.assertIsNone(selection["max_damage_count"])
        self.assertEqual(selection["selected_source_rows"], [10, 11, 13])
        self.assertEqual(selection["selected_wall_cells"], 2)
        self.assertEqual(selection["selected_foundation_cells"], 1)
        self.assertEqual(selection["selected_cells"], 3)
        self.assertAlmostEqual(selection["layer_centre_z_m"], .25)
        self.assertAlmostEqual(selection["layer_band_height_m"], 1.)
        self.assertIn("foundation", selection["rule"])
        self.assertFalse(selection["crack_base"])

    def test_crack_base_selects_the_bottom_two_wall_layers(self):
        # Peak tension sits in the top layer (+2), but crack_base anchors the band at the
        # lowest wall corner (0): layers at 0 and +1 are selected (wall and foundation),
        # the peak layer and a region-2 cell are not.
        offsets = (2., 0., 1., 0., 0.)
        regions = np.array([1, 1, 1, 3, 2], dtype=np.int32)
        coordinates = np.concatenate([UNIT_TET + [2. * index, 0., offset]
                                      for index, offset in enumerate(offsets)])
        cells = np.arange(4 * len(offsets)).reshape(len(offsets), 4)
        tensors = np.zeros((len(offsets), 3, 3))
        for index, value in enumerate((5e6, 1e6, 2e6, 3e6, 4e6)):
            tensors[index] = np.diag([value, -1e5, -2e5])
        result = screen_cells(tensors, regions, np.arange(10, 15), coordinates, cells,
                              2e6, relative_tolerance=1e-3, crack_base=True)
        selection = result["selection"]
        self.assertTrue(selection["crack_base"])
        self.assertEqual(selection["selected_source_rows"], [11, 12, 13])
        self.assertEqual(selection["selected_wall_cells"], 2)
        self.assertEqual(selection["selected_foundation_cells"], 1)
        self.assertIn("base of the wall", selection["rule"])
        self.assertAlmostEqual(selection["layer_centre_z_m"], 1.5)

    def test_crack_base_rejects_a_capped_count_and_non_booleans(self):
        tensors, regions, rows, coordinates, cells = fixture([3e6, 1e6], [1, 1])
        with self.assertRaisesRegex(ValueError, "max_damage_count"):
            screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                         relative_tolerance=1e-3, max_damage_count=2, crack_base=True)
        with self.assertRaisesRegex(ValueError, "boolean"):
            screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                         relative_tolerance=1e-3, crack_base=1)

    def test_crack_base_config_helper_defaults_to_false(self):
        self.assertFalse(crack_base({}))
        self.assertFalse(crack_base({"damage_selection": {}}))
        self.assertTrue(crack_base({"damage_selection": {"crack_base": True}}))

    def test_disabled_selection_selects_nothing_despite_overstress(self):
        tensors, regions, rows, coordinates, cells = fixture([3e6, 1e6], [1, 1])
        result = screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                              relative_tolerance=1e-3, selection_enabled=False)
        selection = result["selection"]
        self.assertIs(selection["enabled"], False)
        self.assertEqual(selection["selected_source_rows"], [])
        self.assertEqual(selection["selected_cells"], 0)
        self.assertIn("disabled", selection["rule"])
        # Screening itself still reports the overstress; only the selection is empty.
        self.assertEqual(result["status"], "overstressed_cells_found")
        with self.assertRaisesRegex(ValueError, "boolean"):
            screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                         relative_tolerance=1e-3, selection_enabled=1)

    def test_selection_enabled_config_helper_defaults_to_true(self):
        self.assertTrue(selection_enabled({}))
        self.assertTrue(selection_enabled({"damage_selection": {}}))
        self.assertTrue(selection_enabled({"damage_selection": {"enabled": True}}))
        self.assertFalse(selection_enabled({"damage_selection": {"enabled": False}}))


# Five critical facets over cells 0..6 (facet stress, owners); facet 5 is below strength.
PAIRS = np.array([[0, 1], [1, 2], [3, 4], [5, 6], [2, 3], [0, 6]])
STRESS = np.array([3.0e6, 2.5e6, 2.5e6, 2.2e6, 2.1e6, 1.0e6])
CRITICAL = STRESS >= 2e6
ROWS = np.arange(100, 107)


def octahedron():
    """Eight wall tetrahedra around an interior centre node: its star is 8 cells, each axis node's is 4."""
    axes = np.array([[1., 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    coordinates = np.vstack(([0., 0, 0], axes))
    cells = np.array([[0, x, y, z] for x in (1, 2) for y in (3, 4) for z in (5, 6)])
    return coordinates, cells, np.ones(8, dtype=np.int32)


def uniform(count, value=3e6):
    return np.repeat(np.diag([value, value, value])[None], count, axis=0)


def capped(count, regions=None, stress=None):
    coordinates, cells, wall = octahedron()
    regions = wall if regions is None else regions
    pairs, facet = interior_wall_facet_normal_stress(coordinates, cells, regions,
                                                     uniform(8) if stress is None else stress)
    return select_damage_cells(pairs, facet, facet >= 2e6, np.arange(200, 208), count,
                               coordinates_m=coordinates, tetrahedra=cells, regions=regions)


class CappedPatchSelectionTest(unittest.TestCase):
    def test_whole_node_star_is_selected_and_openable(self):
        result = capped(8)
        self.assertEqual(result["selected_source_rows"], list(range(200, 208)))
        self.assertEqual((result["selected_cells"], result["selected_facets"]), (8, 12))
        self.assertEqual((result["candidate_facets"], result["openable_facets"]), (12, 12))
        self.assertTrue(result["openable"])
        self.assertIsNone(result["message"])
        self.assertEqual(result["patch_nodes"], 1)
        self.assertFalse(result["capped"])

    def test_cap_takes_the_most_tensioned_star_that_fits(self):
        # The centre star (8) does not fit 4; the +x star (4 cells) carries the highest tension.
        stress = uniform(8)
        stress[:4] = np.diag([4e6, 4e6, 4e6])
        result = capped(4, stress=stress)
        self.assertEqual(result["selected_source_rows"], [200, 201, 202, 203])
        self.assertEqual((result["selected_facets"], result["openable_facets"]), (4, 4))
        self.assertTrue(result["openable"])
        self.assertTrue(result["capped"])
        self.assertEqual(result["max_damage_count"], 4)
        # Facets shared with a -x cell keep the +x cell's 4 MPa (one-sided) but lie outside the patch.
        self.assertEqual(result["peak_unselected_facet_stress_pa"], 4e6)

    def test_count_too_small_for_any_star_is_not_openable(self):
        result = capped(3)
        self.assertEqual(result["selected_cells"], 0)
        self.assertFalse(result["openable"])
        self.assertIn("Increase Max Damage Count", result["message"])

    def test_nodes_touching_other_regions_are_skipped(self):
        regions = np.ones(8, dtype=np.int32)
        regions[7] = 3
        result = capped(8, regions=regions)
        self.assertNotIn(207, result["selected_source_rows"])
        self.assertLessEqual(result["selected_cells"], 8)
        self.assertTrue(result["openable"])


class DamageSelectionTest(unittest.TestCase):
    def test_blank_count_selects_every_cell_owning_a_critical_facet(self):
        result = select_damage_cells(PAIRS, STRESS, CRITICAL, ROWS, None)
        self.assertEqual(result["selected_source_rows"], list(range(100, 107)))
        self.assertEqual((result["eligible_cells"], result["eligible_facets"]), (7, 5))
        self.assertEqual((result["selected_cells"], result["selected_facets"]), (7, 5))
        self.assertFalse(result["capped"])
        self.assertEqual(result["lowest_selected_facet_stress_pa"], 2.1e6)
        self.assertIsNone(result["peak_unselected_facet_stress_pa"])

    def test_capped_selection_needs_the_mesh(self):
        with self.assertRaisesRegex(ValueError, "needs the mesh"):
            select_damage_cells(PAIRS, STRESS, CRITICAL, ROWS, 5)

    def test_no_critical_facet_selects_nothing(self):
        coordinates, cells, regions = octahedron()
        pairs, stress = interior_wall_facet_normal_stress(coordinates, cells, regions, uniform(8))
        result = select_damage_cells(pairs, stress, np.zeros(len(pairs), dtype=bool), np.arange(8), 10,
                                     coordinates_m=coordinates, tetrahedra=cells, regions=regions)
        self.assertEqual((result["eligible_cells"], result["selected_cells"]), (0, 0))
        self.assertEqual(result["selected_source_rows"], [])
        self.assertIsNone(result["lowest_selected_facet_stress_pa"])
        self.assertIsNone(result["openable"])
        self.assertIsNone(result["message"])
        self.assertEqual(select_damage_cells(np.empty((0, 2), int), [], [], ROWS, None)["selected_cells"], 0)

    def test_invalid_count_is_rejected(self):
        for count in (0, 1, 2.5, True):
            with self.assertRaises(ValueError):
                select_damage_cells(PAIRS, STRESS, CRITICAL, ROWS, count)

    def test_count_is_read_from_the_damage_selection_block(self):
        self.assertIsNone(max_damage_count({}))
        self.assertIsNone(max_damage_count({"damage_selection": {"max_damage_count": None}}))
        self.assertEqual(max_damage_count({"damage_selection": {"max_damage_count": 40}}), 40)


class AutomaticScreeningInputTest(unittest.TestCase):
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
