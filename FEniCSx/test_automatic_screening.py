import unittest

import numpy as np

from automatic_screening import (crack_base, crack_thickness, interior_wall_facet_normal_stress, lift_joints,
                                 max_damage_count, screen_cells, select_damage_cells, selection_enabled)

UNIT_TET = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])


def layered_box(nx, ny, nz):
    """Conforming tetrahedral box of unit cubes (Kuhn split), with horizontal faces on integer z."""
    from itertools import permutations
    index = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    coordinates = np.array([[i, j, k] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)],
                           dtype=float)
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for order in permutations(range(3)):
                    corner, tet = [i, j, k], [index(i, j, k)]
                    for axis in order:
                        corner[axis] += 1
                        tet.append(index(*corner))
                    points = coordinates[tet]
                    if np.linalg.det(points[1:] - points[0]) < 0:
                        tet[2], tet[3] = tet[3], tet[2]
                    cells.append(tet)
    return coordinates, np.array(cells, dtype=np.int64)


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

    def test_peak_in_bottom_wall_layer_shifts_the_band_above_the_seat(self):
        # Peak wall cell at 0 sits on the foundation (foundation at -1 and 0, no wall below):
        # the band takes the layer above (+1) so a wall-wall lift plane exists; wall at +2
        # and the foundation below the seat are excluded.
        offsets = (0., 1., 2., -1., 0., 1.)
        regions = np.array([1, 1, 1, 3, 3, 3], dtype=np.int32)
        coordinates = np.concatenate([UNIT_TET + [2. * index, 0., offset]
                                      for index, offset in enumerate(offsets)])
        cells = np.arange(4 * len(offsets)).reshape(len(offsets), 4)
        tensors = np.zeros((len(offsets), 3, 3))
        for index, value in enumerate((3e6, 1e6, 2.5e6, 1e5, 1e5, 1e5)):
            tensors[index] = np.diag([value, -1e5, -2e5])
        result = screen_cells(tensors, regions, np.arange(10, 16), coordinates, cells,
                              2e6, relative_tolerance=1e-3)
        selection = result["selection"]
        self.assertTrue(selection["band_shifted_above_seat"])
        self.assertEqual(selection["selected_source_rows"], [10, 11, 14, 15])
        self.assertEqual((selection["selected_wall_cells"], selection["selected_foundation_cells"]), (2, 2))
        self.assertAlmostEqual(selection["layer_centre_z_m"], 1.25)
        self.assertIn("layer above", selection["rule"])
        # With a wall layer below the peak the band is unchanged.
        regions[3] = 1
        result = screen_cells(tensors, regions, np.arange(10, 16), coordinates, cells,
                              2e6, relative_tolerance=1e-3)
        selection = result["selection"]
        self.assertFalse(selection["band_shifted_above_seat"])
        self.assertEqual(selection["selected_source_rows"], [10, 13, 14])
        self.assertAlmostEqual(selection["layer_centre_z_m"], .25)

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

    def test_lift_joints_report_the_horizontal_facets_between_selected_cells(self):
        # Two wall tets share the horizontal z=0 face (0,1,2); a third shares the vertical
        # x=0 face (0,2,3) with the first. Both planes lie in the two-layer band.
        coordinates = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [-1, 0, 0]])
        cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 3, 5]])
        tensors = np.zeros((3, 3, 3))
        tensors[0] = np.diag([1e5, 1e5, 3e6])
        tensors[1] = np.diag([1e5, 1e5, 1e6])
        tensors[2] = np.diag([3e6, 1e5, 1e5])
        joints = {"tensile_strength_pa": 2e5, "cohesive_fracture_energy_j_m2": 10.}
        result = screen_cells(tensors, np.array([1, 1, 1], dtype=np.int32), np.array([7, 8, 9]), coordinates,
                              cells, 2e6, relative_tolerance=1e-3, crack_base=True, lift_joints=joints)
        selection = result["selection"]
        self.assertEqual(selection["selected_source_rows"], [7, 8, 9])
        lift = selection["lift_joints"]
        self.assertEqual(lift["lift_joint_facets"], 1)
        self.assertEqual(lift["candidate_facets"], 2)
        self.assertEqual(lift["source_faces"], [[0, 1, 2]])
        self.assertAlmostEqual(lift["lift_joint_area_m2"], .5)
        self.assertEqual(lift["lift_plane_z_m"], [0., 0.])
        self.assertEqual(lift["law"], "JOINT_MECA_RUPT")
        self.assertAlmostEqual(lift["critical_opening_m"], 1e-4)
        self.assertEqual(lift["max_normal_deviation_deg"], 15.)
        self.assertIn("15 degrees", lift["rule"])
        # Only the vertical plane: no lift facet exists, which is a configuration error.
        with self.assertRaisesRegex(ValueError, "no near-horizontal"):
            screen_cells(tensors[[0, 2]], np.array([1, 1], dtype=np.int32), np.array([7, 9]),
                         coordinates, cells[[0, 2]], 2e6, relative_tolerance=1e-3, crack_base=True,
                         lift_joints=joints)
        with self.assertRaisesRegex(ValueError, "band selection"):
            screen_cells(tensors, np.array([1, 1, 1], dtype=np.int32), np.array([7, 8, 9]), coordinates, cells,
                         2e6, relative_tolerance=1e-3, max_damage_count=2, lift_joints=joints)
        with self.assertRaisesRegex(ValueError, "lift_joints"):
            screen_cells(tensors, np.array([1, 1, 1], dtype=np.int32), np.array([7, 8, 9]), coordinates, cells,
                         2e6, relative_tolerance=1e-3, crack_base=True, lift_joints={"tensile_strength_pa": 2e5})
        self.assertIsNone(lift_joints({"damage_selection": {}}))
        self.assertEqual(lift_joints({"damage_selection": {"lift_joints": joints}}), joints)

    def test_lift_plane_elevation_places_the_band_and_surveys_every_horizontal_plane(self):
        # 2 x 1 x 4 unit cubes (six tetrahedra each), so horizontal planes lie at z = 1, 2, 3.
        # The peak principal stress is in the bottom layer; the configured pour plane is not.
        coordinates, cells = layered_box(2, 1, 4)
        centroid_z = coordinates[cells].mean(axis=1)[:, 2]
        tensors = np.zeros((len(cells), 3, 3))
        tensors[:, 2, 2] = -1e5 * (4. - centroid_z)       # weight-like vertical compression
        tensors[:, 0, 2] = tensors[:, 2, 0] = 2e4          # uniform horizontal shear on z-planes
        tensors[centroid_z < 1, 0, 0] = 3e6                # peak tension low in the wall
        regions = np.ones(len(cells), dtype=np.int32)
        rows = np.arange(len(cells))
        joints = {"tensile_strength_pa": 2e4, "cohesive_fracture_energy_j_m2": 1., "law": "JOINT_MECA_FROT",
                  "friction_coefficient": .7, "plane_elevation_m": 2.2, "uplift": "none"}
        result = screen_cells(tensors, regions, rows, coordinates, cells, 2e6, relative_tolerance=1e-3,
                              lift_joints=joints)
        selection = result["selection"]
        self.assertEqual(selection["plane_z_m"], 2.)
        self.assertIn("z = 2 m", selection["rule"])
        self.assertFalse(selection["band_shifted_above_seat"])
        chosen = np.asarray(selection["selected_source_rows"])
        np.testing.assert_array_equal(chosen, np.flatnonzero((centroid_z > 1) & (centroid_z < 3)))
        lift = selection["lift_joints"]
        self.assertEqual(lift["lift_plane_z_m"], [2., 2.])
        self.assertEqual(lift["lift_joint_facets"], 4)     # two triangles per unit square
        self.assertAlmostEqual(lift["lift_joint_area_m2"], 2.)
        self.assertEqual((lift["law"], lift["friction_coefficient"], lift["plane_elevation_m"], lift["uplift"]),
                         ("JOINT_MECA_FROT", .7, 2.2, "none"))
        self.assertAlmostEqual(lift["adhesion_pa"], .7 * 2e4)
        survey = result["lift_plane_survey"]
        self.assertEqual([round(plane["z_m"], 9) for plane in survey], [1., 2., 3.])
        for plane in survey:
            self.assertAlmostEqual(plane["area_m2"], 2.)
            self.assertAlmostEqual(plane["shear_resultant_n"], 2e4 * 2.)
            self.assertEqual(plane["tension_area_fraction"], 0.)
        # Each facet averages its two owner cells: z = 2 lies between centroids near 1.5 and 2.5.
        middle = survey[1]
        self.assertAlmostEqual(middle["mean_vertical_stress_pa"], -2e5, delta=1e3)
        self.assertAlmostEqual(middle["shear_to_compression"], 2e4 / 2e5, delta=1e-3)
        self.assertLess(survey[2]["normal_compression_n"], survey[0]["normal_compression_n"])
        # Without an elevation the band snaps to the plane by the peak layer: the peak is in the
        # bottom layer, so the plane above it (z = 1) is used and the band is layers 0-1 and 1-2.
        automatic = screen_cells(tensors, regions, rows, coordinates, cells, 2e6, relative_tolerance=1e-3,
                                 lift_joints={key: value for key, value in joints.items()
                                              if key != "plane_elevation_m"})["selection"]
        self.assertEqual(automatic["plane_z_m"], 1.)
        self.assertTrue(automatic["band_shifted_above_seat"])
        self.assertIn("bottom wall layer", automatic["rule"])
        np.testing.assert_array_equal(automatic["selected_source_rows"], np.flatnonzero(centroid_z < 2))
        self.assertEqual(automatic["lift_joints"]["lift_plane_z_m"], [1., 1.])
        self.assertNotIn("plane_elevation_m", automatic["lift_joints"])
        # A peak higher up takes the plane under its layer.
        raised = tensors.copy()
        raised[:, 0, 0] = np.where((centroid_z > 2) & (centroid_z < 3), 3e6, 0.)
        upper = screen_cells(raised, regions, rows, coordinates, cells, 2e6, relative_tolerance=1e-3,
                             lift_joints={key: value for key, value in joints.items()
                                          if key != "plane_elevation_m"})["selection"]
        self.assertEqual((upper["plane_z_m"], upper["band_shifted_above_seat"]), (2., False))
        np.testing.assert_array_equal(upper["selected_source_rows"],
                                      np.flatnonzero((centroid_z > 1) & (centroid_z < 3)))
        with self.assertRaisesRegex(ValueError, "more than one element layer"):
            screen_cells(tensors, regions, rows, coordinates, cells, 2e6, relative_tolerance=1e-3,
                         lift_joints=dict(joints, plane_elevation_m=10.))
        with self.assertRaisesRegex(ValueError, "crack_base"):
            screen_cells(tensors, regions, rows, coordinates, cells, 2e6, relative_tolerance=1e-3,
                         crack_base=True, lift_joints=joints)
        self.assertIsNone(screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                                       relative_tolerance=1e-3)["lift_plane_survey"])

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

    def radial_fixture(self):
        # Four wall cells across the thickness (centroid radii ~76.25..79.25 along +y)
        # plus foundation pads beyond each face, all at the same elevation band.
        radii = (76.0, 77.0, 78.0, 79.0)
        offsets = [(0.0, r, 0.0) for r in radii] + [(0.0, 80.2, 0.0), (0.0, 74.8, 0.0)]
        regions = np.array([1, 1, 1, 1, 3, 3], dtype=np.int32)
        coordinates = np.concatenate([UNIT_TET + offset for offset in offsets])
        cells = np.arange(4 * len(offsets)).reshape(len(offsets), 4)
        tensors = np.zeros((len(offsets), 3, 3))
        for index, value in enumerate((3e6, 2.5e6, 2.4e6, 2.3e6, 1e5, 1e5)):
            tensors[index] = np.diag([value, -1e5, -2e5])
        return tensors, regions, np.arange(10, 10 + len(offsets)), coordinates, cells

    def test_crack_thickness_limits_the_band_from_the_chosen_face(self):
        tensors, regions, rows, coordinates, cells = self.radial_fixture()
        upstream = screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                                relative_tolerance=1e-3,
                                crack_thickness={"elements": 2, "from": "upstream"})["selection"]
        # Upstream = larger radius: the two outer wall cells plus the outer foundation pad.
        self.assertEqual(upstream["selected_source_rows"], [12, 13, 14])
        self.assertEqual(upstream["crack_thickness"]["max_x_elements"], 4)
        self.assertEqual(upstream["crack_thickness"]["effective_elements"], 2)
        self.assertIn("2 of 4 element-sized steps from the upstream face", upstream["rule"])
        downstream = screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                                  relative_tolerance=1e-3,
                                  crack_thickness={"elements": 1, "from": "downstream"})["selection"]
        # Downstream = smaller radius: the innermost wall cell plus the inner foundation pad.
        self.assertEqual(downstream["selected_source_rows"], [10, 15])
        full = screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                            relative_tolerance=1e-3,
                            crack_thickness={"elements": 99, "from": "upstream"})["selection"]
        unlimited = screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                                 relative_tolerance=1e-3)["selection"]
        # Clamped at the measured maximum = exactly the full-thickness selection.
        self.assertEqual(full["selected_source_rows"], unlimited["selected_source_rows"])
        self.assertEqual(full["crack_thickness"]["effective_elements"], 4)
        self.assertIsNone(unlimited["crack_thickness"])

    def test_crack_thickness_is_validated(self):
        tensors, regions, rows, coordinates, cells = fixture([3e6, 1e6], [1, 1])
        for invalid in ({"elements": 0, "from": "upstream"}, {"elements": 1.5, "from": "upstream"},
                        {"elements": True, "from": "upstream"}, {"elements": 2, "from": "sideways"},
                        {"elements": 2}, {"from": "upstream"}, {"elements": 2, "from": "upstream", "x": 1}, []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                             relative_tolerance=1e-3, crack_thickness=invalid)
        with self.assertRaisesRegex(ValueError, "max_damage_count"):
            screen_cells(tensors, regions, rows, coordinates, cells, 2e6,
                         relative_tolerance=1e-3, max_damage_count=2,
                         crack_thickness={"elements": 1, "from": "upstream"})

    def test_crack_thickness_config_helper_defaults_to_none(self):
        self.assertIsNone(crack_thickness({}))
        self.assertIsNone(crack_thickness({"damage_selection": {}}))
        self.assertEqual(crack_thickness({"damage_selection": {"crack_thickness": {
            "elements": 2, "from": "upstream"}}}), {"elements": 2, "from": "upstream"})


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
