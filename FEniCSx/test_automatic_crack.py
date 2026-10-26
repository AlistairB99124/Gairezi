"""Wall crack placement against the bedrock contour and stress-selection map."""
import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from automatic_crack import SeedGeometry, contour_intersections, locate_crack, snap_crack_location
from crack_field import CrackSpecification, signed_distance
from crack_selection import selection_map

CONFIG = json.loads((Path(__file__).parents[1] / "config.json").read_text())


def synthetic_fields(upstream_tension=True):
    """Wall-only baseline fields over surveyed bedrock, with tension on one radial face."""
    contours = CONFIG["geometry"]["bedrock"]["contours"]
    chainages, elevations = [row["chainage"] for row in contours], [row["height"] for row in contours]
    s, z, r = np.meshgrid(
        np.arange(85, 125.01, 0.5), np.arange(-29, -19.95, 0.1), np.arange(76, 80.01, 1.0),
        indexing="ij",
    )
    s, z, r = s.ravel(), z.ravel(), r.ravel()
    base = np.interp(s, chainages, elevations)
    regions = np.where((z >= base) & (z <= 0), 1, 0)
    keep = regions == 1
    s, z, r, regions = s[keep], z[keep], r[keep], regions[keep]
    theta = s / 78
    points = np.column_stack((r * np.sin(theta), r * np.cos(theta), z))
    stress = np.zeros((len(points), 6))
    stress[:, 0] = np.where((r >= 78) == upstream_tension, 2e6, 0.5e6)
    stress[:, 1] = -3e6
    return points, stress, regions


def location(chainage, z):
    return (78 * math.cos(chainage / 78), z)


class StrengthCalibrationTests(unittest.TestCase):
    def test_length_scale_round_trips_to_the_target_strength(self):
        from crack_contract import implied_tensile_strength, strength_calibrated_length_scale
        for model in ("AT1", "AT2"):
            ell = strength_calibrated_length_scale(model, 35e9, 200.0, 2e6)
            self.assertAlmostEqual(implied_tensile_strength(model, 35e9, 200.0, ell), 2e6, delta=1e-3)
        self.assertAlmostEqual(strength_calibrated_length_scale("AT1", 35e9, 200.0, 2e6), 0.65625)
        # The previous AT2 default (l = 0.25 m) implied about 1.72 MPa.
        self.assertAlmostEqual(implied_tensile_strength("AT2", 35e9, 200.0, 0.25) / 1e6, 1.718, places=3)

    def test_rejects_unknown_models_and_non_positive_inputs(self):
        from crack_contract import strength_calibrated_length_scale
        with self.assertRaises(ValueError):
            strength_calibrated_length_scale("AT3", 35e9, 200.0, 2e6)
        with self.assertRaises(ValueError):
            strength_calibrated_length_scale("AT1", 35e9, 200.0, 0.0)


class UserLocatedCrackTests(unittest.TestCase):
    def setUp(self):
        # These cases check a constant 4 m wall; stepped walls are covered separately.
        self.config = copy.deepcopy(CONFIG)
        # Null fracture_debug: the shipped fracture_defaults.json apply.
        self.config["fracture_debug"] = None
        self.config["geometry"]["wall"].pop("thickness_steps", None)
        self.fields = synthetic_fields()

    def test_crack_carries_upstream_hydrostatic_water(self):
        # Reservoir surface z = -29 + 29 + 3 = 3 m (Elmer datums); the seed is always upstream.
        for fields in (self.fields, synthetic_fields(False)):
            crack, _, report = locate_crack(self.config, location(108.5, -27.5), fields)
            self.assertEqual(report["seed_face"], "upstream")
            self.assertAlmostEqual(crack.internal_pressure_pa, 9810 * 30.5)
            self.assertEqual(report["crack_water_pressure"]["water_surface_z_m"], 3.0)
            self.assertEqual(report["report_crack"]["loading"]["internal_pressure_pa"], crack.internal_pressure_pa)
        crack, _, _ = locate_crack(self.config, location(108.5, -22.0), self.fields)
        self.assertAlmostEqual(crack.internal_pressure_pa, 9810 * 25.0)
    def test_crack_water_datums_match_the_elmer_water_loads(self):
        from automatic_crack import hydrostatic_crack_pressure
        from Elmer.regions.combined_structure import load_environment
        loads = load_environment(Path(__file__).parents[1] / "config.json")
        upstream = hydrostatic_crack_pressure(CONFIG, "upstream", -15.0)
        self.assertAlmostEqual(upstream["water_surface_z_m"], loads.upstream_surface_z_m)
        self.assertAlmostEqual(upstream["pressure_pa"], loads.upstream_pressure_pa(-15.0))
        with self.assertRaises(ValueError):
            hydrostatic_crack_pressure(CONFIG, "crest", -15.0)

    def test_seed_depth_is_a_fixed_3_m_from_the_upstream_face_on_every_wall_step(self):
        for steps in ([], [{"top_z_m": -10.0, "radial_width_m": 4.5}, {"top_z_m": -20.0, "radial_width_m": 5.0}],
                      [{"top_z_m": -20.0, "radial_width_m": 6.0}]):
            self.config["geometry"]["wall"]["thickness_steps"] = steps
            with self.subTest(steps=steps):
                crack, _, report = locate_crack(self.config, location(108.5, -27.5), self.fields)
                self.assertEqual(crack.annular_sector[:2], (77, 80))
                self.assertEqual(report["depth_m"], 3)
                self.assertNotIn("depth_fraction", report)

    def test_seed_deeper_than_the_wall_fails(self):
        self.config["geometry"]["wall"]["thickness_steps"] = [{"top_z_m": -20.0, "radial_width_m": 3.0}]
        with self.assertRaisesRegex(ValueError, "less than the 3 m wall thickness"):
            locate_crack(self.config, location(108.5, -27.5), self.fields)

    def test_missing_cracking_block_uses_the_fracture_defaults(self):
        from automatic_crack import cracking_settings
        expected = {"face": "upstream", "depth_m": 3.0, "opening_m": 0.002,
                    "full_width": True, "buffer_m": 1.0, "length_m": None}
        self.assertEqual(cracking_settings(self.config), expected)
        legacy = copy.deepcopy(self.config)
        del legacy["cracking"]
        self.assertEqual(cracking_settings(legacy), expected)
        self.assertEqual(locate_crack(legacy, location(108.5, -27.5), self.fields)[0],
                         locate_crack(self.config, location(108.5, -27.5), self.fields)[0])

    def test_downstream_seed_opens_from_the_local_downstream_face_with_tailwater(self):
        self.config["cracking"][0].update(face="downstream", depth_m=1.0)
        crack, _, report = locate_crack(self.config, location(108.5, -27.5), self.fields)
        self.assertEqual(report["seed_face"], "downstream")
        self.assertEqual(crack.annular_sector[:2], (76, 77))
        # Tailwater surface z = -29 + 3 = -26 m.
        self.assertAlmostEqual(crack.internal_pressure_pa, 9810 * 1.5)
        self.config["geometry"]["wall"]["thickness_steps"] = [{"top_z_m": -20.0, "radial_width_m": 5.0}]
        crack, _, _ = locate_crack(self.config, location(108.5, -27.5), self.fields)
        self.assertEqual(crack.annular_sector[:2], (75, 76))

    def test_opening_sets_the_seed_band_thickness(self):
        self.config["cracking"][0]["opening_m"] = 0.005
        crack, _, report = locate_crack(self.config, location(108.5, -27.5), self.fields)
        self.assertEqual(crack.size_m[2], 0.005)
        self.assertEqual(report["opening_m"], 0.005)
        self.assertEqual(report["report_crack"]["size_m"]["z"], 0.005)

    def test_partial_length_is_centred_on_the_selection_and_clipped_to_the_buffer(self):
        full = locate_crack(self.config, location(108.5, -27.5), self.fields)[2]["crack_chainage_m"]
        self.config["cracking"][0].update(full_width=False, length_m=4.0)
        crack, _, report = locate_crack(self.config, location(108.5, -27.5), self.fields)
        np.testing.assert_allclose(report["crack_chainage_m"], [106.5, 110.5])
        np.testing.assert_allclose(np.array(crack.annular_sector[2:]) * 78, [106.5, 110.5])
        self.assertEqual(report["warnings"], [])
        self.config["cracking"][0]["length_m"] = 500.0
        _, _, report = locate_crack(self.config, location(108.5, -27.5), self.fields)
        np.testing.assert_allclose(report["crack_chainage_m"], full)
        self.assertIn("clipped", report["warnings"][0])

    def test_zero_buffer_keeps_a_small_clearance_inside_the_wall(self):
        self.config["cracking"][0]["buffer_m"] = 0.0
        y, z, snapped = snap_crack_location(self.config, location(108.5, -28.5))
        self.assertFalse(snapped)
        y, z, snapped = snap_crack_location(self.config, location(108.5, -29.5))
        self.assertTrue(snapped)
        self.assertAlmostEqual(z, -28.99)
        _, _, report = locate_crack(self.config, location(108.5, -28.5), self.fields)
        self.assertEqual(report["buffer_m"], 0.0)
        self.assertEqual(report["foundation_clearance_m"], 0.01)
        self.assertIn("minimum clearance", report["warnings"][0])

    def test_invalid_cracking_settings_fail(self):
        for update in ({"face": "crest"}, {"opening_m": 0.0}, {"buffer_m": -1.0},
                       {"full_width": False, "length_m": None}, {"length_m": -2.0}):
            config = copy.deepcopy(self.config)
            config["cracking"][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                locate_crack(config, location(108.5, -27.5), self.fields)

    def test_cracking_must_be_an_array_with_one_crack(self):
        crack = self.config["cracking"][0]
        for cracks in (crack, [], [crack, crack], ["upstream"]):
            config = copy.deepcopy(self.config)
            config["cracking"] = cracks
            with self.subTest(cracks=cracks), self.assertRaisesRegex(ValueError, "cracking"):
                SeedGeometry(config)

    def test_selected_point_seeds_wall_crack_at_user_chainage(self):
        crack, options, report = locate_crack(self.config, location(108.5, -27.5), self.fields)
        self.assertIs(type(crack), CrackSpecification)
        self.assertEqual(crack.name, "user_located_crack")
        self.assertEqual(report["seed_face"], "upstream")
        self.assertEqual(crack.annular_sector[:2], (77, 80))
        self.assertAlmostEqual(report["selected_chainage_m"], 108.5)
        extent = contour_intersections(self.config["geometry"]["bedrock"]["contours"], -27.5)
        np.testing.assert_allclose(report["concrete_chainage_extent_m"], extent)
        start, end = report["crack_chainage_m"]
        self.assertTrue(extent[0] < start < 108.5 < end < extent[1])
        self.assertAlmostEqual(report["arc_length_m"], end - start)
        self.assertEqual(report["required_region_ids"], [1])
        self.assertEqual(report["damaged_bodies"], ["wall"])
        self.assertEqual(report["depth_m"], 3)
        self.assertEqual(report["radial_extent_m"], 3)
        self.assertEqual(report["warnings"], [])
        self.assertEqual(crack.centre_m[2], -27.5)
        self.assertEqual(crack.size_m[2], 0.002)
        self.assertEqual(crack.model, "AT1")
        # l = 3 E Gc / (8 ft^2) with E = 35 GPa, Gc = 200 J/m2, ft = 2 MPa.
        self.assertAlmostEqual(crack.length_scale_m, 0.65625)
        calibration = report["length_scale_calibration"]
        self.assertAlmostEqual(calibration["implied_homogeneous_tensile_strength_pa"], 2.0e6)
        self.assertEqual(report["report_crack"]["phase_field_parameters"]["model"], "AT1")
        self.assertEqual(options.maximum_refined_cells, 4_000_000)
        theta = 108.5 / 78
        self.assertLess(signed_distance([79 * math.sin(theta), 79 * math.cos(theta), -27.5], crack), 0)
        self.assertIn("user selection", report["placement_provenance"]["elevation_source"])
        json.dumps(report, allow_nan=False)
        self.assertNotIn("crack_simulations", self.config)

    def test_seed_is_upstream_even_with_downstream_tension(self):
        crack, _, report = locate_crack(self.config, location(108.5, -27.5), synthetic_fields(False))
        self.assertEqual(report["seed_face"], "upstream")
        self.assertEqual(report["seed_face_rule"], "upstream (configured)")
        self.assertEqual(crack.annular_sector[:2], (77, 80))

    def test_crack_stops_one_metre_short_of_the_foundation(self):
        crack, _, report = locate_crack(self.config, location(108.5, -27.5), self.fields)
        geometry = SeedGeometry(self.config)
        start, end = report["crack_chainage_m"]
        concrete = report["concrete_chainage_extent_m"]
        self.assertGreater(start, concrete[0] + 1)
        self.assertLess(end, concrete[1] - 1)
        # Both ends sit on the 1 m clearance, measured at the seed's 77 m inner radius.
        ends = [(geometry.scale * s, -27.5) for s in (start, end)]
        np.testing.assert_allclose(geometry.clearance(ends), 1.0, atol=1e-9)
        along = np.linspace(start, end, 200)
        self.assertTrue(np.all(geometry.clearance(np.column_stack((geometry.scale * along, np.full(200, -27.5)))) >= 1 - 1e-9))
        np.testing.assert_allclose(np.array(crack.annular_sector[2:]) * 78, [start, end])
        self.assertEqual(report["foundation_clearance_m"], 1.0)
        self.assertFalse(report["snapped_to_foundation_clearance"])

    def test_selection_too_close_to_the_flat_base_snaps_up_to_the_clearance(self):
        # The bed is flat at -29 m from chainage 101 to 116, so Z=-28.5 snaps to -28 m.
        for z in (-28.5, -30.0):
            crack, _, report = locate_crack(self.config, location(108.5, z), self.fields)
            with self.subTest(z=z):
                self.assertTrue(report["snapped_to_foundation_clearance"])
                self.assertEqual(report["requested_location_m"]["z"], z)
                self.assertEqual(report["selected_location_m"], {"y": location(108.5, z)[0], "z": -28.0})
                self.assertEqual(crack.centre_m[2], -28.0)
                self.assertAlmostEqual(report["selected_chainage_m"], 108.5)

    def test_selection_near_a_sloping_contour_snaps_perpendicular_to_it(self):
        geometry = SeedGeometry(self.config)
        y, z, snapped = snap_crack_location(self.config, location(140, -21))
        self.assertTrue(snapped)
        chainage = 78 * math.acos(y / 78)
        self.assertAlmostEqual(float(geometry.clearance((geometry.scale * chainage, z))[0]), 1.0, places=9)
        # Perpendicular to the 131->146 m segment (rising 8 m over 15 m): up and back toward the valley.
        self.assertLess(chainage, 140)
        self.assertGreater(z, -21)
        self.assertAlmostEqual((z + 21) / (geometry.scale * (140 - chainage)), geometry.scale * 15 / 8, places=6)

    def test_valid_selection_is_unchanged_and_crest_clicks_snap_down(self):
        selected = location(108.5, -27.5)
        self.assertEqual(snap_crack_location(self.config, selected), (*selected, False))
        y, z, snapped = snap_crack_location(self.config, location(108.5, 1.0))
        self.assertTrue(snapped)
        self.assertEqual(z, 0.0)
        self.assertAlmostEqual(y, location(108.5, 1.0)[0])

    def test_peak_stress_near_the_selection_is_recorded_but_does_not_choose_the_face(self):
        points, stress, regions = self.fields
        chainage = 78 * np.arctan2(points[:, 0], points[:, 1])
        radius = np.hypot(points[:, 0], points[:, 1])
        # Downstream tension near the click; a larger upstream peak far along the same crack.
        stress = stress.copy()
        stress[:, 0] = np.where(radius >= 78, 0.5e6, 2e6)
        stress[(np.abs(chainage - 115) < 1) & (radius >= 78), 0] = 9e6
        _, _, report = locate_crack(self.config, location(108.5, -27.5), (points, stress, regions))
        self.assertEqual(report["seed_face"], "upstream")
        self.assertLess(abs(78 * math.atan2(*report["seed_face_source"]["position_xyz_m"][:2]) - 108.5), 2 + 1e-9)

    def test_invalid_or_outside_selection_fails_instead_of_moving(self):
        for selected, message in (
            ((float("nan"), -25), "finite"),
            ((79, -25), "outside the"),
            ((1, 2, 3), "two numbers"),
        ):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, message):
                locate_crack(self.config, selected, self.fields)
    def test_selection_between_coarse_node_layers_still_finds_the_wall(self):
        # Elmer wall nodes are about 0.5 m apart vertically; a click midway between
        # layers is farther than length_scale / 2 from every node.
        points, stress, regions = self.fields
        layer = np.isclose(points[:, 2] * 2, np.round(points[:, 2] * 2))
        coarse = (points[layer], stress[layer], regions[layer])
        crack, _, report = locate_crack(self.config, location(108.5, -26.75), coarse)
        self.assertEqual(report["damaged_bodies"], ["wall"])
        self.assertEqual(crack.centre_m[2], -26.75)

    def test_location_without_wall_baseline_nodes_fails(self):
        points, stress, regions = self.fields
        far = points[:, 2] < -25
        with self.assertRaisesRegex(ValueError, "No wall baseline nodes"):
            locate_crack(self.config, location(110, -20), (points[far], stress[far], regions[far]))

    def test_curved_sector_covers_the_arc_and_respects_wall_faces(self):
        crack, _, _ = locate_crack(self.config, location(108.5, -27.5), self.fields)
        inner, outer, start, end = crack.annular_sector
        z = -27.5
        angles = np.linspace(start + 0.01, end - 0.01, 21)
        inside = np.column_stack((78.5 * np.sin(angles), 78.5 * np.cos(angles), np.full(21, z)))
        np.testing.assert_allclose(signed_distance(inside, crack), -0.001, atol=1e-12)
        downstream = inside.copy()
        downstream[:, :2] *= 76.5 / 78.5
        self.assertTrue(np.all(signed_distance(downstream, crack) > 0))
        above = inside.copy()
        above[:, 2] += 0.003
        np.testing.assert_allclose(signed_distance(above, crack), 0.002, atol=1e-12)
        outside_angle = start - 0.1
        self.assertGreater(signed_distance([79 * math.sin(outside_angle), 79 * math.cos(outside_angle), z], crack), 0)
        for radius in (inner, outer):
            self.assertAlmostEqual(signed_distance(
                [radius * math.sin((start + end) / 2), radius * math.cos((start + end) / 2), z], crack,
            ), 0, places=12)

    def test_bedrock_contour_intersections_reject_invalid_sections(self):
        contours = self.config["geometry"]["bedrock"]["contours"]
        for z in (-30, 1):
            with self.subTest(z=z), self.assertRaises(ValueError):
                contour_intersections(contours, z)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            contour_intersections([{"chainage": i, "height": h} for i, h in enumerate([0, -2, 0, -2, 0])], -1)


class SelectionMapTests(unittest.TestCase):
    def test_map_reports_wall_peaks_and_bedrock_contour(self):
        points, stress, regions = synthetic_fields()
        stress[0, 0] = 5e6
        regions = regions.copy()
        regions[0] |= 2
        extra = np.array([[0.0, -500.0, -60.0]])
        result = selection_map(
            np.vstack((points, extra)), np.vstack((stress, np.zeros((1, 6)))), np.append(regions, 2), CONFIG,
        )
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["chainage_radius_m"], 78)
        self.assertGreaterEqual(result["z_min_m"], -29 - result["z_bin_m"] / 2)
        grid = result["max_principal_stress_mpa"]
        self.assertEqual((len(grid), len(grid[0])), (result["z_count"], result["y_count"]))
        self.assertAlmostEqual(max(v for row in grid for v in row if v is not None), 5.0)
        peaks = {(peak["body"], peak["kind"]): peak for peak in result["peaks"]}
        self.assertEqual(set(peaks), {("wall", "tension"), ("wall", "compression")})
        tension = peaks[("wall", "tension")]
        self.assertAlmostEqual(tension["value_mpa"], 5.0)
        self.assertAlmostEqual(tension["chainage_m"], 85)
        self.assertAlmostEqual(78 * math.acos(tension["y_m"] / 78), tension["chainage_m"])
        self.assertEqual([outline["name"] for outline in result["outlines"]], ["wall_base"])

    def test_rows_follow_the_wall_course_spacing_without_empty_rows(self):
        import copy

        for size in (0.5, 0.75, 1.0):
            config = copy.deepcopy(CONFIG)
            config["geometry"]["global_element_size_m"] = size
            levels = -size * np.arange(0, round(27 / size) + 1)
            angles = np.linspace(0.3, 2.8, 40)
            points = np.array([[78 * math.sin(a), 78 * math.cos(a), z] for z in levels for a in angles])
            with self.subTest(size=size):
                result = selection_map(points, np.zeros((len(points), 6)), np.ones(len(points), dtype=int), config)
                self.assertEqual(result["z_bin_m"], size)
                self.assertEqual(result["z_count"], len(levels))
                self.assertTrue(all(any(v is not None for v in row) for row in result["max_principal_stress_mpa"]))
                # Each course sits at the centre of its row.
                centres = result["z_min_m"] + size * (np.arange(result["z_count"]) + 0.5)
                np.testing.assert_allclose(np.sort(centres), np.sort(levels), atol=1e-9)

    def test_map_rejects_a_plan_angle_outside_the_dam(self):
        points = np.array([[-1, 0, -10], [0, 78, -10], [0, 0, -10]], dtype=float)
        with self.assertRaisesRegex(ValueError, "between 0 and pi"):
            selection_map(points, np.zeros((3, 6)), np.ones(3, dtype=int), CONFIG)
