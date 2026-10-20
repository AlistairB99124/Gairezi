"""User-located crack geometry and its connection to the phase-field distance."""
import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from automatic_crack import locate_crack, plinth_intersections
from crack_field import signed_distance
from crack_selection import selection_map

CONFIG = json.loads((Path(__file__).parents[1] / "config.json").read_text())


def synthetic_fields(upstream_tension=True):
    """Wall (r 76-80) above and plinth (r 75-81) below the plinth top contour; tension on one face."""
    contours = CONFIG["geometry"]["plinth"]["contours"]
    chainages, heights = [row["chainage"] for row in contours], [row["height"] for row in contours]
    s, z, r = np.meshgrid(np.arange(85, 125.01, 0.5), np.arange(-29, -19.95, 0.1), np.arange(75, 81.01, 1.0),
                          indexing="ij")
    s, z, r = s.ravel(), z.ravel(), r.ravel()
    top = np.interp(s, chainages, heights)
    regions = np.where((z <= top), 1, np.where((r >= 76) & (r <= 80), 2, 0))
    keep = regions > 0
    s, z, r, regions = s[keep], z[keep], r[keep], regions[keep]
    theta = s / 78
    points = np.column_stack((r * np.sin(theta), r * np.cos(theta), z))
    stress = np.zeros((len(points), 6))
    stress[:, 0] = np.where((r >= 78) == upstream_tension, 2e6, 0.5e6)
    stress[:, 1] = -3e6
    return points, stress, regions


def location(chainage, z):
    return (78 * math.cos(chainage / 78), z)


class UserLocatedCrackTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(CONFIG)
        self.fields = synthetic_fields()

    def test_selected_point_sets_z_and_chainage_and_crosses_wall_and_plinth(self):
        crack, options, report = locate_crack(self.config, location(97.988, -27.5), self.fields)
        self.assertEqual(crack.name, "user_located_crack")
        self.assertEqual(report["seed_face"], "upstream")
        # 75 % of the 4 m wall from the upstream face, extended to the plinth's upstream face.
        self.assertEqual(crack.annular_sector[:2], (77, 81))
        self.assertAlmostEqual(report["selected_chainage_m"], 97.988)
        np.testing.assert_allclose(report["crack_chainage_m"], [95.988, 99.988])
        self.assertAlmostEqual(report["arc_length_m"], 4.0)
        self.assertEqual(report["required_region_ids"], [1, 2])
        self.assertEqual(report["damaged_bodies"], ["plinth", "wall"])
        self.assertEqual(report["depth_m"], 3)
        self.assertEqual(report["radial_extent_m"], 4)
        self.assertEqual(report["warnings"], [])
        self.assertEqual(crack.centre_m[2], -27.5)
        self.assertEqual(crack.size_m[2], 0.002)
        self.assertEqual(crack.length_scale_m, 0.25)
        self.assertEqual(options.maximum_refined_cells, 4_000_000)
        theta = 97.988 / 78
        self.assertLess(signed_distance([79 * math.sin(theta), 79 * math.cos(theta), -27.5], crack), 0)
        self.assertIn("user selection", report["placement_provenance"]["elevation_source"])
        json.dumps(report, allow_nan=False)
        self.assertNotIn("crack_simulations", self.config)

    def test_downstream_tension_seeds_from_downstream_face_through_plinth(self):
        crack, _, report = locate_crack(self.config, location(97.988, -27.5), synthetic_fields(False))
        self.assertEqual(report["seed_face"], "downstream")
        self.assertEqual(crack.annular_sector[:2], (75, 79))

    def test_wall_only_location_does_not_require_plinth(self):
        crack, _, report = locate_crack(self.config, location(110, -20), self.fields)
        self.assertEqual(crack.annular_sector[:2], (77, 80))
        self.assertEqual(report["required_region_ids"], [2])
        self.assertEqual(crack.centre_m[2], -20)

    def test_arc_is_shifted_inside_narrow_concrete_section(self):
        _, _, report = locate_crack(self.config, location(97, -28.5), self.fields)
        np.testing.assert_allclose(report["concrete_chainage_extent_m"], [96.5, 118.5])
        np.testing.assert_allclose(report["crack_chainage_m"], [96.5, 100.5])
        self.assertIn("shifted", report["warnings"][0])

    def test_invalid_or_outside_selection_fails_instead_of_moving(self):
        for selected, message in (
            ((float("nan"), -25), "finite"),
            ((79, -25), "outside the"),
            (location(110, -30), "No concrete section"),
            (location(140, -25), "outside the concrete section"),
            ((1, 2, 3), "two numbers"),
        ):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, message):
                locate_crack(self.config, selected, self.fields)

    def test_location_without_baseline_nodes_fails(self):
        points, stress, regions = self.fields
        far = points[:, 2] < -25
        with self.assertRaisesRegex(ValueError, "No wall or plinth baseline nodes"):
            locate_crack(self.config, location(110, -20), (points[far], stress[far], regions[far]))

    def test_curved_sector_covers_whole_arc_not_a_straight_box(self):
        crack, _, _ = locate_crack(self.config, location(97.988, -27.5), self.fields)
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

    def test_invalid_sections_fail_instead_of_clamping(self):
        contours = self.config["geometry"]["plinth"]["contours"]
        for z in (-30, 1):
            with self.subTest(z=z), self.assertRaises(ValueError):
                plinth_intersections(contours, z)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            plinth_intersections([
                {"chainage": i, "height": h} for i, h in enumerate([0, -2, 0, -2, 0])
            ], -1)


class SelectionMapTests(unittest.TestCase):
    def test_map_bins_concrete_stress_marks_body_peaks_and_round_trips_chainage(self):
        points, stress, regions = synthetic_fields()
        stress[0, 0] = 5e6
        regions = regions.copy()
        regions[regions == 1] |= 4  # Plinth nodes shared with the rock seat stay plinth nodes.
        extra = np.array([[0.0, -500.0, -60.0]])  # Bedrock-only nodes are excluded.
        result = selection_map(
            np.vstack((points, extra)), np.vstack((stress, np.zeros((1, 6)))), np.append(regions, 4), CONFIG,
        )
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["chainage_radius_m"], 78)
        self.assertGreaterEqual(result["z_min_m"], -29)
        grid = result["max_principal_stress_mpa"]
        self.assertEqual((len(grid), len(grid[0])), (result["z_count"], result["y_count"]))
        self.assertAlmostEqual(max(v for row in grid for v in row if v is not None), 5.0)
        peaks = {(peak["body"], peak["kind"]): peak for peak in result["peaks"]}
        self.assertEqual(set(peaks), {("plinth", "tension"), ("plinth", "compression"),
                                      ("wall", "tension"), ("wall", "compression")})
        tension = peaks[("plinth", "tension")]
        self.assertAlmostEqual(tension["value_mpa"], 5.0)
        self.assertAlmostEqual(tension["chainage_m"], 85)
        self.assertAlmostEqual(78 * math.acos(tension["y_m"] / 78), tension["chainage_m"])
        self.assertAlmostEqual(peaks[("wall", "compression")]["value_mpa"], -3.0)
        self.assertEqual([outline["name"] for outline in result["outlines"]], ["plinth_top", "rock_seat"])
        self.assertFalse(result["engineering_validated"])

    def test_map_rejects_missing_concrete(self):
        with self.assertRaisesRegex(ValueError, "no wall or plinth"):
            selection_map(np.zeros((1, 3)) + [1, 1, 0], np.zeros((1, 6)), np.array([4]), CONFIG)


if __name__ == "__main__":
    unittest.main()
