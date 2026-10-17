"""Automatic crack geometry and its connection to the phase-field distance."""
import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from automatic_crack import generate_crack, plinth_intersections
from crack_field import signed_distance


class AutomaticCrackTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parents[1] / "config.json").read_text())
        self.baseline = {"max_principal_stress": {
            "value_pa": 2e6, "centroid_xyz_m": [79, 0, -25], "region_id": 2, "element_id": 1,
        }}

    def test_exact_plinth_intersections_and_75_percent_depth(self):
        crack, options, report = generate_crack(self.config, self.baseline)
        np.testing.assert_allclose(report["chainage_intersections_m"], [87.96091205211727, 123.05329153605015])
        self.assertEqual(crack.annular_sector[:2], (77, 80))
        self.assertEqual(crack.size_m[2], 0.002)
        self.assertEqual(crack.centre_m[2], -25)
        self.assertEqual(options.maximum_refinement_levels, 5)
        np.testing.assert_allclose(
            [point[1] for point in report["upstream_intersections_xyz_m"]],
            [34.29881690368324, -0.5447937850532466],
        )
        self.assertNotIn("crack_simulations", self.config)

    def test_curved_sector_covers_whole_arc_not_a_straight_box(self):
        crack, _, _ = generate_crack(self.config, self.baseline)
        inner, outer, start, end = crack.annular_sector
        angles = np.linspace(start + 0.01, end - 0.01, 21)
        inside = np.column_stack((78.5 * np.sin(angles), 78.5 * np.cos(angles), np.full(21, -25)))
        np.testing.assert_allclose(signed_distance(inside, crack), -0.001, atol=1e-12)
        downstream = inside.copy()
        downstream[:, :2] *= 76.5 / 78.5
        self.assertTrue(np.all(signed_distance(downstream, crack) > 0))
        above = inside.copy()
        above[:, 2] += 0.003
        np.testing.assert_allclose(signed_distance(above, crack), 0.002, atol=1e-12)
        outside_angle = start - 0.1
        self.assertGreater(signed_distance([79 * math.sin(outside_angle), 79 * math.cos(outside_angle), -25], crack), 0)
        for radius in (inner, outer):
            self.assertAlmostEqual(signed_distance(
                [radius * math.sin((start + end) / 2), radius * math.cos((start + end) / 2), -25], crack,
            ), 0, places=12)

    def test_uses_peak_z_not_a_default_elevation(self):
        self.baseline["max_principal_stress"]["centroid_xyz_m"][2] = -20
        crack, _, _ = generate_crack(self.config, self.baseline)
        self.assertEqual(crack.centre_m[2], -20)

    def test_peak_shift_changes_prescription_and_warns_without_moving_crack(self):
        original = copy.deepcopy(self.config)
        self.baseline["max_principal_stress"]["centroid_xyz_m"] = [78.03, 17.04, -27.75]
        old, _, old_report = generate_crack(self.config, self.baseline)
        self.baseline["max_principal_stress"]["centroid_xyz_m"][2] = -27.875
        new, _, new_report = generate_crack(self.config, self.baseline)
        self.assertEqual(old.centre_m[2], -27.75)
        self.assertEqual(new.centre_m[2], -27.875)
        self.assertEqual(old.annular_sector[:2], new.annular_sector[:2])
        self.assertLess(new.size_m[1], old.size_m[1])
        np.testing.assert_allclose(
            [old.size_m[1], new.size_m[1]], [16.974299945743915, 16.03522689593621],
            atol=1e-9,
        )
        for report, clearance in ((old_report, 0.25), (new_report, 0.125)):
            placement = report["placement_provenance"]
            self.assertTrue(placement["regenerated_for_each_baseline"])
            self.assertAlmostEqual(placement["plinth_elevation_at_peak_m"], -28)
            self.assertAlmostEqual(placement["clearance_above_plinth_at_peak_m"], clearance)
            self.assertAlmostEqual(placement["clearance_in_length_scales"], clearance / 0.1)
            json.dumps(report, allow_nan=False)
        self.assertEqual(old_report["warnings"], [])
        self.assertEqual(len(new_report["warnings"]), 1)
        self.assertIn("interface where damage is fixed at zero", new_report["warnings"][0])
        self.assertEqual(self.config, original)

    def test_invalid_sections_fail_instead_of_clamping(self):
        contours = self.config["geometry"]["plinth"]["contours"]
        for z in (-30, 1, -28):
            with self.subTest(z=z), self.assertRaises(ValueError):
                plinth_intersections(contours, z)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            plinth_intersections([
                {"chainage": i, "height": h} for i, h in enumerate([0, -2, 0, -2, 0])
            ], -1)

    def test_compression_only_fails_explicitly(self):
        self.baseline["max_principal_stress"]["value_pa"] = -1
        with self.assertRaisesRegex(ValueError, "no positive tensile"):
            generate_crack(self.config, self.baseline)

    def test_placement_provenance_rejects_invalid_peak_coordinates_without_clamping(self):
        for position in ([float("nan"), 17, -25], [79, float("inf"), -25], [79, 0], [-79, 0, -25]):
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, "peak position|peak chainage"):
                self.baseline["max_principal_stress"]["centroid_xyz_m"] = position
                generate_crack(self.config, self.baseline)


if __name__ == "__main__":
    unittest.main()
