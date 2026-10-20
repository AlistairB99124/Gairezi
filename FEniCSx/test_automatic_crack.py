"""Automatic crack geometry and its connection to the phase-field distance."""
import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from automatic_crack import generate_crack, plinth_intersections
from crack_field import signed_distance


def peak_at(radius, chainage, z, region_id=1):
    theta = chainage / 78
    return {"max_principal_stress": {
        "value_pa": 2e6, "position_xyz_m": [radius * math.sin(theta), radius * math.cos(theta), z],
        "region_id": region_id, "node_id": 1,
    }}


class AutomaticCrackTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parents[1] / "config.json").read_text())
        # Nodal concrete peak of the reference job: plinth cut-back corner at Z=-27.5 m.
        self.baseline = peak_at(79.5, 97.988, -27.5)

    def position(self):
        return self.baseline["max_principal_stress"]["position_xyz_m"]

    def test_crack_at_concrete_peak_spans_wall_plinth_junction(self):
        crack, options, report = generate_crack(self.config, self.baseline)
        self.assertEqual(report["seed_face"], "upstream")
        # 75 % of the 4 m wall from the upstream face, extended to the plinth's upstream face.
        self.assertEqual(crack.annular_sector[:2], (77, 81))
        np.testing.assert_allclose(report["concrete_chainage_extent_m"], [89.85714285714286, 123.5])
        np.testing.assert_allclose(report["wall_chainage_intersections_m"], [98.44318181818181, 117.17554858934169])
        self.assertAlmostEqual(report["wall_plinth_junction_chainage_m"], 98.44318181818181)
        np.testing.assert_allclose(report["crack_chainage_m"], [95.988, 99.988])
        self.assertAlmostEqual(report["arc_length_m"], 4.0)
        self.assertEqual(report["depth_m"], 3)
        self.assertEqual(report["radial_extent_m"], 4)
        self.assertEqual(report["warnings"], [])
        self.assertEqual(crack.size_m[2], 0.002)
        self.assertEqual(crack.centre_m[2], -27.5)
        self.assertEqual(crack.length_scale_m, 0.25)
        self.assertEqual(report["damaged_bodies"], ["plinth", "wall"])
        self.assertEqual(options.maximum_refinement_levels, 5)
        self.assertEqual(options.maximum_refined_cells, 4_000_000)
        self.assertLess(signed_distance(self.position(), crack), 0)
        provenance = report["placement_provenance"]
        self.assertEqual(provenance["peak_body"], "plinth")
        self.assertTrue(provenance["regenerated_for_each_baseline"])
        json.dumps(report, allow_nan=False)
        self.assertNotIn("crack_simulations", self.config)

    def test_downstream_peak_seeds_from_downstream_face_through_plinth(self):
        self.baseline = peak_at(76.2, 97.988, -27.5)
        crack, _, report = generate_crack(self.config, self.baseline)
        self.assertEqual(report["seed_face"], "downstream")
        self.assertEqual(crack.annular_sector[:2], (75, 79))
        np.testing.assert_allclose(report["crack_chainage_m"], [95.988, 99.988])
        self.assertLess(signed_distance(self.position(), crack), 0)

    def test_arc_is_clipped_to_contain_peak_and_junction(self):
        self.baseline = peak_at(79, 90.5, -25)
        crack, _, report = generate_crack(self.config, self.baseline)
        junction = report["wall_plinth_junction_chainage_m"]
        self.assertAlmostEqual(junction, 87.96091205211727)
        start, end = report["crack_chainage_m"]
        self.assertLess(start, junction)
        self.assertLess(junction, end)
        self.assertLessEqual(start, 90.5)
        self.assertLessEqual(90.5, end)
        self.assertAlmostEqual(end - start, 4.0)
        self.assertLess(signed_distance(self.position(), crack), 0)

    def test_peak_far_from_junction_fails_instead_of_moving_crack(self):
        self.baseline = peak_at(79, 105, -25, region_id=2)
        with self.assertRaisesRegex(ValueError, "cannot contain both the peak"):
            generate_crack(self.config, self.baseline)

    def test_curved_sector_covers_whole_arc_not_a_straight_box(self):
        crack, _, _ = generate_crack(self.config, self.baseline)
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

    def test_uses_peak_z_not_a_default_elevation(self):
        self.baseline = peak_at(79, 72.5, -20)
        crack, _, _ = generate_crack(self.config, self.baseline)
        self.assertEqual(crack.centre_m[2], -20)

    def test_peak_below_wall_fails(self):
        self.baseline = peak_at(79, 105, -28.5)
        with self.assertRaisesRegex(ValueError, "wall does not reach"):
            generate_crack(self.config, self.baseline)

    def test_legacy_centroid_peak_key_is_still_accepted(self):
        peak = self.baseline["max_principal_stress"]
        peak["centroid_xyz_m"] = peak.pop("position_xyz_m")
        crack, _, _ = generate_crack(self.config, self.baseline)
        self.assertEqual(crack.centre_m[2], -27.5)
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
                self.baseline["max_principal_stress"]["position_xyz_m"] = position
                generate_crack(self.config, self.baseline)


if __name__ == "__main__":
    unittest.main()
