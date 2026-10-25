"""Wall crack placement against the bedrock contour and stress-selection map."""
import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from automatic_crack import contour_intersections, locate_crack
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
        self.config["geometry"]["wall"].pop("thickness_steps", None)
        self.fields = synthetic_fields()

    def test_stepped_wall_uses_the_local_thickness_at_the_crack_elevation(self):
        self.config["geometry"]["wall"]["thickness_steps"] = [
            {"top_z_m": -10.0, "radial_width_m": 4.5}, {"top_z_m": -20.0, "radial_width_m": 5.0},
        ]
        crack, _, report = locate_crack(self.config, location(108.5, -27.5), self.fields)
        self.assertEqual(report["seed_face"], "upstream")
        self.assertEqual(crack.annular_sector[:2], (76.25, 80))
        crack, _, report = locate_crack(self.config, location(108.5, -27.5), synthetic_fields(False))
        self.assertEqual(report["seed_face"], "downstream")
        self.assertEqual(crack.annular_sector[:2], (75, 78.75))

    def test_selected_point_seeds_wall_crack_at_user_chainage(self):
        crack, options, report = locate_crack(self.config, location(108.5, -27.5), self.fields)
        self.assertIs(type(crack), CrackSpecification)
        self.assertEqual(crack.name, "user_located_crack")
        self.assertEqual(report["seed_face"], "upstream")
        self.assertEqual(crack.annular_sector[:2], (77, 80))
        self.assertAlmostEqual(report["selected_chainage_m"], 108.5)
        np.testing.assert_allclose(report["crack_chainage_m"], [106.5, 110.5])
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

    def test_downstream_tension_seeds_from_downstream_wall_face(self):
        crack, _, report = locate_crack(self.config, location(108.5, -27.5), synthetic_fields(False))
        self.assertEqual(report["seed_face"], "downstream")
        self.assertEqual(crack.annular_sector[:2], (76, 79))

    def test_crack_arc_shifts_to_stay_inside_the_wall_section(self):
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
        self.assertGreaterEqual(result["z_min_m"], -29)
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

    def test_map_rejects_a_plan_angle_outside_the_dam(self):
        points = np.array([[-1, 0, -10], [0, 78, -10], [0, 0, -10]], dtype=float)
        with self.assertRaisesRegex(ValueError, "between 0 and pi"):
            selection_map(points, np.zeros((3, 6)), np.ones(3, dtype=int), CONFIG)
