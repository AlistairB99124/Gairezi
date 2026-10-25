"""Solver-free checks of the native Aster cracking submodel geometry and transfer."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from Aster.crack_submodel import (
    block_layout, boundary_sources, crack_metrics, moving_least_squares, normal_edges,
    thickness_segment, uniform_edges,
)

RADIUS = 78.0
CONFIG = {"geometry": {"wall": {"centerline_radius_m": RADIUS, "upstream_radius_m": 80.0,
                                "radial_width_m": 4.0, "thickness_steps": []}}}


def seed(elevation=-15.0, chainage=(100.0, 104.0), radial=(76.0, 79.0)):
    return {"geometry": {"annular_sector": [radial[0], radial[1], chainage[0] / RADIUS, chainage[1] / RADIUS],
                         "elevation_m": elevation, "seed_thickness_m": 0.002}}


def wall_nodes(chainage=(90.0, 115.0), z=(-25.0, 0.0), abutment=None):
    arc, radius, elevation = np.meshgrid(np.arange(*chainage, 0.5), np.linspace(76.0, 80.0, 5),
                                         np.arange(z[0], z[1] + 1e-9, 0.5), indexing="ij")
    theta = arc.ravel() / RADIUS
    points = np.column_stack((radius.ravel() * np.sin(theta), radius.ravel() * np.cos(theta), elevation.ravel()))
    regions = np.ones(len(points), dtype=int)
    if abutment is not None:
        regions[arc.ravel() > abutment] = 2
    return points, regions


def layout_for(shared, points, regions, **options):
    with patch("FEniCSx.crack_contract.validate_selected_seed", side_effect=lambda value: value):
        return block_layout(CONFIG, shared, points, regions, **options)


class EdgeTests(unittest.TestCase):
    def test_uniform_edges_keep_required_values_and_spacing(self):
        edges = uniform_edges(0.0, 5.0, 0.5, (1.23, 3.9))
        self.assertEqual((edges[0], edges[-1]), (0.0, 5.0))
        for value in (1.23, 3.9):
            self.assertTrue(any(math.isclose(edge, value) for edge in edges))
        self.assertTrue(np.all(np.diff(edges) > 0) and np.max(np.diff(edges)) <= 0.5 + 1e-12)

    def test_normal_edges_have_one_exact_seed_layer_and_bounded_growth(self):
        edges = np.asarray(normal_edges(-17.0, -15.0, -13.0, 0.001, 0.5))
        self.assertEqual((edges[0], edges[-1]), (-17.0, -13.0))
        steps = np.diff(edges)
        self.assertTrue(np.all(steps > 0))
        self.assertEqual(int(np.count_nonzero(np.isclose(steps, 0.002))), 1)
        # The final graded layer may absorb a short remainder.
        self.assertLessEqual(steps.max(), 1.5 * 0.5)
        with self.assertRaises(ValueError):
            normal_edges(-15.0005, -15.0, -13.0, 0.001, 0.5)

    def test_thickness_segment_uses_step_bounds_and_rejects_a_step(self):
        wall = {"upstream_radius_m": 80.0, "radial_width_m": 6.0,
                "thickness_steps": [{"top_z_m": -10.0, "radial_width_m": 4.0},
                                    {"top_z_m": -20.0, "radial_width_m": 5.0}]}
        middle = thickness_segment(wall, -15.0)
        self.assertEqual((middle["thickness_m"], middle["upper_z_m"], middle["lower_z_m"]), (4.0, -10.0, -20.0))
        self.assertFalse(middle["upper_is_crest"])
        top = thickness_segment(wall, -5.0)
        self.assertEqual((top["thickness_m"], top["upper_z_m"], top["upper_is_crest"]), (6.0, 0.0, True))
        with self.assertRaisesRegex(ValueError, "thickness step"):
            thickness_segment(wall, -10.0)


class LayoutTests(unittest.TestCase):
    def test_block_keeps_seed_bounds_and_margins_in_open_wall(self):
        points, regions = wall_nodes()
        layout = layout_for(seed(), points, regions)
        self.assertEqual(layout["bounds"]["z_m"], [-17.0, -13.0])
        self.assertEqual(layout["bounds"]["radius_m"], [76.0, 80.0])
        np.testing.assert_allclose(layout["bounds"]["chainage_m"], [98.0, 106.0])
        arc = np.asarray(layout["theta_edges_rad"]) * RADIUS
        for value in (100.0, 104.0):
            self.assertTrue(np.any(np.isclose(arc, value)))
        self.assertTrue(np.any(np.isclose(layout["radius_edges_m"], 79.0)))
        self.assertFalse(layout["top_is_crest"])

    def test_block_reaches_crest_and_clips_at_abutment(self):
        points, regions = wall_nodes(abutment=105.0)
        layout = layout_for(seed(elevation=-1.0), points, regions)
        self.assertTrue(layout["top_is_crest"])
        self.assertEqual(layout["bounds"]["z_m"][1], 0.0)
        first_abutment = RADIUS * np.arctan2(points[regions == 2, 0], points[regions == 2, 1]).min()
        self.assertLessEqual(layout["bounds"]["chainage_m"][1], first_abutment - 0.25 + 1e-9)
        self.assertGreaterEqual(layout["margins_m"]["end_m"], 0.5)

    def test_seed_against_abutment_or_bedrock_is_rejected(self):
        points, regions = wall_nodes(abutment=104.2)
        with self.assertRaisesRegex(ValueError, "at least 0.5 m"):
            layout_for(seed(), points, regions)
        points, regions = wall_nodes()
        regions[points[:, 2] <= -15.5] = 4
        with self.assertRaisesRegex(ValueError, "at least 0.5 m"):
            layout_for(seed(), points, regions)

    def test_boundary_sources_are_wall_only(self):
        points, regions = wall_nodes(abutment=107.0)
        layout = layout_for(seed(), points, regions)
        sources = boundary_sources(layout, points, regions, np.zeros_like(points))
        chosen = np.asarray(sources["coordinates_m"])
        arc = RADIUS * np.arctan2(chosen[:, 0], chosen[:, 1])
        self.assertTrue(np.all(arc <= 107.0 + 1e-9))
        self.assertGreater(len(chosen), 20)


class TransferAndMetricTests(unittest.TestCase):
    def test_moving_least_squares_reproduces_linear_displacement(self):
        generator = np.random.default_rng(3)
        sources = generator.uniform(-2, 2, (200, 3))
        matrix = np.array([[1e-3, 2e-4, -3e-4], [0.0, 5e-4, 1e-4], [-2e-4, 0.0, 7e-4]])
        offset = np.array([1e-3, -2e-3, 5e-4])
        targets = generator.uniform(-1, 1, (25, 3))
        values, nearest = moving_least_squares(sources, sources @ matrix.T + offset, targets)
        np.testing.assert_allclose(values, targets @ matrix.T + offset, atol=1e-12)
        self.assertTrue(np.all(nearest < 1.0))

    def test_metrics_count_only_growth_beyond_the_seed(self):
        layout = {"chainage_radius_m": RADIUS, "segment": {"thickness_m": 4.0}, "element_size_m": 0.5,
                  "top_is_crest": False,
                  "bounds": {"chainage_m": [98.0, 106.0], "radius_m": [76.0, 80.0], "z_m": [-17.0, -13.0]}}
        theta = 102.0 / RADIUS
        xyz = [[r * math.sin(theta), r * math.cos(theta), -15.0] for r in (76.5, 77.5, 78.5, 79.5)]
        data = {"audit": {"coordinates_m": xyz, "damage_values": [0.999, 0.999, 0.0, 0.0]},
                "history": [{"time": 1.0, "damage_values": [0.999, 0.999, 0.2, 0.0],
                             "maximum_principal_stress_pa": 0.0},
                            {"time": 2.0, "damage_values": [0.999, 0.999, 0.97, 0.6],
                             "maximum_principal_stress_pa": 1.5e6}]}
        metrics = crack_metrics(data, layout, seed()["geometry"])
        final = metrics["final"]
        self.assertEqual((final["load_factor"], final["new_points_above_0_95"], final["new_points_above_0_5"]),
                         (1.0, 1, 2))
        self.assertAlmostEqual(final["cracked_radial_extent_m"], 2.0)
        self.assertAlmostEqual(final["cracked_through_thickness_fraction"], 0.5)
        self.assertEqual(final["new_damage_near_cut_faces_max"], 0.0)
        self.assertEqual(metrics["history"][0]["new_points_above_0_95"], 0)


if __name__ == "__main__":
    unittest.main()
