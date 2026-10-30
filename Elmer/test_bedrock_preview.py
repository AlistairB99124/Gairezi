"""Checks for the standalone, explicitly unsolved three-body export."""
import math
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import meshio
import numpy as np
from matplotlib.path import Path as PlotPath

from Elmer.bedrock_preview import (
    attributed_mesh, bedrock_geometry, configured_bedrock_geometry, constrain_edges,
    export_preview, graded_axis,
    layered_bedrock_geometry, parallel_rock_surface, refine_rock_surface,
    seated_bedrock_geometry, survey_coordinates,
)
from Elmer.model_config import load_config
from Elmer.regions.tetrahedral import _CELL_FACES


ROOT = Path(__file__).resolve().parents[1]


class BedrockPreviewTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")
        self.concrete = np.array([[0., 0., -29.], [80., 80., 0.]])

    def test_projection_and_extrusion_preserve_approved_section(self):
        points, cells, report = bedrock_geometry(
            self.config["geometry"]["bedrock"]["contours"], self.concrete,
        )
        self.assertAlmostEqual(report["projected_span_m"], 160 * math.sin(180 / 160))
        np.testing.assert_array_equal(points[::4, 2], -31.)
        np.testing.assert_array_equal(
            points[1::4, 2],
            [row["height"] for row in self.config["geometry"]["bedrock"]["contours"]],
        )
        np.testing.assert_array_equal(points[1::4, 2], points[3::4, 2])
        np.testing.assert_allclose(
            points[2::4, :2] - points[::4, :2],
            np.tile(np.array(report["depth_unit_vector_xy"]) * report["extrusion_depth_m"],
                    (len(points) // 4, 1)),
        )
        for cell in cells:
            p = points[cell]
            self.assertGreater(np.linalg.det(np.column_stack(
                (p[1]-p[0], p[3]-p[0], p[4]-p[0]),
            )), 0)

    def test_bedrock_section_includes_five_metre_crest_extensions(self):
        contours = self.config["geometry"]["bedrock"]["contours"]
        points, _, _ = bedrock_geometry(contours, self.concrete)
        self.assertEqual(contours[0]["chainage"], 0)
        self.assertEqual(contours[-1]["chainage"], 180)
        self.assertEqual(points[1, 2], 5)
        self.assertEqual(points[-3, 2], 5)

    def test_rejects_invalid_geometry(self):
        for rows in (
            [{"chainage": 6, "height": 0}],
            [{"chainage": 6, "height": 0}, {"chainage": 6, "height": -2}],
            [{"chainage": 6, "height": 0}, {"chainage": 174, "height": -31}],
        ):
            with self.assertRaises(ValueError):
                bedrock_geometry(rows, self.concrete)

    def test_manifest_has_readable_attributed_wall_and_bedrock(self):
        self.config["geometry"]["global_element_size_m"] = .25
        structure = SimpleNamespace(
            nodes=[(0., 0., -29.), (1., 0., -29.), (0., 1., -29.), (0., 0., 0.)],
            cells=[(4, (0, 1, 2, 3))],
            region_ids=[1],
            boundaries=[],
            suggested_global_element_size_m=.25,
            element_size_target_m=.75,
            target_wall_tetrahedral_elements=150000,
            wall_tetrahedral_element_count=1,
        )
        rock = np.array([
            [0., 0., -31.], [1., 0., -31.], [0., 1., -31.],
            [0., 0., -29.], [1., 0., -29.], [0., 1., -29.],
        ])
        graded_rock = SimpleNamespace(
            points=rock,
            blocks={6: np.tile(np.array([[0, 1, 2, 3, 4, 5]]), (8, 1))},
            report={},
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "Elmer.bedrock_preview.load_config", return_value=self.config,
        ), patch(
            "Elmer.bedrock_preview.build_combined_structure", return_value=structure,
        ), patch(
            "Elmer.bedrock_preview.configured_rock_mesh", return_value=graded_rock,
        ) as rock_mesh:
            path = export_preview(ROOT, Path(directory))
            effective_config = rock_mesh.call_args.args[0]
            self.assertEqual(effective_config["geometry"]["global_element_size_m"], .75)
            self.assertEqual(effective_config["geometry"]["bedrock"]["mesh"]["near_size_m"], .75)
            self.assertEqual(effective_config["geometry"]["bedrock"]["mesh"]["far_size_m"], 15.)
            sizing = json.loads((path.parent / "geometry_review.json").read_text())["mesh_sizing"]
            self.assertEqual(sizing["suggested_global_element_size_m"], .25)
            self.assertEqual(sizing["effective_wall_element_size_m"], .75)
            self.assertEqual(sizing["configured_bedrock_near_size_m"], .5)
            self.assertEqual(sizing["effective_bedrock_near_size_m"], .75)
            pieces = ET.parse(path).findall(".//Piece")
            self.assertEqual(len(pieces), 2)
            ids = set()
            for piece in pieces:
                mesh = meshio.read(path.parent / piece.attrib["Source"])
                ids.update(np.concatenate(mesh.cell_data["RegionId"]).tolist())
                material = self.config["material"][
                    "bedrock" if piece.attrib["Source"] == "bedrock.vtu" else "concrete"
                ]
                for values in mesh.cell_data["YoungsModulus"]:
                    np.testing.assert_array_equal(values, material["youngs_modulus"])
            self.assertEqual(ids, {1, 2})
            rock_mesh = meshio.read(path.parent / "bedrock.vtu")
            self.assertEqual(sum(len(block.data) for block in rock_mesh.cells), 8)

    def test_domain_defaults_overrides_and_legacy_fallback(self):
        with patch("Elmer.bedrock_preview.seated_bedrock_geometry") as generate:
            configured_bedrock_geometry(self.config, self.concrete, [])
            self.assertEqual(generate.call_args.args[3:5], (58., 58.))
            self.assertEqual(generate.call_args.kwargs["bottom_z_m"], -87.)
            configured_bedrock_geometry(self.config, self.concrete, [], 0., 10., -100.)
            self.assertEqual(generate.call_args.args[3:5], (0., 10.))
            self.assertEqual(generate.call_args.kwargs["bottom_z_m"], -100.)
            del self.config["geometry"]["bedrock"]["domain"]
            configured_bedrock_geometry(self.config, self.concrete, [])
            self.assertEqual(generate.call_args.args[3:5], (20., 5.))
            self.assertEqual(generate.call_args.kwargs["bottom_z_m"], -31.)
            self.assertFalse(generate.call_args.kwargs["grade_mesh"])

    def test_grading_is_fine_near_dam_and_coarser_in_far_field(self):
        coordinates = graded_axis(-87., 5., -34., 5., .5)
        self.assertEqual(coordinates[0], -87.)
        self.assertEqual(coordinates[-1], 5.)
        steps = np.diff(coordinates)
        self.assertTrue(np.all(steps > 0))
        self.assertLessEqual(steps.max(), 5. + 1e-10)
        self.assertTrue(np.allclose(steps[coordinates[:-1] >= -34.], .5))
        self.assertGreater(steps[coordinates[1:] < -34.].max(), .5)
        self.assertLess(len(coordinates), 1 + 92 / .5)
        for spacing in (0., -1., float("nan"), float("inf")):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                graded_axis(-87., 5., -34., 5., spacing)

    def test_graded_layers_preserve_volume_with_configured_deep_bottom(self):
        top = np.array([[0., 0., -29.], [1., 0., -29.], [0., 1., -29.]])
        bottom = top.copy()
        bottom[:, 2] = -87.
        points, blocks, layers = layered_bedrock_geometry(
            np.vstack((bottom, top)), np.array([[0, 1, 2, 3, 4, 5]]), .5,
            fine_layer_bottom_z_m=-34.,
        )
        self.assertEqual(set(blocks), {6})
        self.assertLess(layers, 116)
        cells = blocks[6]
        steps = points[cells[:, 3], 2] - points[cells[:, 0], 2]
        self.assertTrue(np.all(steps > 0))
        self.assertLessEqual(steps.max(), 5. + 1e-10)
        self.assertGreater(steps.max(), .5)
        self.assertTrue(np.allclose(steps[points[cells[:, 0], 2] >= -34.], .5))
        self.assertAlmostEqual(.5 * steps.sum(), .5 * 58.)
        np.testing.assert_allclose([points[:, 2].min(), points[:, 2].max()], [-87., -29.])

    def test_survey_reference_changes_angles_not_radii_or_elevations(self):
        old = np.array([[78*math.sin(100/78), 78*math.cos(100/78), -20.]])
        new = survey_coordinates(old, 78., 80.)
        self.assertAlmostEqual(math.atan2(new[0, 0], new[0, 1]), 100/80)
        self.assertAlmostEqual(np.linalg.norm(new[0, :2]), 78.)
        self.assertEqual(new[0, 2], -20.)

    def test_inserts_seat_edge_without_losing_area(self):
        points = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        result = constrain_edges(points, triangles, {(1, 3)})
        self.assertTrue(all({1, 3} <= set(tri) for tri in result))

    def test_vtk_wedges_use_the_export_coordinate_convention(self):
        points = np.array([
            [0., 0., -31.], [1., 0., -31.], [0., 1., -31.],
            [0., 0., -29.], [1., 0., -29.], [0., 1., -29.],
        ])
        mesh = attributed_mesh(
            points, [("wedge", np.array([[0, 1, 2, 3, 4, 5]]))],
            6, 2, self.config["material"]["bedrock"],
        )
        np.testing.assert_array_equal(mesh.cells[0].data, [[0, 2, 1, 3, 5, 4]])

    def test_surface_refinement_preserves_terrain_area_and_has_no_hanging_edges(self):
        points = np.array([[0., 0., -30.], [2., 0., -30.],
                           [2., 2., -29.], [0., 2., -29.]])
        points, triangles = refine_rock_surface(
            points, np.array([[0, 1, 2], [0, 2, 3]]), set(), .5,
        )
        edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        self.assertLessEqual(
            np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1).max(),
            math.sqrt(2) * .5 * (1 + 1e-10),
        )
        np.testing.assert_allclose(points[:, 2], -30 + points[:, 1] / 2)
        areas = np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        )[:, 2] / 2
        self.assertTrue(np.all(areas > 0))
        self.assertAlmostEqual(areas.sum(), 4.)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(counts <= 2))
        boundary = points[unique[counts == 1], :2]
        self.assertTrue(np.all(np.any(
            np.all(np.isclose(boundary, 0), axis=1)
            | np.all(np.isclose(boundary, 2), axis=1), axis=1,
        )))

    def test_parallel_rows_preserve_seat_and_terrain_without_fans(self):
        points = np.array([
            [0., 0., -30.], [4., 0., -30.], [4., 4., -28.], [0., 4., -28.],
            [1.5, 1.5, -29.25], [2.5, 1.5, -29.25],
            [2.5, 2.5, -28.75], [1.5, 2.5, -28.75],
        ])
        exterior = np.array([
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ])
        seat_edges = {(4, 5), (5, 6), (6, 7), (4, 7)}
        surface, triangles = parallel_rock_surface(
            points, exterior, seat_edges, PlotPath(points[4:, :2]), .5,
        )
        np.testing.assert_array_equal(surface[:len(points)], points)
        np.testing.assert_allclose(surface[:, 2], -30 + surface[:, 1] / 2)
        edges = {
            tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
            for tri in triangles for i in range(3)
        }
        self.assertTrue(seat_edges <= edges)
        ids = {tuple(point[:2]): index for index, point in enumerate(surface)}
        for y in (0.5, 1., 3., 3.5):
            for x in np.arange(0., 4., .5):
                self.assertIn(tuple(sorted((ids[(x, y)], ids[(x + .5, y)]))), edges)
        areas = np.cross(
            surface[triangles[:, 1]] - surface[triangles[:, 0]],
            surface[triangles[:, 2]] - surface[triangles[:, 0]],
        )[:, 2] / 2
        self.assertTrue(np.all(areas > 0))
        self.assertAlmostEqual(areas.sum(), 15.)

    def test_layered_rock_has_half_metre_steps_and_positive_conforming_closures(self):
        top = np.array([[0., 0., -30.75], [1., 0., -29.75], [0., 1., -29.25]])
        bottom = top.copy()
        bottom[:, 2] = -31.
        points, blocks, layers = layered_bedrock_geometry(
            np.vstack((bottom, top)), np.array([[0, 1, 2, 3, 4, 5]]), .5,
        )
        self.assertEqual(layers, 4)
        self.assertEqual(set(blocks), {4, 6, 7})
        np.testing.assert_array_equal(points[:3], top)
        np.testing.assert_array_equal(points[3:6], bottom)
        self.assertEqual(len(np.unique(points, axis=0)), len(points))
        faces = Counter()
        volume = 0.
        tetrahedra = {
            4: [(0, 1, 2, 3)],
            6: [(0, 1, 2, 3), (1, 4, 2, 3), (2, 4, 5, 3)],
            7: [(0, 1, 2, 4), (0, 2, 3, 4)],
        }
        patterns = {
            **_CELL_FACES,
            4: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
            7: ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
        }
        for kind, block in blocks.items():
            self.assertTrue(all(len(set(cell)) == len(cell) for cell in block))
            for a in range(block.shape[1]):
                for b in range(a + 1, block.shape[1]):
                    same_column = np.all(points[block[:, a], :2] == points[block[:, b], :2], axis=1)
                    steps = np.abs(points[block[same_column, a], 2] - points[block[same_column, b], 2])
                    if len(steps):
                        self.assertLessEqual(steps.max(), .5 + 1e-10)
            for cell in block:
                for pattern in patterns[kind]:
                    faces[tuple(sorted(cell[list(pattern)]))] += 1
            for pattern in tetrahedra[kind]:
                p = points[block[:, pattern]]
                determinants = np.linalg.det(p[:, 1:] - p[:, :1])
                self.assertTrue(np.all(determinants > 0))
                volume += determinants.sum() / 6
        self.assertAlmostEqual(volume, .5 * np.mean(top[:, 2] + 31))
        self.assertTrue(all(count <= 2 for count in faces.values()))
        self.assertEqual(faces[(0, 1, 2)], 1)
        self.assertEqual(faces[(3, 4, 5)], 1)

    def test_refinement_rejects_nonpositive_or_nonfinite_spacing(self):
        points = np.array([[0., 0., -30.], [1., 0., -30.], [0., 1., -30.]])
        for spacing in (0., -1., float("nan"), float("inf")):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                refine_rock_surface(points, np.array([[0, 1, 2]]), set(), spacing)
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                layered_bedrock_geometry(
                    np.vstack((points - [0., 0., 1.], points)),
                    np.array([[0, 1, 2, 3, 4, 5]]), spacing,
                )

    def test_rock_seat_matches_wall_and_retains_full_canyon_span(self):
        contours = [{"chainage": 6, "height": 0.}, {"chainage": 174, "height": 0.}]
        _, _, frame = bedrock_geometry(contours, self.concrete)
        origin = np.array(frame["section_origin_xy_m"])
        across = np.array(frame["across_unit_vector_xy"])
        depth = np.array(frame["depth_unit_vector_xy"])
        local = np.array([[40., 10.], [80., 10.], [80., 20.], [40., 20.],
                          [0., 0.], [frame["projected_span_m"], 40.]])
        plan = origin + local[:, :1] * across + local[:, 1:] * depth
        concrete = np.column_stack((plan, [-20., -21., -22., -23., 0., 0.]))
        points, cells, report = seated_bedrock_geometry(
            contours, concrete, [(0, 1, 2, 3)], element_size_m=50.,
        )
        self.assertLess(report["seat_max_coordinate_difference_m"], 1e-10)
        self.assertEqual(report["seat_triangles"], 2)
        self.assertAlmostEqual(report["rock_across_limits_m"][0], 0.)
        self.assertAlmostEqual(report["rock_across_limits_m"][1], frame["projected_span_m"])
        self.assertTrue(np.all(points[:len(points)//2, 2] == -31.))
        self.assertEqual(cells.shape[1], 6)
        self.assertAlmostEqual(report["rock_footprint_area_m2"], frame["projected_span_m"]*40)

    def test_rock_extends_upstream_downstream_and_beyond_banks(self):
        contours = [{"chainage": 6, "height": 0.}, {"chainage": 90, "height": -10.},
                    {"chainage": 174, "height": 0.}]
        _, _, frame = bedrock_geometry(contours, self.concrete)
        origin = np.array(frame["section_origin_xy_m"])
        across = np.array(frame["across_unit_vector_xy"])
        depth = np.array(frame["depth_unit_vector_xy"])
        local = np.array([[40., 10.], [80., 10.], [80., 20.], [40., 20.],
                          [0., 0.], [frame["projected_span_m"], 40.]])
        plan = origin + local[:, :1] * across + local[:, 1:] * depth
        concrete = np.column_stack((plan, [-20., -21., -22., -23., 0., 0.]))
        points, _, report = seated_bedrock_geometry(
            contours, concrete, [(0, 1, 2, 3)], stream_margin_m=20., bank_margin_m=5.,
            element_size_m=50.,
        )
        span = frame["projected_span_m"]
        np.testing.assert_allclose(report["rock_across_limits_m"], [-5., span + 5.])
        np.testing.assert_allclose(report["rock_depth_limits_m"], [-20., 60.])
        self.assertAlmostEqual(report["rock_footprint_area_m2"], (span + 10.) * 80.)
        top = points[len(points)//2:]
        top_local = np.column_stack(((top[:, :2] - origin) @ across,
                                     (top[:, :2] - origin) @ depth))
        bank = np.isclose(top_local[:, 0], -5.) | np.isclose(top_local[:, 0], span + 5.)
        self.assertTrue(np.allclose(top[bank, 2], 0.))
        upstream = np.isclose(top_local[:, 1], -20.)
        middle = upstream & np.isclose(top_local[:, 0], span / 2, atol=1.)
        self.assertTrue(np.any(middle))
        self.assertTrue(np.all(top[middle, 2] < 0.))
        with self.assertRaises(ValueError):
            seated_bedrock_geometry(contours, concrete, [(0, 1, 2, 3)], stream_margin_m=-1.)


if __name__ == "__main__":
    unittest.main()
